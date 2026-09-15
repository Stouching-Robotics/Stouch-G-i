#!/usr/bin/env python3
"""Combined terminal monitor for the left and right STM32 gloves."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

# NB: check frozen BEFORE _sdk_root() — frozen builds have no algorithm
# directory (it lives in the PYZ archive), so _sdk_root always fails there.
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys._MEIPASS)
else:
    PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.usb_cdc import (  # noqa: E402
    DEFAULT_CHANNEL_TO_HAND, remap_physical_to_hand,
    validate_channel_to_hand)
from runtime import DeviceManager, RawImuStream  # noqa: E402

DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "glove_devices.json"


def _enable_windows_vt() -> None:
    """Enable smooth console rendering on legacy Windows consoles.

    Modern terminals handle the escape sequences below natively, but legacy
    console hosts render the raw bytes as garbage text on every repaint,
    which looks like heavy stutter.  This mirrors what colorama does and is
    a no-op everywhere else.  Also bumps the Windows timer resolution so the
    per-frame sleeps in the render loop can actually land at 30+ FPS.
    """

    if os.name != "nt":
        return
    import ctypes

    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return
        if not (mode.value & 0x0004):  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except (AttributeError, OSError):
        pass
    try:
        # Windows quantizes timed waits to ~15.6 ms by default, which makes
        # sub-33 ms frame pacing lumpy; ask for 1 ms timer resolution so the
        # GUI refresh rate actually lands near the requested value.  Windows
        # restores the default automatically when this process exits.
        ctypes.windll.winmm.timeBeginPeriod(1)
    except (AttributeError, OSError):
        pass


def _euler_text(quat_xyzw):
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-6:
        return " invalid "
    sinr = 2 * (w*x + y*z); cosr = 1 - 2 * (x*x + y*y)
    sinp = max(-1.0, min(1.0, 2 * (w*y - z*x)))
    siny = 2 * (w*z + x*y); cosy = 1 - 2 * (y*y + z*z)
    return (f"{math.degrees(math.atan2(sinr, cosr)):6.1f}/"
            f"{math.degrees(math.asin(sinp)):6.1f}/"
            f"{math.degrees(math.atan2(siny, cosy)):6.1f}")


class SideReader(threading.Thread):
    def __init__(self, side, port, mapping, stop_event):
        super().__init__(name=f"imu-{side}", daemon=True)
        self.side = side
        self.port = port
        self.mapping = mapping
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self.quaternions = np.full((16, 4), np.nan, dtype=float)
        self.valid = np.zeros(16, dtype=bool)
        self.frames = 0
        self.error = None
        self.started = time.monotonic()

    @property
    def fps(self):
        return self.frames / max(time.monotonic() - self.started, 1e-6)

    def snapshot(self):
        with self.lock:
            return self.quaternions.copy(), self.valid.copy(), self.frames, self.error

    def run(self):
        stream = RawImuStream(self.port)
        try:
            stream.start()
            while not self.stop_event.is_set():
                frames = stream.poll()
                if not frames:
                    self.stop_event.wait(0.005)
                    continue
                # The display repaints at most once per --refresh interval,
                # so remapping one frame per poll is enough at full stream
                # rate; the rest just contribute to the frame counter.
                frame = frames[-1]
                valid_bits = sum(
                    (1 << index) for index, valid in
                    enumerate(frame.valid_mask) if valid)
                mano, valid = remap_physical_to_hand(
                    frame.quaternions_xyzw, valid_bits, self.mapping)
                with self.lock:
                    self.quaternions = mano
                    self.valid = valid
                    self.frames += len(frames)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
            self.stop_event.set()
        finally:
            stream.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--refresh", type=float, default=1.0 / 30.0,
        help="minimum GUI refresh interval in seconds; default 1/30 (~30 "
             "FPS), 0 renders back-to-back as fast as the terminal allows")
    parser.add_argument("--exit-after", type=float, default=0.0)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    bindings = DeviceManager(args.registry).validate_bimanual()
    ports = {"left": bindings.left_port, "right": bindings.right_port}
    serials = {"left": bindings.left_serial, "right": bindings.right_serial}
    mappings = {
        side: validate_channel_to_hand(
            (payload.get(side) or {}).get(
                "channel_to_hand", DEFAULT_CHANNEL_TO_HAND))
        for side in ("left", "right")
    }
    stop_event = threading.Event()
    readers = {
        side: SideReader(side, ports[side], mappings[side], stop_event)
        for side in ("left", "right")
    }
    for reader in readers.values():
        reader.start()
    _enable_windows_vt()
    start = time.monotonic()
    is_tty = sys.stdout.isatty()
    if is_tty:
        # Hide the blinking cursor: with the screen cleared every frame it
        # otherwise leaves a visible flickering artifact at the write position.
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
    period = max(args.refresh, 0.0)
    last_text = None
    prev_start = None
    gui_fps = 0.0
    try:
        while not stop_event.is_set():
            frame_start = time.monotonic()
            if prev_start is not None:
                gui_fps = 1.0 / max(frame_start - prev_start, 1e-6)
            prev_start = frame_start
            snapshots = {side: readers[side].snapshot() for side in readers}
            lq, lv, _, le = snapshots["left"]
            rq, rv, _, re = snapshots["right"]
            # Build the whole screen as one string and write it with a single
            # flush: ~20 separate print() calls per refresh make Windows
            # consoles visibly flicker/stutter between partial frames.
            lines = []
            if is_tty:
                lines.append("\033[H\033[J")
            lines.append(
                f"STM32 bimanual IMU — roll/pitch/yaw (deg)  "
                f"[display {gui_fps:5.1f} fps]")
            lines.append(
                f"LEFT  {ports['left']} serial={serials['left']} "
                f"{readers['left'].fps:5.1f}fps | RIGHT {ports['right']} "
                f"serial={serials['right']} {readers['right'].fps:5.1f}fps")
            lines.append(
                " ID |       LEFT R/P/Y       ok ||       RIGHT R/P/Y      ok")
            for index in range(16):
                lines.append(
                    f"{index:3d} | {_euler_text(lq[index])}  "
                    f"{'Y' if lv[index] else 'N'} "
                    f"|| {_euler_text(rq[index])}  "
                    f"{'Y' if rv[index] else 'N'}")
            text = "\n".join(lines) + "\n"
            # Skip frames whose text is identical to the last one written:
            # even a no-op write costs a full console repaint on Windows.
            if text != last_text:
                sys.stdout.write(text)
                sys.stdout.flush()
                last_text = text
            if le or re:
                raise RuntimeError(f"left={le or 'OK'} right={re or 'OK'}")
            if args.exit_after and time.monotonic() - start >= args.exit_after:
                break
            # Adaptive pacing: sleep only for the time left in the refresh
            # budget, so render cost itself doesn't eat into the frame rate.
            stop_event.wait(max(0.0, period - (time.monotonic() - frame_start)))
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for reader in readers.values():
            reader.join(timeout=1.0)
        if is_tty:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
    if any(readers[side].frames == 0 for side in readers):
        print("[FAIL] at least one hand received no IMU frames", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
