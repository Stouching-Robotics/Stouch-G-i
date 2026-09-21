#!/usr/bin/env python3
"""tactile_right_hand_usb_gui.py - real-time tactile display for the STM32 glove (right hand)

The display is copied verbatim from the ``/api/hand`` rendering in
/touch/glove-sensor/server.py (i.e. the tactile view of that HTML page), with the
data source replaced by a live USB CDC stream:

    top row: thumb/index/middle/ring/pinky, each a 3-row x 4-column pressure grid
    bottom row: palm, 15 rows x 5 columns
    each cell is colored with Viridis stretched by the current frame's maximum,
    with the value drawn inside the cell and the part name below
    background 18, cell border (70,70,70), matching the original HTML

Data pipeline (measured ~70 fps):
    STM32 USB CDC -> sdk.compat.tactile.TactileStream.frames()
        -> TactileFrame.samples (16x16 float32 ADC)
        -> TactilePreprocessor zeroing/de-drift (no noise gating, keeps the HTML's original normalization)
        -> render_hand tiled view

Keys:
    C  re-zero (do not touch while calibrating)      Q/ESC  quit
    M  mirror flip (in the HTML the left glove is mirrored by default, right is not)

Usage:
    python pc/tactile_right_hand_usb_gui.py
    python pc/tactile_right_hand_usb_gui.py --scale 2 --serial-port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
def _sdk_root(start):
    probe = start
    while True:
        if os.path.isdir(os.path.join(probe, "algorithm")):
            return probe
        parent = os.path.dirname(probe)
        if parent == probe:
            raise RuntimeError("cannot locate SDK root")
        probe = parent

_PROJECT_ROOT = _sdk_root(os.path.dirname(os.path.abspath(__file__)))
if getattr(sys, "frozen", False):
    _PROJECT_ROOT = sys._MEIPASS
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

from glove_io.tactile_processing import TactilePreprocessor
from sdk import TactileStream

CALIBRATION_FRAMES = 100


# Shared plaintext renderer (see gui/rendering/tactile.py).
from gui.rendering.tactile import (  # noqa: E402
    GRID_H, GRID_W, HAND_H, HAND_W, render_grid, render_hand)


class UsbPressureReceiver:
    """USB CDC tactile receiver thread.  Public interface matches tactile_hand_gui.Receiver."""

    daemon = True

    def __init__(self, serial_port, usb_vid, usb_pid, config, stop_event,
                 side="right"):
        self.stop_event = stop_event
        self.serial_port = serial_port
        self.usb_vid = usb_vid
        self.usb_pid = usb_pid
        self.status = f"connecting {serial_port or 'auto USB CDC'}"
        self.lock = threading.Lock()
        self.frame = None
        self.frame_times = deque(maxlen=60)
        self.n_frames = 0
        self.n_bad = 0
        self._channel_to_hand = self._load_channel_map(config, side)
        self._thread = threading.Thread(target=self._run, name="usb-pressure", daemon=True)

    @staticmethod
    def _load_channel_map(config: str | None, side: str = "right") -> list[int]:
        fallback = list(range(16))
        if not config:
            return fallback
        try:
            with open(config, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            mapping = data.get("channel_to_hand")
            if mapping is None:
                mapping = (data.get(f"{side}_hand_config") or {}).get("channel_to_hand")
            if (isinstance(mapping, list) and len(mapping) == 16
                    and sorted(mapping) == list(range(16))):
                return [int(v) for v in mapping]
        except Exception:
            pass
        return fallback

    @property
    def fps(self):
        with self.lock:
            t = list(self.frame_times)
        if len(t) < 2:
            return 0.0
        span = t[-1] - t[0]
        return (len(t) - 1) / span if span > 0 else 0.0

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def start(self):
        self._thread.start()

    def stop(self, timeout_s: float = 1.0):
        self.stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout_s)))

    def _run(self):
        stream = TactileStream(
            serial_port=self.serial_port,
            usb_vid=self.usb_vid,
            usb_pid=self.usb_pid,
        )
        try:
            stream.start()
            while not self.stop_event.is_set():
                frames = stream.poll()
                if not frames:
                    self.stop_event.wait(0.005)
                    continue
                for frame in frames:
                    with self.lock:
                        self.frame = frame.samples
                        self.frame_times.append(time.time())
                        self.n_frames += 1
                    self.status = f"USB CDC {self.fps:.0f} fps"
        except Exception as exc:  # noqa: BLE001
            self.status = f"USB error: {exc}"
        else:
            self.status = "stopped"
        finally:
            stream.stop()


class HtmlHandView:
    """HTML tiled hand view + zeroing/denoising chain; ports /api/hand normalization per frame."""

    def __init__(self, receiver, args):
        self.receiver = receiver
        self.side = args.side
        self.mirror = args.side == "left"
        # Zeroing/de-drift only; no noise gating or isolated-point filtering -- matches the HTML's raw normalization
        self.pre = TactilePreprocessor(
            base_gate=0.0,
            dynamic_noise_ratio=0.0,
            temporal_smooth=0.15,
            spatial_filter=False,
            calibration_frames=args.calibration_frames,
            bypass_gates=False,
        )
        self.scale = max(1, int(args.scale))
        side_label = "Left" if args.side == "left" else "Right"
        self.window_name = f"STM32 glove - Tactile {side_label} Hand + 16x16"
        # Combined canvas: the six-region hand view on the left, the full
        # 16x16 grid (every cell, with indices) on the right.
        self.canvas_w = HAND_W + GRID_W + 8
        self.canvas_h = max(HAND_H, GRID_H)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name,
                         self.canvas_w * self.scale, self.canvas_h * self.scale)

    def _compose(self, hand: np.ndarray | None, grid: np.ndarray | None) -> np.ndarray:
        """Assemble the hand panel and the full-matrix grid side by side."""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), 18, np.uint8)
        if hand is not None:
            y0 = (self.canvas_h - HAND_H) // 2
            canvas[y0:y0 + HAND_H, :HAND_W] = hand
        if grid is not None:
            canvas[:, HAND_W + 8:] = grid
        return canvas

    def _hud(self, frame: np.ndarray, text: str, color=(0, 200, 255)):
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)

    def run(self, exit_after: float = 0.0):
        started = time.time()
        peak = 0.0
        while True:
            raw = self.receiver.latest()
            if raw is None:
                frame = self._compose(None, None)
                self._hud(frame, f"Waiting for tactile frames... {self.receiver.status}")
            else:
                processed, peak = self.pre.process(raw)
                if processed is None:
                    frame = self._compose(None, None)
                    self._hud(frame,
                              f"Calibrating... ({self.pre.calibration_progress[0]}/"
                              f"{self.pre.calibration_progress[1]}) DO NOT TOUCH",
                              (0, 0, 255))
                else:
                    hand = render_hand(processed, mirror=self.mirror, side=self.side)
                    grid = render_grid(processed)
                    frame = self._compose(hand, grid)

            if self.scale > 1:
                show = cv2.resize(frame, (HAND_W * self.scale, HAND_H * self.scale),
                                  interpolation=cv2.INTER_NEAREST)
            else:
                show = frame

            cv2.imshow(self.window_name, show)
            cv2.setWindowTitle(
                self.window_name,
                f"{self.window_name} | {self.receiver.fps:.1f} fps | "
                f"{self.receiver.n_frames} frames | max {peak:.0f}")

            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("c"), ord("C")):
                self.pre.start_calibration()
            elif key in (ord("m"), ord("M")):
                self.mirror = not self.mirror

            if exit_after and time.time() - started > exit_after:
                break
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:  # Qt backend raises when the window is destroyed externally; equivalent to it being closed
                break
        cv2.destroyAllWindows()


def main(argv):
    parser = argparse.ArgumentParser(
        description="STM32 glove tactile live view over the public USB API")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--serial-port", help="STM32 CDC port; default: auto-discover")
    parser.add_argument("--usb-vid", type=lambda s: int(s, 0), default=0x0483)
    parser.add_argument("--usb-pid", type=lambda s: int(s, 0), default=0x5740)
    parser.add_argument("--config", help="Legacy config path; sensor mapping is not used")
    parser.add_argument("--scale", type=int, default=2,
                        help="Display scale; native panel is 800x220")
    parser.add_argument("--calibration-frames", type=int, default=CALIBRATION_FRAMES)
    parser.add_argument("--exit-after", type=float, default=0.0,
                        help="Exit automatically after N seconds")
    args = parser.parse_args(argv)

    stop_event = threading.Event()
    receiver = UsbPressureReceiver(
        args.serial_port, args.usb_vid, args.usb_pid,
        args.config or os.path.join(_PROJECT_ROOT, "config", "config.json"),
        stop_event, args.side,
    )
    receiver.start()

    view = HtmlHandView(receiver, args)
    try:
        view.run(args.exit_after)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        print(f"Received {receiver.n_frames} frames at {receiver.fps:.1f} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
