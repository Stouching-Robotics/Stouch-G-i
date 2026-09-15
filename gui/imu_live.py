#!/usr/bin/env python3
"""imu_live.py - live terminal monitor for 16 IMUs (STM32 USB CDC).

The STM32G431 firmware streams fixed 128-byte Q14 quaternion frames over the
USB CDC virtual serial port (each frame holds a snapshot of all 16 IMUs; see
stm32_demo/USB_PROTOCOL.md type 0x02).  This tool reads **raw physical-channel
quaternions** through the public ``runtime.RawImuStream.frames()`` interface
(no remapping, no filtering, no axis correction) and refreshes a table in real
time: each IMU's W/X/Y/Z quaternion, norm, Euler angles and validity flag.

Compared to ESP_Glove1.7, the data interface switched from WebSocket(:18890) +
protobuf to the STM32 USB CDC serial port, so:
  - --host/--port (network) are gone; --port is now the serial port,
    auto-scanned by default with VID:0483 PID:5740
  - quaternions are int16 Q14 fixed-point (value / 16384)
  - each frame is a 16-IMU snapshot; valid comes from the frame flags bit mask
    (bit i = physical channel i)
  - the MANO joint mapping comes from right_hand_config.channel_to_hand in
    config.json; otherwise DEFAULT_CHANNEL_TO_HAND in modules/hand/usb_cdc.py
    is used (display only)

Usage:
    python pc/imu_live.py                       # auto-scan for STM32 CDC port
    python pc/imu_live.py --port /dev/ttyACM0   # specify the port (COMx on Windows)
    python pc/imu_live.py --list                # list all serial ports and exit
    python pc/imu_live.py --raw                 # show firmware physical channel ids, no MANO remapping
    python pc/imu_live.py --csv imu_log.csv     # also write to disk
    python pc/imu_live.py --refresh 0.5         # table refresh interval (s)

Ctrl-C exits.  ANSI colors and screen clearing are disabled automatically when
output is piped or redirected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time

import numpy as np
from pathlib import Path

def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys._MEIPASS)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import usb_cdc as uc  # noqa: E402
from runtime import RawImuStream  # noqa: E402

# Firmware physical channel -> MANO joint id (physical channels 0..15; matches
# DEFAULT_CHANNEL_TO_HAND in modules/hand/usb_cdc.py / config.json's channel_to_hand)
# MANO joint id -> body-part name
MANO_NAMES = {
    0: "Wrist", 1: "Index MCP", 2: "Index DIP", 3: "Index TIP",
    4: "Middle MCP", 5: "Middle DIP", 6: "Middle TIP",
    7: "Ring MCP", 8: "Ring DIP", 9: "Ring TIP",
    10: "Pinky MCP", 11: "Pinky DIP", 12: "Pinky TIP",
    13: "Thumb MCP", 14: "Thumb DIP", 15: "Thumb TIP",
}

# ANSI (enabled only on a tty)
ANSI = {
    "reset": "\033[0m", "dim": "\033[2m", "green": "\033[92m",
    "yellow": "\033[93m", "red": "\033[91m", "cyan": "\033[96m",
    "bold": "\033[1m",
}
_C = {k: (v if sys.stdout.isatty() else "") for k, v in ANSI.items()}
_CLEAR = "\x1b[H\x1b[2J" if sys.stdout.isatty() else ""

# Fixed Q14 frame payload byte count (only for approximate throughput estimation; real traffic includes header/CRC)
_Q14_PAYLOAD_BYTES = 128


def quat_norm_and_euler(w, x, y, z):
    """Return (norm, (roll, pitch, yaw)) in degrees; angles are None for an invalid quaternion."""
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-6:
        return norm, (None, None, None)
    # scipy-style [x,y,z,w], in degrees
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    sinp = 2 * (w * y - z * x)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return norm, (roll, pitch, yaw)


def load_channel_to_hand(config_path: Path | None = None) -> tuple[int, ...]:
    """Read channel_to_hand from config.json; fall back to the firmware default mapping when missing/invalid."""
    cfg_path = config_path or (PROJECT_ROOT / "config" / "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
        hand = cfg.get("right_hand_config") or cfg.get("left_hand_config") or {}
        mapping = hand.get("channel_to_hand")
        if mapping is not None:
            return uc.validate_channel_to_hand(mapping)
    except Exception:
        pass
    return uc.DEFAULT_CHANNEL_TO_HAND


def list_serial_ports() -> None:
    for device, desc, vid, pid in uc.list_serial_ports():
        vidpid = f"VID:{vid:04X}:PID:{pid:04X}" if vid else "N/A"
        print(f"  {device} - {desc} - {vidpid}")


def render(channel_to_hand, raw_mode, latest, n_frames, n_missing,
           fps, started) -> None:
    """Redraw the whole screen once.  latest: {physical channel: (w, x, y, z, valid)}"""
    lines = []
    a = lines.append
    hdr = "  ch    " if raw_mode else "phys→MANO"
    a(f"{_C['bold']}STM32 IMU Live — 16-channel real-time monitor{_C['reset']}   "
      f"{'raw physical channels' if raw_mode else 'channel_to_hand applied'}")
    a("")
    a(f"Overall {_C['cyan']}{fps:5.1f} fps{_C['reset']} | "
      f"{len(latest):>2}/16 online | running {time.time()-started:.0f}s | "
      f"frames {n_frames} | missing {n_missing}")
    a("")
    a(f"  {hdr}   part      W        X        Y        Z    |q|   "
      f"roll   pitch    yaw  valid")
    a("-" * 88)
    for ch in sorted(latest.keys()):
        w, x, y, z, valid = latest[ch]
        mano = channel_to_hand[ch] if not raw_mode else ch
        name = MANO_NAMES.get(mano, str(mano)) if not raw_mode else "-"
        ch_id = f"{ch:02d}→{mano:02d}" if not raw_mode else f"ch{ch:02d}"
        norm, (r, pi, ya) = quat_norm_and_euler(w, x, y, z)
        f6 = lambda v: "--" if v is None else f"{v:6.1f}"
        valid_txt = f"{_C['green']}√{_C['reset']}" if valid else f"{_C['red']}×{_C['reset']}"
        color = "" if valid else _C['dim']
        a(f"{color}{ch_id:>7} {name:<9} "
          f"{w:6.3f} {x:6.3f} {y:6.3f} {z:6.3f} "
          f"{norm:5.3f} {f6(r):>6} {f6(pi):>6} {f6(ya):>6}  {valid_txt}{_C['reset']}")
    a("-" * 88)
    n_online = sum(1 for v in latest.values() if v[4])
    a(f"{_C['dim']}√ {n_online}/16 valid; snapshot of the latest frame — the firmware only sends quat+timestamp+valid,"
      f" Euler angles and norm are derived here. × rows keep the previous valid values. Ctrl-C exits.{_C['reset']}")
    sys.stdout.write(_CLEAR + "\n".join(line + "\033[K" for line in lines) + "\033[J\n")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None, help="serial port (COMx on Windows); auto-scan if omitted")
    ap.add_argument("--baud", type=int, default=115200,
                    help="kept for compatibility; STM32 CDC firmware is fixed at 115200, the SDK receive thread ignores this")
    ap.add_argument("--list", action="store_true", help="list all serial ports and exit")
    ap.add_argument("--refresh", type=float, default=1.0, help="refresh interval (seconds)")
    ap.add_argument("--raw", action="store_true", help="show firmware physical channel ids, no MANO remapping")
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    ap.add_argument("--config", default=None, help="config.json path (default: project root)")
    ap.add_argument("--exit-after", type=float, default=0.0,
                    help="exit automatically after N seconds (for acceptance; 0 = keep running)")
    args = ap.parse_args()

    if args.list:
        list_serial_ports()
        return 0

    channel_to_hand = load_channel_to_hand(
        Path(args.config) if args.config else None)

    port = args.port or uc.find_stm32_cdc_port()
    if port is None:
        print("[!] No STM32 CDC device found (VID:0483 PID:5740). Confirm the USB is plugged in,"
              " or specify a port with --port (see --list)")
        return 1

    print(f"Connecting {port} ... (Ctrl-C to exit)")
    imu = RawImuStream(serial_port=port)
    imu.start()
    print(f"Connected. {args.refresh}s refresh, "
          f"{'' if args.raw else 'MANO remapping applied'} (raw values via the runtime.compat.imu interface)")

    csv_fp = None
    if args.csv:
        csv_fp = open(args.csv, "w", newline="", encoding="utf-8")
        cw = csv.writer(csv_fp)
        cw.writerow(["recv_time", "phys_ch", "mano_id", "quat_w", "quat_x",
                     "quat_y", "quat_z", "norm", "valid"])

    latest = {}
    last_valid = {}  # {physical channel: (w,x,y,z)} last valid values, kept displayed for invalid frames
    n_frames = 0
    started = time.time()
    last_render = 0.0
    frame_times = [started]
    try:
        for frame in imu.frames():
            now = time.time()
            xyz_w = frame.quaternions_xyzw  # (16,4) x/y/z/w, raw physical-channel values
            for ch in range(xyz_w.shape[0]):
                w = float(xyz_w[ch, 3])
                x = float(xyz_w[ch, 0])
                y = float(xyz_w[ch, 1])
                z = float(xyz_w[ch, 2])
                valid = bool(frame.present_mask[ch])
                # Firmware sends [0,0,0,0] for invalid channels; keep showing the
                # last valid value and only mark valid as x, so a periodic dropout
                # does not make a still hand look like it is moving.
                if valid:
                    last_valid[ch] = (w, x, y, z)
                else:
                    w, x, y, z = last_valid.get(ch, (0.0, 0.0, 0.0, 0.0))
                latest[ch] = (w, x, y, z, valid)
                if csv_fp:
                    norm, _ = quat_norm_and_euler(w, x, y, z)
                    cw.writerow(
                        [f"{frame.host_timestamp_us / 1e6:.6f}", ch, channel_to_hand[ch],
                         f"{w:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                         f"{norm:.4f}", int(valid)])
            n_frames += 1
            frame_times.append(now)
            if len(frame_times) > 120:
                frame_times.pop(0)

            if now - last_render < args.refresh:
                continue
            last_render = now
            if len(frame_times) >= 2:
                fps = (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-6)
            else:
                fps = 0.0
            render(channel_to_hand, args.raw, latest, n_frames,
                   int(np.count_nonzero(~frame.valid_mask)), fps, started)
            if args.exit_after and now - started >= args.exit_after:
                break
    except KeyboardInterrupt:
        print("\nExited")
    finally:
        imu.stop()
        if csv_fp:
            csv_fp.close()
            print(f"CSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
