"""Read raw IMUs, run live pose in a worker, or simulate a slow consumer."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from math import isfinite
from pathlib import Path
from threading import Condition
from time import sleep, time_ns

import numpy as np

import sdk


def read_frames(args: argparse.Namespace) -> Generator[sdk.RawImuFrame, None, None]:
    with sdk.open_raw_glove(args.side, link=args.link, port=args.port,
                            serial_number=args.serial_number) as glove:
        for _ in range(args.frames):
            yield glove.read(timeout=args.timeout)


def simulated_frames(count: int) -> Generator[sdk.RawImuFrame, None, None]:
    """Generate roughly 100 Hz of raw frames without hardware or calibration."""
    xyzw = np.tile([0.0, 0.0, 0.0, 1.0], (16, 1))
    mask = np.ones(16, dtype=bool)
    host_start_us = time_ns() // 1000
    for sequence in range(1, count + 1):
        sleep(0.01)
        device_us = sequence * 10_000
        yield sdk.RawImuFrame(sequence, device_us, host_start_us + device_us, xyzw, mask, mask)


def run_bounded(
    frames: Generator[sdk.RawImuFrame, None, None],
    consume: Callable[[sdk.RawImuFrame], None],
    queue_size: int,
) -> tuple[int, int, int]:
    """Return counts of reads, processed frames, and dropped pending frames."""
    pending: deque[sdk.RawImuFrame] = deque(maxlen=queue_size)
    available = Condition()
    finished = False

    def worker() -> int:
        processed = 0
        while True:
            with available:
                available.wait_for(lambda: pending or finished)
                if not pending:
                    return processed
                raw = pending.popleft()
            consume(raw)
            processed += 1

    read = dropped = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(worker)
        try:
            with closing(frames):
                for raw in frames:
                    if result.done():
                        result.result()  # Propagate a worker error to the reader.
                    with available:
                        if len(pending) == queue_size:
                            dropped += 1  # append discards the oldest pending frame
                        pending.append(raw)
                        available.notify()
                    read += 1
        except BaseException:
            with available:
                pending.clear()
            raise
        finally:
            with available:
                finished = True
                available.notify()
    return read, result.result(), dropped


def show_raw(args: argparse.Namespace) -> None:
    for raw in read_frames(args):
        pass  # Drain the stream. Report only the final sample.
    print(f"side={args.side} read={args.frames} sequence={raw.sequence} "
          f"device_us={raw.device_timestamp_us} host_us={raw.host_timestamp_us} "
          f"valid={np.count_nonzero(raw.valid_mask)}/16")
    print(raw.quaternions_xyzw)


def show_pose(args: argparse.Namespace) -> None:
    processor = sdk.open_pose_processor(args.side, args.calibration)
    latest = None
    ready = 0

    def consume(raw: sdk.RawImuFrame) -> None:
        nonlocal latest, ready
        frame = processor.process(raw)
        details = frame.status.details
        if not (details.get("warming_up") or details.get("warmup_timed_out")):
            latest = frame
            ready += 1
        if args.delay_ms:
            sleep(args.delay_ms / 1000)

    read, processed, dropped = run_bounded(read_frames(args), consume, args.queue_size)
    print(f"read={read} processed={processed} dropped_by_app={dropped} ready_poses={ready}")
    if latest is None:
        print("No pose passed the warmup checks in this run.")
    else:
        print(f"side={latest.side} sequence={latest.sequence} "
              f"joint_xyzw={latest.joint_xyzw.shape} keypoints_m={latest.joints_m.shape}")


def show_demo(args: argparse.Namespace) -> None:
    def consume(_raw: sdk.RawImuFrame) -> None:
        sleep(args.delay_ms / 1000)

    read, processed, dropped = run_bounded(simulated_frames(args.frames), consume, args.queue_size)
    print(f"simulated={read} processed={processed} dropped_by_app={dropped}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    raw = commands.add_parser("raw", help="read raw IMU frames without a wearing calibration")
    pose = commands.add_parser("pose", help="process live pose in a separate worker")
    demo = commands.add_parser("demo", help="simulate overload without a glove or pose solver")

    for command in (raw, pose):
        command.add_argument("--side", choices=("left", "right"), default="right")
        command.add_argument("--link", choices=("auto", "usb", "bluetooth"), default="auto")
        command.add_argument("--port", help="pin the session to this glove port")
        command.add_argument("--serial-number", help="select a glove by serial number")
        command.add_argument("--timeout", type=float, default=5.0, help="seconds per raw read")

    raw.add_argument("--frames", type=int, default=1)
    pose.add_argument("--calibration", type=Path, required=True)
    pose.add_argument("--frames", type=int, default=300)
    demo.add_argument("--frames", type=int, default=100)
    for command, delay_ms in ((pose, 0.0), (demo, 50.0)):
        command.add_argument("--queue-size", type=int, default=8, help="maximum pending frames")
        command.add_argument("--delay-ms", type=float, default=delay_ms, help="extra work per frame")

    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.mode != "demo" and (not isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be finite and positive")
    if args.mode != "raw":
        if args.queue_size < 1:
            parser.error("--queue-size must be at least 1")
        if not isfinite(args.delay_ms) or args.delay_ms < 0:
            parser.error("--delay-ms must be finite and non-negative")

    try:
        if args.mode == "raw":
            show_raw(args)
        elif args.mode == "pose":
            show_pose(args)
        else:
            show_demo(args)
    except sdk.GloveSdkError as exc:
        parser.exit(1, f"SDK error: {exc}\n")
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
