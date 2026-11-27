"""Read Friston geometry at rest, or display samples from one or both hands."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import monotonic

import numpy as np

import sdk


def show_rest() -> None:
    for side in ("left", "right"):
        skeleton = sdk.load_skeleton(side)
        rest = skeleton.rest_local_xyzw
        points = skeleton.keypoints_m(rest)
        vertices = skeleton.mesh_vertices_m(rest)
        print(f"{side}: {len(points)} keypoints, {len(vertices)} vertices, "
              f"{len(skeleton.faces)} triangles")


def prepare_force(glove: sdk.Glove) -> None:
    """Load the device profile and measure its connection's unloaded baseline."""
    connection_id = glove.force_calibration_status["connection_id"]
    if connection_id is None:
        glove.read(timeout=5.0)
        connection_id = glove.force_calibration_status["connection_id"]
        if connection_id is None:
            raise RuntimeError("The glove connection is not ready for force calibration")
    profile = glove.read_force_calibration()
    if not profile.ok:
        raise RuntimeError(f"Could not read device force parameters: {profile.status} {profile.error}")
    if profile.calibration["model"] != "delta_adc":
        return

    print("Keep the glove unloaded while the ADC baseline is collected.")
    deadline = monotonic() + 5.0
    scans = []
    last_sequence = None
    while len(scans) < 32:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Fewer than 32 fresh, finite ADC scans arrived for the baseline")
        frame = glove.read(timeout=remaining)
        if frame.sequence == last_sequence:
            continue
        last_sequence = frame.sequence
        if frame.force_fresh and frame.tactile_adc is not None and np.isfinite(frame.tactile_adc).all():
            scans.append(frame.tactile_adc.copy())
    glove.set_force_baseline(np.median(scans, axis=0), connection_id=connection_id)
    print("Baseline stored. Apply pressure to observe force in newtons.")


def show_frame(frame: sdk.HandFrame, last_sample: dict[str, tuple[int, int]]) -> None:
    sample = (frame.sequence, frame.timestamp_us)
    if last_sample.get(frame.side) == sample:
        return
    last_sample[frame.side] = sample
    if frame.status.details.get("warming_up"):
        print(f"{frame.side}:{frame.sequence} warming up")
        return

    skeleton = sdk.load_skeleton(frame.side)
    points = skeleton.keypoints_m(frame.joint_xyzw)
    vertices = skeleton.mesh_vertices_m(frame.joint_xyzw)
    print(f"{frame.side}:{frame.sequence} wrist_m={points[0].round(4)} "
          f"mesh={vertices.shape}, force={frame.force_source}/{frame.force_status}")
    if frame.force_fresh and frame.tactile_adc is not None:
        finite_force = [] if frame.force_n is None else frame.force_n[np.isfinite(frame.force_n)]
        peak_newtons = f"{max(finite_force):.3f}" if len(finite_force) else "unavailable"
        print(f"  new scan ADC={frame.tactile_adc.shape} peak_N={peak_newtons} "
              f"tactile_fps={frame.tactile_fps:.1f}")
    if frame.force_error:
        print(f"  {frame.force_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("rest", help="calculate both rest poses without hardware")
    single = commands.add_parser("single", help="read one live glove")
    single.add_argument("--side", choices=("left", "right"), default="right")
    single.add_argument("--calibration", type=Path, required=True)
    single.add_argument("--force", action="store_true", help="load device force parameters and measure an unloaded baseline")
    single.add_argument("--frames", type=int, default=10)
    both = commands.add_parser("bimanual", help="read both live gloves")
    both.add_argument("--left-calibration", type=Path, required=True)
    both.add_argument("--right-calibration", type=Path, required=True)
    both.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "rest":
        show_rest()
        return
    if args.frames < 1:
        parser.error("--frames must be at least 1")

    last_sample: dict[str, tuple[int, int]] = {}
    if args.mode == "single":
        with sdk.open_glove(args.side, calibration=args.calibration) as glove:
            if args.force:
                prepare_force(glove)
            for _ in range(args.frames):
                show_frame(glove.read(timeout=1.0), last_sample)
    else:
        with sdk.open_bimanual(left_calibration=args.left_calibration,
                               right_calibration=args.right_calibration) as gloves:
            for _ in range(args.frames):
                pair = gloves.read(timeout=1.0)
                show_frame(pair.left, last_sample)
                show_frame(pair.right, last_sample)


if __name__ == "__main__":
    main()
