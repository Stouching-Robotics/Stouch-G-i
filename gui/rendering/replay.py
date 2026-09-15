#!/usr/bin/env python3
"""Render recorded 21-keypoint Parquet data as an MP4 or PNG.

The renderer uses a rotating or static perspective camera, per-finger colors,
depth shading, a ground grid, and optional left/right tactile panels.  It only
depends on NumPy, OpenCV, and Polars; no camera hardware is required.

New recordings include ``session.json`` beside the Parquet file.  View,
sample-rate, hand-side, and bimanual layout metadata are restored when present;
older recordings are inferred from their first valid frame.

Example::

    ./glove.sh replay data/session/chunk-000.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import polars as pl
from scipy.spatial.transform import Rotation as R

def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys._MEIPASS)
else:
    PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)
ROOT = PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.live_3d import (  # noqa: E402
    align_left_to_right_reference,
    apply_hand_display_rotation,
    hand_display_basis,
    place_hands_at_wrist_anchors,
)
from gui.live_3d_bimanual import (  # noqa: E402
    fixed_bimanual_display_rotations,
    remap_hand_root_motion,
    stabilize_bimanual_palm_bases,
)
from gui.rendering.tactile_overlay import (  # noqa: E402
    draw_bimanual_tactile_overlay,
)
from glove_io.session import load_session_metadata  # noqa: E402

N_HANDS, N_KPTS = 2, 21
W, H = 1280, 720
FOV_DEG = 50.0                                   # vertical field of view
_COL = "observation.keypoints.hand_3d"
_SM = "observation.keypoints.hand_3d_smoothed"
_TACTILE_COLUMNS = {
    "left": "observation.tactile.left_glove",
    "right": "observation.tactile.right_glove",
}

# MANO 21-keypoint bone connections + groups: 0 thumb 1 index 2 middle 3 ring 4 pinky 5 metacarpal
BONES = [
    (0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),                # thumb
    (0, 5, 1), (5, 6, 1), (6, 7, 1), (7, 8, 1),                # index
    (0, 9, 2), (9, 10, 2), (10, 11, 2), (11, 12, 2),           # middle
    (0, 13, 3), (13, 14, 3), (14, 15, 3), (15, 16, 3),         # ring
    (0, 17, 4), (17, 18, 4), (18, 19, 4), (19, 20, 4),         # pinky
    (5, 9, 5), (9, 13, 5), (13, 17, 5),                        # metacarpal
]
# Per-finger group colors (BGR)
FINGER_BGR = [
    (60, 80, 255),     # 0 thumb   red
    (60, 255, 255),    # 1 index   yellow
    (60, 220, 60),     # 2 middle  green
    (255, 210, 60),    # 3 ring    cyan
    (255, 120, 255),   # 4 pinky   purple
    (210, 210, 210),   # 5 metacarpal white
]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def load(path, smooth) -> tuple[np.ndarray, list[str], list[str]]:
    df = pl.read_parquet(path)
    col = _SM if smooth else _COL
    arr = np.asarray(df[col].to_list(), np.float32).reshape(-1, N_HANDS, N_KPTS, 3)
    order = None
    if "frame_index" in df.columns:
        order = np.argsort(df["frame_index"].to_numpy(), kind="stable")
        arr = arr[order]
    def lab(c):
        values = df[c].to_list() if c in df.columns else [""] * arr.shape[0]
        return [values[index] for index in order] if order is not None else values
    return arr, lab("observation.keypoints.hand_0_label"), \
        lab("observation.keypoints.hand_1_label")


def load_tactile(path) -> dict[str, np.ndarray | None]:
    """Load processed 16x16 tactile frames in recording frame order."""
    df = pl.read_parquet(path)
    order = None
    if "frame_index" in df.columns:
        order = np.argsort(df["frame_index"].to_numpy(), kind="stable")
    result: dict[str, np.ndarray | None] = {}
    for side, column in _TACTILE_COLUMNS.items():
        if column not in df.columns:
            result[side] = None
            continue
        values = np.asarray(df[column].to_numpy(), dtype=np.float32).reshape(
            -1, 16, 16)
        result[side] = values[order] if order is not None else values
    return result


def prepare_bimanual_replay(
        frames: np.ndarray,
        metadata: dict,
        separation_m: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    """Apply the live bimanual display transform without touching source data."""
    source = np.asarray(frames, dtype=np.float32).reshape(-1, 2, 21, 3)
    display = metadata.get("display") if isinstance(metadata, dict) else None
    display = display if isinstance(display, dict) else {}
    saved_targets = display.get("palm_target_bases")
    if saved_targets is None:
        initial = source[0].copy()
        initial[0] = apply_hand_display_rotation(
            initial[0], align_left_to_right_reference(initial[0], initial[1]))
        targets = np.stack([
            hand_display_basis(initial[slot]) for slot in range(2)])
    else:
        targets = np.asarray(saved_targets, dtype=float).reshape(2, 3, 3)

    fixed_matrices = display.get("fixed_rotation_matrices")
    use_dynamic_stabilization = bool(
        display.get("stabilize_palm_bases", fixed_matrices is None))
    if fixed_matrices is not None:
        fixed_rotations = [
            R.from_matrix(matrix)
            for matrix in np.asarray(fixed_matrices, dtype=float).reshape(2, 3, 3)
        ]
    elif not use_dynamic_stabilization:
        fixed_rotations = fixed_bimanual_display_rotations(source[0], targets)
    else:
        fixed_rotations = None
    anchor_axis = -targets[1, :, 0]
    left_root_rotvec_sign = np.asarray(
        display.get("left_root_rotvec_sign", [1.0, 1.0, 1.0]),
        dtype=float).reshape(3)
    right_root_rotvec_sign = np.asarray(
        display.get("right_root_rotvec_sign", [1.0, 1.0, 1.0]),
        dtype=float).reshape(3)

    shown_frames = []
    for frame in source:
        if fixed_rotations is None:
            # Dynamic palm stabilization is a both-hands operation; a frame
            # with a NaN slot (single-hand) is left in its raw pose.
            if np.isfinite(frame).all():
                stable = stabilize_bimanual_palm_bases(frame, targets)
                frame_anchor_axis = None
            else:
                shown_frames.append(frame.copy())
                continue
        else:
            stable = frame.copy()
            for slot, rotation in enumerate(fixed_rotations):
                if rotation is not None and np.isfinite(stable[slot]).all():
                    stable[slot] = apply_hand_display_rotation(
                        stable[slot], rotation)
            # Both hands get the per-frame root-rotation remap, matching the
            # live viewer.  The remap is a handedness/sign correction for the
            # root rotation (it never changes how far each palm axis deviates
            # from its target basis); root drift removal is the stabilization
            # path above.  The per-slot isfinite guard keeps a connected single
            # hand transformed even when the other slot is NaN.
            for slot, signs in enumerate(
                    (left_root_rotvec_sign, right_root_rotvec_sign)):
                if np.isfinite(stable[slot]).all():
                    stable[slot] = remap_hand_root_motion(
                        stable[slot], targets[slot], signs)
            frame_anchor_axis = anchor_axis
        shown_frames.append(place_hands_at_wrist_anchors(
            stable, separation_m=separation_m,
            anchor_axis=frame_anchor_axis))
    return np.asarray(shown_frames, dtype=np.float32), targets


class Camera:
    """Spherical-coordinate camera looking at the origin with perspective projection.
    pan_px is the image offset applied for drag panning.

    Three-axis rotation: yaw about the world vertical axis, elev about the
    horizontal axis, roll about the view (fwd) axis.  Keeps the same math as the
    Camera in pc/glove_21_live_3d.py so offline replay matches the live 3D view
    pixel-for-pixel.
    """

    def __init__(self, yaw_deg, elev_deg, dist, roll_deg=0.0, pan_px=(0.0, 0.0)):
        yaw, elev = np.deg2rad(yaw_deg), np.deg2rad(elev_deg)
        roll = np.deg2rad(roll_deg)
        cp = np.array([dist * np.cos(elev) * np.sin(yaw),
                       dist * np.sin(elev),
                       dist * np.cos(elev) * np.cos(yaw)])
        self.pos = cp
        self.fwd = -cp / np.linalg.norm(cp)
        self.right = np.cross(self.fwd, [0, 1, 0])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        # Roll about the view axis: rotate the image-plane basis by the roll angle
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        r0, u0 = self.right, self.up
        self.right = r0 * cos_r + u0 * sin_r
        self.up = -r0 * sin_r + u0 * cos_r
        self.f = (H / 2) / np.tan(np.deg2rad(FOV_DEG) / 2)
        self.pan_px = [float(pan_px[0]), float(pan_px[1])]

    def project(self, pts3d):
        """(N,3) → (N,2) image coordinates + (N,) depth (positive in front of the camera)."""
        v = pts3d - self.pos
        z = v @ self.fwd
        x = v @ self.right
        y = v @ self.up
        u = self.f * x / z + W / 2
        vv = H / 2 - self.f * y / z
        u += self.pan_px[0]
        vv += self.pan_px[1]
        return np.stack([u, vv], -1), z


def _fit_dist(kpts) -> float:
    """Set the camera distance from the spatial scale of the whole dataset so the hand stays in frame."""
    v = kpts[np.isfinite(kpts).all(axis=-1)]
    if len(v) == 0:
        return 0.5
    r = float(np.abs(v).max())
    return max(0.25, r * 2.6 + 0.1)


def _bg() -> np.ndarray:
    img = np.full((H, W, 3), 12, np.uint8)
    # Light gradient at the top adds depth.
    for y in range(H):
        img[y] = np.full(3, 10 + int(10 * y / H), np.uint8)
    return img


def _grid(img, cam, r: float, basis: np.ndarray | None = None) -> None:
    """Draw a grid in world z=0 or a supplied [x, y, normal] basis."""
    grid_basis = (np.eye(3, dtype=float) if basis is None
                  else np.asarray(basis, dtype=float).reshape(3, 3))
    axis_x, axis_y, axis_z = (
        grid_basis[:, 0], grid_basis[:, 1], grid_basis[:, 2])
    x = [(i / 6) * r for i in range(-6, 7)]
    lines = ([(axis_x * v - axis_y * r,
               axis_x * v + axis_y * r) for v in x]
             + [(-axis_x * r + axis_y * v,
                 axis_x * r + axis_y * v) for v in x])
    for p0, p1 in lines:
        pts, z = cam.project(np.array([p0, p1], np.float32))
        if (z > 0).all():
            cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)),
                     (40, 44, 52), 1, cv2.LINE_AA)
    # Axes (BGR): X red Y green Z blue
    origin = np.zeros((3,), np.float32)
    for axis, col in [(axis_x * r, (40, 60, 255)),
                      (axis_y * r, (40, 255, 60)),
                      (axis_z * r, (255, 60, 40))]:
        pts, z = cam.project(np.stack([origin, axis]))
        if (z > 0).all():
            cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)),
                     col, 2, cv2.LINE_AA)


def _draw_hand(img, cam, kpts, hand_idx) -> None:
    pts, z = cam.project(kpts)                       # (21,2) + (21,)
    vis = np.isfinite(kpts).all(axis=1) & (z > 0)
    # Depth shading (near bright, far dim); fingers slightly brighter than metacarpals
    zmin, zmax = 0.05, 0.6
    lum = np.clip((zmax - z) / (zmax - zmin), 0.35, 1.0)
    for a, b, g in BONES:
        if vis[a] and vis[b]:
            k = 0.5 * (lum[a] + lum[b])
            col = tuple(int(c * k) for c in FINGER_BGR[g])
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     col, 3, cv2.LINE_AA)
    for p, ok, g in zip(pts, vis, range(21)):
        if ok:
            # White dot radius 2, matching the pc/glove_21_live_3d.py live preview (pixel-aligned)
            cv2.circle(img, tuple(p.astype(int)), 2, (255, 255, 255), -1, cv2.LINE_AA)


def render_frame(
        kpts, lab0, lab1, i, n, cam, title, grid_r=None,
        grid_basis: np.ndarray | None = None,
        tactile_frames: dict[str, np.ndarray | None] | None = None,
        tactile_threshold: float = 0.0,
        tactile_scale: float = 0.6) -> np.ndarray:
    img = _bg()
    if grid_r is None:
        grid_r = _fit_dist(kpts) * 0.5  # grid/axes scaled to half the hand span
    _grid(img, cam, grid_r, basis=grid_basis)
    for s in range(N_HANDS):
        _draw_hand(img, cam, kpts[s], s)
    cv2.putText(img, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(img, f"frame {i}/{n - 1}", (W - 240, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1, cv2.LINE_AA)
    for dy, lab in ((66, lab0), (92, lab1)):
        if lab:
            cv2.putText(img, lab, (20, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (170, 170, 170), 1, cv2.LINE_AA)
    if tactile_frames is not None:
        draw_bimanual_tactile_overlay(
            img,
            tactile_frames,
            threshold=tactile_threshold,
            scale=tactile_scale,
        )
    return img


def main(argv=None, required_hands: tuple[str, ...] = (),
         layout_separation: float = 0.0):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Input Parquet path")
    ap.add_argument("--view", default=None, choices=("rotating", "static"),
                    help="Default: static with session.json, otherwise rotating")
    ap.add_argument("--revolutions", type=float, default=2.0, help="Camera revolutions")
    ap.add_argument("--yaw", type=float, default=None,
                    help="Static-view yaw; default: session.json or 180")
    ap.add_argument("--elev", type=float, default=None,
                    help="Elevation; default: session.json or 25")
    ap.add_argument("--roll", type=float, default=None,
                    help="Camera roll in degrees")
    ap.add_argument("--dist", type=float, default=None,
                    help="Camera distance in metres; default: fit recording")
    ap.add_argument("--pan", default=None,
                    help="Image pan in pixels as x,y")
    ap.add_argument("--smooth", action="store_true", help="Use hand_3d_smoothed")
    ap.add_argument("--fps", type=float, default=None,
                    help="Default: session sample rate or 30")
    ap.add_argument("--out", help="Output MP4; default: preview.mp4 beside Parquet")
    ap.add_argument("--frame", type=int, default=None, help="Render only frame N as PNG")
    ap.add_argument("--no-tactile", action="store_true",
                    help="Disable recorded tactile overlays")
    ap.add_argument("--tactile-threshold", type=float, default=0.0,
                    help="Set tactile samples below this value to zero")
    ap.add_argument("--tactile-scale", type=float, default=0.6,
                    help="Tactile-panel scale; default matches bimanual live")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.input):
        raise SystemExit(f"Error: file does not exist: {args.input}")

    metadata = load_session_metadata(args.input)
    kpts, lab0, lab1 = load(args.input, args.smooth)
    tactile = ({"left": None, "right": None}
               if args.no_tactile else load_tactile(args.input))
    present = {
        "left": bool(np.any(np.all(np.isfinite(kpts[:, 0]), axis=-1))),
        "right": bool(np.any(np.all(np.isfinite(kpts[:, 1]), axis=-1))),
    }
    missing_required = [side for side in required_hands if not present.get(side, False)]
    if missing_required:
        raise SystemExit(
            f"Error: required hand slots are missing: {missing_required}; present={present}")

    # The other slot is NaN for single-hand recordings; both slots valid means
    # the recording is unambiguously bimanual, so older bimanual recordings
    # without session.json still get the compatible palm-basis inference.
    is_bimanual = bool(present["left"] and present["right"])
    view_meta = metadata.get("view") if isinstance(metadata.get("view"), dict) else {}
    display_meta = (metadata.get("display")
                    if isinstance(metadata.get("display"), dict) else {})
    separation_m = float(
        display_meta.get("separation_m", layout_separation or 0.30))
    grid_basis = None
    # Apply the live bimanual display transform whenever the recording carries
    # display metadata (written by live_3d_bimanual_auto) -- including single-
    # hand takes, whose other slot is NaN but whose live view was still
    # rotated/anchored.  Older bimanual recordings without metadata keep the
    # both-hands inference via is_bimanual.
    if bool(display_meta) or is_bimanual:
        kpts, targets = prepare_bimanual_replay(
            kpts, metadata, separation_m=separation_m)
        grid_x = -targets[1, :, 0]
        grid_y = targets[1, :, 1]
        grid_z = np.cross(grid_x, grid_y)
        grid_z /= max(float(np.linalg.norm(grid_z)), 1e-9)
        grid_basis = np.stack([grid_x, grid_y, grid_z], axis=1)
        if display_meta.get("palm_target_bases") is None:
            print("[Info] session.json has no palm bases; inferred from first frame")
        motion_mode = ("per-frame palm stabilization"
                       if display_meta.get("stabilize_palm_bases", True)
                       else "fixed startup alignment with root motion preserved")
        print(
            f"Bimanual layout: {separation_m * 100:.1f} cm wrist separation; "
            f"{motion_mode}")

    view_mode = args.view or ("static" if metadata else "rotating")
    yaw_value = float(args.yaw if args.yaw is not None
                      else view_meta.get("yaw", 180.0))
    elev_value = float(args.elev if args.elev is not None
                       else view_meta.get("elev", 25.0))
    roll_value = float(args.roll if args.roll is not None
                       else view_meta.get("roll", 0.0))
    fps_value = float(args.fps if args.fps is not None
                      else metadata.get("sample_fps", 30.0))
    n = kpts.shape[0]
    tactile_valid_frames = sum(
        int(np.isfinite(values).any(axis=(1, 2)).sum())
        for values in tactile.values() if values is not None)
    tactile_nonzero = sum(
        int(np.count_nonzero(np.nan_to_num(values)))
        for values in tactile.values() if values is not None)
    if not args.no_tactile and tactile_valid_frames:
        print(
            f"Tactile replay: {tactile_valid_frames} valid hand-frames, "
            f"{tactile_nonzero} non-zero samples")
    stem = os.path.splitext(os.path.basename(args.input))[0]
    title = f"{stem} ({'smoothed' if args.smooth else 'hand_3d'}, {view_mode})"
    pan_source = args.pan
    if pan_source is None:
        pan_values = view_meta.get("pan_px", [0.0, 0.0])
        pan_source = ",".join(str(value) for value in pan_values)
    try:
        pan = tuple(float(v) for v in pan_source.split(","))
        if len(pan) != 2:
            raise ValueError
    except ValueError:
        raise SystemExit(f"Error: --pan requires x,y; got {pan_source!r}")
    metadata_dist = view_meta.get("dist") if metadata else None
    dist = (args.dist if args.dist is not None
            else float(metadata_dist) if metadata_dist is not None
            else _fit_dist(kpts))
    fixed_dist = args.dist is not None or metadata_dist is not None
    grid_r = max(0.1, dist * 0.5) if fixed_dist else None
    print(f"Loaded {n} frames; camera distance {dist:.3f} m"
          + (" (fixed)" if fixed_dist else " (auto-fit)"))

    def cam_at(i):
        if view_mode == "static":
            yaw = yaw_value
        else:
            yaw = 360.0 * args.revolutions * i / max(n - 1, 1)
        return Camera(yaw, elev_value, dist, roll_deg=roll_value, pan_px=pan)

    def layout_at(frame, camera):
        del camera
        return frame

    def tactile_at(index: int) -> dict[str, np.ndarray | None]:
        return {
            side: (None if values is None else values[index])
            for side, values in tactile.items()
        }

    if args.frame is not None:
        if not 0 <= args.frame < n:
            raise SystemExit(f"Error: --frame must be in 0..{n - 1}")
        camera = cam_at(args.frame)
        img = render_frame(layout_at(kpts[args.frame], camera),
                           lab0[args.frame], lab1[args.frame],
                           args.frame, n, camera, title, grid_r, grid_basis,
                           tactile_at(args.frame), args.tactile_threshold,
                           args.tactile_scale)
        png = str(Path(args.input).resolve().parent / f"preview_frame{args.frame}.png")
        cv2.imwrite(png, img)
        print(f"Preview frame: {png}")
        return

    out = args.out or str(Path(args.input).resolve().parent / "preview.mp4")
    vw = cv2.VideoWriter(
        out, cv2.VideoWriter_fourcc(*"mp4v"), fps_value, (W, H))
    if not vw.isOpened():
        raise SystemExit(f"Error: cannot create MP4: {out}")
    try:
        for i in range(n):
            camera = cam_at(i)
            vw.write(render_frame(layout_at(kpts[i], camera), lab0[i], lab1[i],
                                  i, n, camera, title, grid_r, grid_basis,
                                  tactile_at(i), args.tactile_threshold,
                                  args.tactile_scale))
    finally:
        vw.release()
    print(f"Video: {out} ({n} frames at {fps_value:g} fps)")


if __name__ == "__main__":
    main()
