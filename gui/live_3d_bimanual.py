#!/usr/bin/env python3
"""Two STM32 gloves -> HAND2mm-calibrated MANO meshes -> live 3D.

Both hands arrive through the SDK's single entry point: ``sdk.open_bimanual``
returns a ``BimanualGlove`` whose ``read`` hands over one ``BimanualFrame``
carrying a complete ``HandFrame`` per hand, so this viewer owns only display,
joint smoothing and pressure presentation.  Each hand's link (wired USB or
Bluetooth dongle) is resolved by the SDK, not here.

This is the client edition, so it renders exactly what that edition hands over
-- solved joint positions and finger force in newtons -- and nothing else.
There is no recording path (capturing and writing data out is the admin
edition's job) and no raw IMU, pressure ADC or magnetometer register is read
anywhere here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

if getattr(sys, "frozen", False):
    BUNDLE_ROOT = Path(sys._MEIPASS)
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    BUNDLE_ROOT = PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)
DEFAULT_BIMANUAL_DISPLAY_CONFIG = PROJECT_ROOT / "config" / "bimanual_display_config.json"
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "calibration"
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from gui.calibration_selector import select_calibration_files  # noqa: E402
from gui.live_3d import (  # noqa: E402
    Live3DViewer, LiveViewState, VIEW3D_CONFIG_PATH,
    align_left_to_right_reference, apply_hand_display_rotation,
    hand_display_basis, place_hands_at_wrist_anchors)
from gui.rendering.tactile_overlay import (  # noqa: E402
    draw_bimanual_tactile_overlay)
from gui.rendering.tactile_pressure_hand import (  # noqa: E402
    PressureHandMapper, draw_pressure_color_points, draw_pressure_tiles)
from sdk.io.recording import JointPositionSmoother  # noqa: E402
from sdk.io.session import view_state_metadata  # noqa: E402
from sdk.device import DeviceManager  # noqa: E402
from sdk import StreamClosedError, StreamTimeoutError, open_bimanual  # noqa: E402
from sdk.calibration import find_latest_hand2mm_calibration  # noqa: E402

DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "glove_devices.json"
DEFAULT_GEOMETRY_PATH = (
    BUNDLE_ROOT / "assets/hand_geometry/hand_measured_runtime_v1.json")

# How long one pass may wait for the next bimanual frame.  The pass also drives
# the viewer, so it must come back often enough to keep the redraw on schedule;
# a frame that is already queued returns immediately.
READ_TIMEOUT_S = 0.02


@dataclass
class CalibrationSettings:
    side: str
    channel_to_hand: list[int]
    fps: float
    hardware_id: str
    usb_vid: int
    usb_pid: int
    param_inst_calib: dict | None = None
    param_shape_calib: dict | list | None = None
    transport: str = "usb_cdc"
    serial_port: str | None = None

    @property
    def calibration_hardware_id(self) -> str:
        mapping = "-".join(str(value) for value in self.channel_to_hand)
        return f"{self.hardware_id}|physical-to-hand:{mapping}"


def load_calibration(path: Path, side: str) -> CalibrationSettings:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {side} calibration {path}: {exc}") from exc
    saved_side = str(data.get("side") or "right").lower()
    if saved_side != side:
        raise RuntimeError(f"{path} is a {saved_side} calibration, not {side}")
    return CalibrationSettings(
        side=side,
        channel_to_hand=list(data.get("channel_to_hand") or range(16)),
        fps=float(data.get("fps") or 80),
        hardware_id=str(data.get("hardware_id", f"stm32-glove-{side}")),
        usb_vid=int(data.get("usb_vid", 0x0483)),
        usb_pid=int(data.get("usb_pid", 0x5740)),
        param_inst_calib=data.get("param_inst_calib"),
        param_shape_calib=data.get("param_shape_calib"),
    )




def resolve_calibration_paths(
        calibration_dir: Path | None,
        left_calib: Path | None,
        right_calib: Path | None) -> dict[str, Path]:
    """Resolve explicit files first, then a folder shortcut, then defaults."""
    if calibration_dir is not None:
        folder = Path(calibration_dir)
        left = (Path(left_calib) if left_calib is not None
                else find_latest_hand2mm_calibration(folder, "left"))
        right = (Path(right_calib) if right_calib is not None
                 else find_latest_hand2mm_calibration(folder, "right"))
    else:
        left = (Path(left_calib) if left_calib is not None
                else DEFAULT_CALIBRATION_DIR / "imu_calibration_left.json")
        right = (Path(right_calib) if right_calib is not None
                 else DEFAULT_CALIBRATION_DIR
                 / "imu_calibration_right.json")
    return {"left": left.expanduser(), "right": right.expanduser()}


def _validate_palm_target_bases(value) -> np.ndarray:
    targets = np.asarray(value, dtype=float).reshape(2, 3, 3)
    if not np.isfinite(targets).all():
        raise RuntimeError("Bimanual palm bases contain non-finite values")
    for slot, basis in enumerate(targets):
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5):
            raise RuntimeError(f"Bimanual palm base slot {slot} is not orthogonal")
        if not np.isclose(np.linalg.det(basis), 1.0, atol=1e-5):
            raise RuntimeError(f"Bimanual palm base slot {slot} is not a rotation")
    return targets


def load_bimanual_display_config(
        path: Path = DEFAULT_BIMANUAL_DISPLAY_CONFIG) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read bimanual display config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Bimanual display config must be a JSON object: {path}")
    targets = _validate_palm_target_bases(payload.get("palm_target_bases"))
    separation_m = float(payload.get("separation_m", 0.30))
    if not np.isfinite(separation_m) or separation_m <= 0.0:
        raise RuntimeError(f"Invalid bimanual wrist separation: {separation_m}")
    result = dict(payload)
    result["palm_target_bases"] = targets
    result["separation_m"] = separation_m
    root_sign = np.asarray(
        payload.get("left_root_rotvec_sign", [1.0, 1.0, 1.0]),
        dtype=float).reshape(3)
    if (not np.isfinite(root_sign).all()
            or not np.all(np.isin(root_sign, [-1.0, 1.0]))):
        raise RuntimeError(
            f"Left root rotation signs must be -1/+1: {root_sign.tolist()}")
    result["left_root_rotvec_sign"] = root_sign
    return result


def save_bimanual_display_config(
        path: Path, palm_target_bases: np.ndarray, separation_m: float,
        source_session: str = "runtime-relearn",
        view: dict | None = None,
        left_root_rotvec_sign: np.ndarray | None = None) -> Path:
    path = Path(path)
    targets = _validate_palm_target_bases(palm_target_bases)
    separation_m = float(separation_m)
    if not np.isfinite(separation_m) or separation_m <= 0.0:
        raise RuntimeError(f"Invalid bimanual wrist separation: {separation_m}")
    payload = {
        "schema": "stm32-imu-usb-bimanual-display",
        "schema_version": 1,
        "source_session": str(source_session),
        "updated_at": datetime.now().astimezone().isoformat(),
        "separation_m": separation_m,
        "palm_target_bases": targets.tolist(),
    }
    if view is not None:
        payload["view"] = view
    if left_root_rotvec_sign is not None:
        root_sign = np.asarray(left_root_rotvec_sign, dtype=float).reshape(3)
        if (not np.isfinite(root_sign).all()
                or not np.all(np.isin(root_sign, [-1.0, 1.0]))):
            raise RuntimeError(
                f"Left root rotation signs must be -1/+1: {root_sign.tolist()}")
        payload["left_root_rotvec_sign"] = root_sign.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)
    return path


def _vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).reshape(3)
    second = np.asarray(second, dtype=float).reshape(3)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return float("nan")
    cosine = float(np.clip((first @ second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def bimanual_relative_metrics(
        hands: np.ndarray,
        reference_hands: np.ndarray | None = None) -> dict[str, float]:
    """Measure camera-independent left/right display relationships."""
    slots = np.asarray(hands, dtype=np.float32).reshape(2, 21, 3)
    left_basis = hand_display_basis(slots[0])
    right_basis = hand_display_basis(slots[1])
    left_palm_facing = -left_basis[:, 2]
    right_palm_facing = right_basis[:, 2]
    wrist_delta = slots[1, 0] - slots[0, 0]
    metrics = {
        "wrist_distance_m": float(np.linalg.norm(wrist_delta)),
        "forward_angle_deg": _vector_angle_deg(
            left_basis[:, 1], right_basis[:, 1]),
        "palm_angle_deg": _vector_angle_deg(
            left_palm_facing, right_palm_facing),
        "forward_height_error_m": max(
            abs(float(wrist_delta @ left_basis[:, 1])),
            abs(float(wrist_delta @ right_basis[:, 1]))),
        "palm_height_error_m": max(
            abs(float(wrist_delta @ left_palm_facing)),
            abs(float(wrist_delta @ right_palm_facing))),
    }
    if reference_hands is not None:
        reference = np.asarray(reference_hands, dtype=np.float32).reshape(2, 21, 3)
        current_local = slots - slots[:, 0:1]
        reference_local = reference - reference[:, 0:1]
        movement = np.linalg.norm(current_local - reference_local, axis=-1)
        for slot, side in enumerate(("left", "right")):
            metrics[f"{side}_thumb_motion_m"] = float(
                np.sqrt(np.mean(np.square(movement[slot, 1:5]))))
            metrics[f"{side}_nonthumb_motion_m"] = float(
                np.sqrt(np.mean(np.square(movement[slot, 5:]))))
    return metrics


def stabilize_bimanual_palm_bases(
        hands: np.ndarray, target_bases: np.ndarray) -> np.ndarray:
    """Remove independent root drift while preserving articulated geometry.

    Each hand receives exactly one proper rigid rotation about its wrist.  The
    rotation maps the current non-thumb MCP palm basis onto the corresponding
    startup target basis; thumb/finger articulation and all within-hand
    distances are otherwise untouched.
    """
    slots = np.asarray(hands, dtype=np.float32).reshape(2, 21, 3).copy()
    targets = np.asarray(target_bases, dtype=float).reshape(2, 3, 3)
    for slot in range(2):
        current_basis = hand_display_basis(slots[slot])
        matrix = targets[slot] @ current_basis.T
        if np.linalg.det(matrix) < 0.0:
            raise ValueError("bimanual palm stabilization produced a reflection")
        slots[slot] = apply_hand_display_rotation(
            slots[slot], R.from_matrix(matrix))
    return slots


def fixed_bimanual_display_rotations(
        hands: np.ndarray, target_bases: np.ndarray) -> list[R]:
    """Compute one startup rotation per hand; later root motion is preserved."""
    slots = np.asarray(hands, dtype=np.float32).reshape(2, 21, 3)
    targets = _validate_palm_target_bases(target_bases)
    rotations = []
    for slot in range(2):
        current_basis = hand_display_basis(slots[slot])
        matrix = targets[slot] @ current_basis.T
        if np.linalg.det(matrix) < 0.0:
            raise ValueError("fixed bimanual display alignment produced a reflection")
        rotations.append(R.from_matrix(matrix))
    return rotations


def remap_hand_root_motion(
        hand: np.ndarray, target_basis: np.ndarray,
        local_rotvec_sign: np.ndarray) -> np.ndarray:
    """Remap root-rotation components without changing finger articulation."""
    shown = np.asarray(hand, dtype=np.float32).reshape(21, 3)
    target = np.asarray(target_basis, dtype=float).reshape(3, 3)
    signs = np.asarray(local_rotvec_sign, dtype=float).reshape(3)
    current = hand_display_basis(shown)
    delta_world = current @ target.T
    delta_local = target.T @ delta_world @ target
    local_rotvec = R.from_matrix(delta_local).as_rotvec()
    corrected_local = R.from_rotvec(local_rotvec * signs).as_matrix()
    corrected_world = target @ corrected_local @ target.T
    correction = corrected_world @ delta_world.T
    return apply_hand_display_rotation(shown, R.from_matrix(correction))


@dataclass
class HandRuntime:
    """One hand's newest SDK frame, reduced to what the display consumes.

    No raw field is carried: the frame the SDK hands over here has none to
    copy.  ``latest_smoothed`` is the display's own low-pass copy of the solved
    keypoints, kept distinct from the frame's ``joints_smoothed_m`` so this
    viewer's smoothing stays a presentation choice.
    """

    side: str
    smoother: JointPositionSmoother
    latest_joints: np.ndarray | None = None
    latest_smoothed: np.ndarray | None = None
    latest_host_s: float = 0.0
    last_seq: int = -1
    safety: bool = False
    contact: str | None = None
    # While the IMUs are still settling the SDK returns a neutral resting pose
    # rather than a solve of the wearer's hand, and flags it here.
    warming_up: bool = False
    # Device-side arrival rate, carried on the frame.
    device_fps: float = 0.0
    # The pressure stream's own arrival rate, a separate device stream: the
    # scan a frame carries is often the newest one standing over.
    tactile_fps: float = 0.0


def _hand_engine(bm, side: str):
    """One hand of the bimanual facade, for what the frame itself lacks.

    A ``BimanualFrame`` carries joints and no MANO surface, so the mesh the
    renderer draws and the solver backend it names come from the hand's own
    facade -- ``BimanualGlove.left`` / ``.right``, which is the same public
    surface ``sdk.open_glove`` hands back, read through its public
    ``mesh_vertices_m`` and ``backend``.  No solver is poked and no solve
    happens here.
    """

    return getattr(bm, side)


class BimanualViewer(Live3DViewer):
    def __init__(self, *args, separation_m=0.30,
                 display_rotations=None, palm_target_bases=None,
                 left_root_rotvec_sign=None,
                 display_config_path=DEFAULT_BIMANUAL_DISPLAY_CONFIG,
                 **kwargs):
        self.side_status = {
            "left": {"fps": 0.0, "device_fps": 0.0, "tactile_fps": 0.0,
                     "warming_up": False,
                     "safety": False, "contact": None},
            "right": {"fps": 0.0, "device_fps": 0.0, "tactile_fps": 0.0,
                      "warming_up": False,
                      "safety": False, "contact": None},
        }
        self.tactile_frames = {"left": None, "right": None}
        self.tactile_calibrating_sides = {"left": True, "right": True}
        # Force samples can be projected onto the animated MANO surface as
        # either coloured points or cell-shaped tiles.  The barycentric anchor
        # map is static; only the live mesh positions change each frame.
        self.pressure_display_mode = "none"
        self._pressure_mappers: dict[str, PressureHandMapper] = {}
        self.separation_m = float(separation_m)
        self._relative_reference_slots = None
        raw_initial_slots = np.asarray(
            args[1] if len(args) >= 2 else kwargs["initial_slots"],
            dtype=np.float32).reshape(2, 21, 3).copy()
        target_seed_slots = raw_initial_slots.copy()
        startup_rotations = list(display_rotations or [None, None])
        for slot, rotation in enumerate(startup_rotations):
            if rotation is not None:
                target_seed_slots[slot] = apply_hand_display_rotation(
                    target_seed_slots[slot], rotation)
        saved_display = (load_bimanual_display_config(Path(display_config_path))
                         if display_config_path is not None else None)
        if saved_display is not None:
            if palm_target_bases is None:
                palm_target_bases = saved_display["palm_target_bases"]
            if left_root_rotvec_sign is None:
                left_root_rotvec_sign = saved_display["left_root_rotvec_sign"]
        self._palm_target_bases = (
            _validate_palm_target_bases(palm_target_bases)
            if palm_target_bases is not None
            else np.stack([
                hand_display_basis(target_seed_slots[slot]) for slot in range(2)
            ]))
        fixed_rotations = fixed_bimanual_display_rotations(
            raw_initial_slots, self._palm_target_bases)
        self._anchor_axis = -self._palm_target_bases[1, :, 0].copy()
        self._left_root_rotvec_sign = np.asarray(
            left_root_rotvec_sign
            if left_root_rotvec_sign is not None else [1.0, 1.0, 1.0],
            dtype=float).reshape(3)
        grid_forward = self._palm_target_bases[1, :, 1].copy()
        grid_normal = np.cross(self._anchor_axis, grid_forward)
        grid_normal /= max(float(np.linalg.norm(grid_normal)), 1e-9)
        self._bimanual_grid_basis = np.stack([
            self._anchor_axis, grid_forward, grid_normal], axis=1)
        super().__init__(*args, hand_side="right",
                         display_rotations=fixed_rotations,
                         show_surface_color=True, **kwargs)
        if self._mesh_faces is not None:
            for side in ("left", "right"):
                try:
                    self._pressure_mappers[side] = PressureHandMapper(
                        side, self._mesh_faces,
                        BUNDLE_ROOT / "assets" / "hand")
                except (OSError, RuntimeError, ValueError) as exc:
                    print(
                        f"[Pressure view warning] Cannot prepare {side} hand map: {exc}",
                        file=sys.stderr)
        self._relative_reference_slots = self._display_slots().copy()
        self._redraw_requested = True

    def _hud_lines(self):
        lines = [
            f"Bimanual frames: {self.frame_count}",
        ]
        metrics = bimanual_relative_metrics(
            self._display_slots(), self._relative_reference_slots)
        lines.extend([
            "REL wrist={:.1f}cm  height(fwd/palm)={:.1f}/{:.1f}mm".format(
                metrics["wrist_distance_m"] * 100.0,
                metrics["forward_height_error_m"] * 1000.0,
                metrics["palm_height_error_m"] * 1000.0),
            "REL fingers={:.2f}deg  palms={:.2f}deg".format(
                metrics["forward_angle_deg"], metrics["palm_angle_deg"]),
        ])
        if self._relative_reference_slots is not None:
            lines.append(
                "MOVE thumb/nonthumb mm  L={:.1f}/{:.1f}  R={:.1f}/{:.1f}".format(
                    metrics["left_thumb_motion_m"] * 1000.0,
                    metrics["left_nonthumb_motion_m"] * 1000.0,
                    metrics["right_thumb_motion_m"] * 1000.0,
                    metrics["right_nonthumb_motion_m"] * 1000.0))
        for side in ("left", "right"):
            status = self.side_status[side]
            # Solve rate first, then each device stream's own arrival rate:
            # when they differ, the gap is the host dropping frames, not the
            # glove withholding them, and that is the whole point of showing
            # all three.  Attitude and pressure are separate streams, so one
            # rate cannot stand in for the other.
            # Warm-up is flagged because until it clears the joints are a
            # neutral placeholder rather than a solve of the wearer's hand.
            lines.append(
                f"{side.upper()}: {status['fps']:5.1f} FPS  "
                f"IMU {float(status.get('device_fps') or 0.0):5.1f} Hz  "
                f"TACT {float(status.get('tactile_fps') or 0.0):5.1f} Hz"
                + ("  WARMUP" if status.get("warming_up") else "")
                + ("  SAFETY" if status["safety"] else "")
                + (f"  contact={status['contact']}" if status["contact"] else ""))
        return lines

    def update_both(self, left, right, frame_count, statuses,
                    mesh_slots: list[np.ndarray | None] | None = None):
        slots = np.stack([
            np.asarray(left, np.float32).reshape(21, 3),
            np.asarray(right, np.float32).reshape(21, 3),
        ])
        self._kpts_slots = slots
        # Keep the mesh in the solver's wrist-relative coordinate frame.  It
        # receives the same display-space transforms as the 21 joints below.
        self._mesh_slots = [None, None]
        if mesh_slots is not None:
            for slot, vertices in enumerate(mesh_slots[:2]):
                if vertices is None:
                    continue
                mesh = np.asarray(vertices, dtype=np.float32)
                if mesh.shape == (778, 3) and np.isfinite(mesh).all():
                    self._mesh_slots[slot] = mesh
        self.frame_count = int(frame_count)
        self.side_status = statuses
        self._redraw_requested = True

    def _display_slots(self):
        slots = super()._display_slots()
        slots[0] = remap_hand_root_motion(
            slots[0], self._palm_target_bases[0],
            self._left_root_rotvec_sign)
        return place_hands_at_wrist_anchors(
            slots, separation_m=self.separation_m,
            anchor_axis=self._anchor_axis)

    def _display_mesh_slots(self) -> list[np.ndarray | None]:
        """Place MANO surfaces exactly where their corresponding joints land."""
        source_slots = super()._display_slots()
        source_mesh_slots = super()._display_mesh_slots()
        final_slots = self._display_slots()
        result: list[np.ndarray | None] = [None, None]
        for slot, source_mesh in enumerate(source_mesh_slots):
            if (source_mesh is None or not np.isfinite(source_slots[slot]).all()
                    or not np.isfinite(final_slots[slot]).all()):
                continue
            source_basis = hand_display_basis(source_slots[slot])
            final_basis = hand_display_basis(final_slots[slot])
            rotation = final_basis @ source_basis.T
            mesh = np.asarray(source_mesh, dtype=np.float32).reshape(778, 3)
            result[slot] = (
                (mesh - source_slots[slot, 0]) @ rotation.T
                + final_slots[slot, 0]
            ).astype(np.float32)
        return result

    def _grid_basis(self):
        return self._bimanual_grid_basis

    def _pressure_display_mode_label(self) -> str:
        return "Pressure on hand"

    def _pressure_display_mode_options(self) -> list[tuple[str, str]]:
        return [
            ("none", "None"),
            ("points", "Pressure points"),
            ("tiles", "Pressure tiles"),
        ]

    def _on_pressure_display_mode(self, mode: str) -> None:
        self.pressure_display_mode = (
            str(mode) if str(mode) in {"none", "points", "tiles"} else "none")
        self._redraw_requested = True

    def _draw(self):
        img = super()._draw()
        camera = self._camera()
        wrists = self._controlled_display_slots()[:, 0, :]
        points, depth = camera.project(wrists)
        for index, label, color in ((0, "L", (255, 180, 80)),
                                    (1, "R", (80, 180, 255))):
            if np.isfinite(points[index]).all() and depth[index] > 0:
                position = tuple(points[index].astype(int) + np.array([8, -8]))
                cv2.putText(img, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, color, 2, cv2.LINE_AA)
        return img

    def set_tactile_side(self, side, force):
        """Store one hand's newest force map, or clear it until the first scan.

        The map arrives already fitted to newtons, so it is only reshaped here:
        NaN cells are kept -- the overlay reads them as zero force for the
        positions no fitted region covers.
        """
        self.tactile_frames[side] = (None if force is None else
                                     np.asarray(force, np.float32).reshape(16, 16))
        self.tactile_calibrating_sides[side] = force is None
        self._redraw_requested = True

    def _draw_tactile(self, img):
        # Force comes from the SDK already in newtons, so it is drawn with the
        # newton scale: ``newtons=True`` on the mesh overlays, and
        # ``linear_fitted=True`` on the fixed matrix card.  The ADC-scale
        # threshold slider no longer applies -- newtons carry their own zero.
        if (self.display_mode == "hand" and self._pressure_mappers
                and self.pressure_display_mode != "none"):
            draw_args = (
                img, self._camera(), self._controlled_display_mesh_slots(),
                self.tactile_frames, self._pressure_mappers,
                {"left": False, "right": False},
            )
            if self.pressure_display_mode == "tiles":
                draw_pressure_tiles(*draw_args, newtons=True)
            else:
                draw_pressure_color_points(*draw_args, newtons=True)
        draw_bimanual_tactile_overlay(
            img,
            self.tactile_frames,
            scale=self.tactile_scale,
            linear_fitted=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", "--calib-dir", type=Path,
                        help="Folder shortcut for the newest left/right calibrations")
    parser.add_argument("--left-calib", type=Path,
                        help="Explicit left-hand IMU calibration JSON")
    parser.add_argument("--right-calib", type=Path,
                        help="Explicit right-hand IMU calibration JSON")
    parser.add_argument(
        "--select-calibration", action="store_true",
        help="Select left and right calibration JSON files before startup")
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_PATH,
                        help="Hand geometry JSON")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--display-mode", default="hand", choices=("bones", "hand"),
                        help="Initial 3D display mode: bones or full MANO hand mesh")
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--max-frame-age", type=float, default=0.25)
    parser.add_argument("--display-config", type=Path,
                        default=DEFAULT_BIMANUAL_DISPLAY_CONFIG,
                        help="Persistent palm bases and wrist separation")
    parser.add_argument("--view-config", type=Path,
                        default=PROJECT_ROOT / "config" / "view_config_3d_bimanual.json")
    parser.add_argument("--view-reference-config", type=Path,
                        default=VIEW3D_CONFIG_PATH)
    parser.add_argument("--relearn-display-default", action="store_true",
                        help="Relearn and overwrite display defaults from the first frame")
    parser.add_argument("--separation", type=float, default=None,
                        help="Override wrist separation in metres for this run")
    parser.add_argument("--yaw", type=float, default=None)
    parser.add_argument("--elev", type=float, default=None)
    parser.add_argument("--roll", type=float, default=None)
    parser.add_argument("--dist", type=float, default=None)
    parser.add_argument("--no-display", action="store_true",
                        help="Run the display loop headless (no 3D window)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="Deprecated compatibility option; both calibrations are required")
    args = parser.parse_args(argv)
    if args.select_calibration and args.calibration_dir is None:
        missing_sides = tuple(
            side for side, value in (
                ("left", args.left_calib), ("right", args.right_calib))
            if value is None)
        if missing_sides:
            try:
                selected = select_calibration_files(
                    DEFAULT_CALIBRATION_DIR, missing_sides,
                    title="Select Bimanual IMU Calibrations")
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"[Calibration selection error] {exc}", file=sys.stderr)
                return 2
            if selected is None:
                print("Calibration selection cancelled.")
                return 130
            if args.left_calib is None:
                args.left_calib = selected.get("left")
            if args.right_calib is None:
                args.right_calib = selected.get("right")
    # sdk.open_bimanual builds its solver on the SDK's own default geometry, so
    # an override cannot be routed through it: refuse loudly rather than show a
    # differently-fitted hand than the caller asked for.
    if Path(args.geometry).resolve() != DEFAULT_GEOMETRY_PATH.resolve():
        parser.error(
            f"--geometry has no effect: sdk.open_bimanual always solves with "
            f"{DEFAULT_GEOMETRY_PATH}")
    calib_paths = resolve_calibration_paths(
        args.calibration_dir, args.left_calib, args.right_calib)

    saved_display = load_bimanual_display_config(args.display_config)
    if saved_display is not None and not args.relearn_display_default:
        palm_target_bases = saved_display["palm_target_bases"].copy()
        print(
            f"Bimanual display config: {args.display_config.resolve()} "
            f"(source={saved_display.get('source_session', 'unknown')})")
    else:
        palm_target_bases = None
        if saved_display is None and not args.relearn_display_default:
            print(
                f"[WARN] Missing bimanual display config {args.display_config}; "
                "it will be created from the first frame.",
                file=sys.stderr)
    left_root_rotvec_sign = np.asarray(
        saved_display["left_root_rotvec_sign"]
        if saved_display is not None else [-1.0, -1.0, 1.0],
        dtype=float).reshape(3)
    if args.separation is not None:
        separation_m = float(args.separation)
    elif saved_display is not None:
        separation_m = float(saved_display["separation_m"])
    else:
        separation_m = 0.30

    bindings = DeviceManager(args.registry).validate_bimanual()
    serials = {
        "left": bindings.left_serial,
        "right": bindings.right_serial,
    }
    for side in ("left", "right"):
        print(f"{side} device: serial={serials[side]}")
    print(f"Left IMU calibration: {calib_paths['left'].resolve()}")
    print(f"Right IMU calibration: {calib_paths['right'].resolve()}")
    runtimes = {}
    for side in ("left", "right"):
        if not calib_paths[side].is_file():
            raise RuntimeError(
                f"Missing {side} calibration: {calib_paths[side]}")
        from sdk.calibration import prepare_viewer_calibration
        calib_paths[side] = prepare_viewer_calibration(calib_paths[side], side)
        load_calibration(calib_paths[side], side)
        runtimes[side] = HandRuntime(
            side=side, smoother=JointPositionSmoother(0.035))
    # The calibration files resolved for this run are handed over explicitly:
    # a folder shortcut or the file picker must win over the SDK's own
    # "newest usable file" discovery.  Which link carries each hand (wired or
    # Bluetooth dongle) is left to the SDK.
    bm = open_bimanual(
        left_calibration=calib_paths["left"],
        right_calibration=calib_paths["right"],
        registry=args.registry,
    )
    hands = {side: _hand_engine(bm, side) for side in ("left", "right")}
    runtime_backend = None
    for side in ("left", "right"):
        selected_backend = hands[side].backend
        if runtime_backend is None:
            runtime_backend = selected_backend
        elif runtime_backend["backend"] != selected_backend["backend"]:
            raise RuntimeError("Left and right solver backends do not match")

    backend_labels = {
        "hand_pinky_plus_2mm": "HAND pinky +2 mm LOCAL",
        "retargeted_direct": "measured direct / calibrated HAND orientation",
        "fitted_direct": "fitted project direct FK",
        "direct_fk": "direct FK",
    }
    backend_label = backend_labels[runtime_backend["backend"]]
    print("Pipeline: sdk.open_bimanual -> 21 solved keypoints + finger force "
          f"in newtons ({backend_label})")

    right_view = LiveViewState(args.view_reference_config)
    view_state = LiveViewState(args.view_config)
    saved_view = (saved_display.get("view", {})
                  if saved_display is not None else {})
    view_state.yaw = float(saved_view.get("yaw", right_view.yaw))
    view_state.elev = float(saved_view.get("elev", right_view.elev))
    view_state.roll = float(saved_view.get("roll", right_view.roll)) % 360.0
    view_state.dist = float(saved_view.get("dist", right_view.dist))
    view_state.pan_px = [float(value) for value in
                         saved_view.get("pan_px", right_view.pan_px)]
    if args.yaw is not None:
        view_state.yaw = float(args.yaw)
    if args.elev is not None:
        view_state.elev = float(args.elev)
    if args.roll is not None:
        view_state.roll = float(args.roll) % 360.0
    if args.dist is not None:
        view_state.dist = float(args.dist)
    viewer = None
    startup_initialized = False
    started = time.monotonic()
    next_sample = started
    processed_frame_index = 0
    fps_times = {"left": [], "right": []}
    try:
        bm.start()
        # Printed once the SDK has resolved them, because the link a hand is
        # reached over (wired or Bluetooth dongle) is the SDK's choosing, not
        # the registry's.
        health = bm.health()
        for side in ("left", "right"):
            print(f"{side} link: {getattr(health, side).message}")
        while True:
            now = time.monotonic()
            try:
                newest = bm.read(timeout=READ_TIMEOUT_S)
            except StreamTimeoutError:
                newest = None
            except StreamClosedError as exc:
                # A link that died under us must not read as a slow glove: end
                # the run cleanly.
                print(f"[Stream error] {exc}", file=sys.stderr)
                break
            if newest is not None:
                # Keep only the newest frame of what the link has queued.  The
                # SDK queue is a FIFO, so taking one frame per pass would
                # replay a Bluetooth batch the dongle handed over long ago;
                # draining it is the "newest, never history" policy
                # common.frame_pacing.recent_batch applied to one poll.
                while True:
                    try:
                        newest = bm.read(timeout=0)
                    except (StreamTimeoutError, StreamClosedError):
                        break
            if newest is not None:
                for side, runtime in runtimes.items():
                    frame = getattr(newest, side)
                    if frame.sequence == runtime.last_seq:
                        continue
                    runtime.last_seq = frame.sequence
                    # The local smoother runs on the raw keypoints, so display
                    # smoothing stays a presentation choice here instead of
                    # following the SDK's smoothing_ms default.
                    runtime.latest_joints = frame.joints_raw_m
                    runtime.latest_smoothed = runtime.smoother.update(
                        frame.joints_raw_m, frame.timestamp_us / 1_000_000.0)
                    runtime.latest_host_s = now
                    runtime.warming_up = bool(
                        frame.status.details.get("warming_up"))
                    runtime.safety = frame.status.safety_intervened
                    runtime.contact = frame.status.active_contact
                    runtime.device_fps = float(frame.device_fps)
                    runtime.tactile_fps = float(frame.tactile_fps)
                    # Force rides the frame as the newest scan, repeated until a
                    # newer one lands.  None means no scan has arrived yet, so
                    # whatever was last shown stays in place.  It is already
                    # fitted to newtons inside the SDK, so it goes straight to
                    # the display with no further processing.
                    if frame.force_n is not None and viewer:
                        viewer.set_tactile_side(side, frame.force_n)
                    fps_times[side].append(now)
                    fps_times[side] = fps_times[side][-60:]

            ready = all(runtime.latest_joints is not None
                        for runtime in runtimes.values())
            fresh = ready and all(
                now - runtime.latest_host_s <= args.max_frame_age
                for runtime in runtimes.values())
            if ready and not startup_initialized:
                initial = np.stack([
                    runtimes["left"].latest_joints,
                    runtimes["right"].latest_joints,
                ]).astype(np.float32)
                if palm_target_bases is None:
                    # Use the first frame on explicit relearn or a missing default, and persist it immediately.
                    left_rotation = align_left_to_right_reference(
                        initial[0], initial[1])
                    initial_display = initial.copy()
                    initial_display[0] = apply_hand_display_rotation(
                        initial_display[0], left_rotation)
                    palm_target_bases = np.stack([
                        hand_display_basis(initial_display[slot])
                        for slot in range(2)
                    ])
                    saved_path = save_bimanual_display_config(
                        args.display_config, palm_target_bases, separation_m,
                        source_session=(
                            "runtime-explicit-relearn"
                            if args.relearn_display_default
                            else "runtime-auto-create"),
                        view=view_state_metadata(view_state),
                        left_root_rotvec_sign=left_root_rotvec_sign)
                    # save_bimanual_display_config is the only path allowed to
                    # change this persistent left-root handedness correction.
                    print(f"Bimanual display config saved: {saved_path}")
                if not args.no_display:
                    viewer = BimanualViewer(
                        view_state, initial,
                        f"STM32 bimanual live 21 joints / {backend_label}",
                        separation_m=separation_m,
                        palm_target_bases=palm_target_bases,
                        left_root_rotvec_sign=left_root_rotvec_sign,
                        display_config_path=None,
                        display_mode=args.display_mode,
                        initial_mesh_vertices=hands["right"].mesh_vertices_m,
                        redraw_interval_s=1.0 / max(args.fps, 1.0))
                    def set_contacts(enabled, blend):
                        for hand in hands.values():
                            hand.set_contact_controls(enabled, blend)
                    def set_root_limits(enabled):
                        for hand in hands.values():
                            hand.set_root_limits_enabled(enabled)
                    viewer.bind_root_limits(set_root_limits)
                    viewer.bind_contact_controls(
                        set_contacts,
                        experimental=any(hand.backend.get('pose_postprocess') == 'gesture-retarget-v1'
                                         for hand in hands.values()),
                        source_refit=any(hand.backend.get('pose_postprocess') == 'sensor-motion-refit-v1'
                                         for hand in hands.values()))
                startup_initialized = True
            if fresh and now >= next_sample:
                if viewer:
                    statuses = {}
                    for side in ("left", "right"):
                        times = fps_times[side]
                        fps = ((len(times)-1) / max(times[-1]-times[0], 1e-6)
                               if len(times) >= 2 else 0.0)
                        statuses[side] = {
                            "fps": fps,
                            "device_fps": runtimes[side].device_fps,
                            "tactile_fps": runtimes[side].tactile_fps,
                            "warming_up": runtimes[side].warming_up,
                            "safety": runtimes[side].safety,
                            "contact": runtimes[side].contact,
                        }
                    viewer.update_both(
                        runtimes["left"].latest_smoothed,
                        runtimes["right"].latest_smoothed,
                        processed_frame_index + 1, statuses,
                        mesh_slots=[
                            hands["left"].mesh_vertices_m,
                            hands["right"].mesh_vertices_m,
                        ])
                processed_frame_index += 1
                next_sample = now + 1.0 / max(args.fps, 1.0)
                if args.max_frames and processed_frame_index >= args.max_frames:
                    break
            elif not ready and now - started > args.startup_timeout:
                waiting = [side for side, runtime in runtimes.items()
                           if runtime.latest_joints is None]
                raise RuntimeError(f"Timed out waiting for bimanual data: {waiting}")

            if viewer and not viewer.tick():
                break
            time.sleep(0.002)
    finally:
        if viewer:
            viewer.close()
        bm.close()
    print(f"Bimanual run finished: {processed_frame_index} preview frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
