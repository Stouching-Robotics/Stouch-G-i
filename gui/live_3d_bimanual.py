#!/usr/bin/env python3
"""Two STM32 gloves -> HAND2mm-calibrated MANO meshes -> live 3D."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
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
    H, W, Live3DViewer, LiveViewState, VIEW3D_CONFIG_PATH,
    align_left_to_right_reference, apply_hand_display_rotation,
    hand_display_basis, _latest_solver_mesh_vertices,
    place_hands_at_wrist_anchors)
from gui.rendering.tactile import render_hand  # noqa: E402
from gui.rendering.tactile_pressure_hand import (  # noqa: E402
    PressureHandMapper, draw_pressure_color_points, draw_pressure_tiles,
)
from glove_io.recording import (  # noqa: E402
    BimanualRecorder, BimanualRecordingSession,
    JointPositionSmoother, KeypointParquetRecorder,
    SkeletonRenderer)
from glove_io.session import (  # noqa: E402
    session_metadata_path, update_session_metadata,
    view_state_metadata, write_session_metadata)
from glove_io.tactile_processing import TactilePreprocessor  # noqa: E402
from runtime import DeviceManager, HandSolver, RawImuStream  # noqa: E402

DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "glove_devices.json"
DEFAULT_GEOMETRY_PATH = (
    BUNDLE_ROOT / "assets/hand_geometry/hand_measured_runtime_v1.json")


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


def find_latest_hand2mm_calibration(directory: Path, side: str) -> Path:
    """Find the newest HAND2mm-compatible calibration for one hand."""
    folder = Path(directory).expanduser().resolve()
    if not folder.is_dir():
        raise RuntimeError(f"Calibration directory does not exist: {folder}")
    matches = []
    for path in folder.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        saved_side = str(payload.get("side") or "right").lower()
        if saved_side != side or not isinstance(payload.get("param_inst_calib"), dict):
            continue
        # The local HAND2mm path consumes the original root/installation and
        # contact payload. Direct-FK factory files use a different solver.
        if payload.get("kinematics") in {"direct_fk", "fitted_direct_fk"}:
            continue
        if str(payload.get("tool") or "") in {
                "direct_fk_calibrate_cli", "hand_fk_fit_cli"}:
            continue
        matches.append(path)
    if not matches:
        raise RuntimeError(
            f"No usable HAND2mm {side} IMU calibration JSON in {folder}")
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, path.name))


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
                else DEFAULT_CALIBRATION_DIR / "imu_2d_calibration.json")
        right = (Path(right_calib) if right_calib is not None
                 else DEFAULT_CALIBRATION_DIR / "imu_calibration.json")
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
    side: str
    config: CalibrationSettings
    solver: HandSolver
    stream: RawImuStream
    tactile_stream: object
    smoother: JointPositionSmoother
    tactile_pre: TactilePreprocessor
    missing: list[str]
    latest_joints: np.ndarray | None = None
    latest_smoothed: np.ndarray | None = None
    latest_uv: np.ndarray | None = None
    latest_imu: np.ndarray | None = None
    latest_raw_imu: np.ndarray | None = None
    latest_raw_present: np.ndarray | None = None
    latest_raw_valid: np.ndarray | None = None
    latest_raw_device_timestamp_us: int | None = None
    latest_tactile: np.ndarray | None = None
    latest_tactile_raw: np.ndarray | None = None
    latest_host_s: float = 0.0
    last_seq: int = -1
    safety: bool = False
    contact: str | None = None


def draw_bimanual_tactile_overlay(
        img: np.ndarray,
        tactile_frames: dict[str, np.ndarray | None],
        threshold: float = 0.0,
        scale: float = 0.6) -> np.ndarray:
    """Draw the exact fixed left/right tactile panels used by live view."""
    for side, x_anchor in (("left", 12), ("right", None)):
        value = tactile_frames.get(side)
        if value is None:
            continue
        frame = np.asarray(value, dtype=np.float32).reshape(16, 16)
        if not np.isfinite(frame).any():
            continue
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        if threshold > 0:
            frame = np.where(frame < float(threshold), 0.0, frame)
        hand_img = render_hand(frame, mirror=side == "left", side=side)
        shown_scale = min(max(float(scale), 0.1), 0.7)
        tw = max(1, int(round(hand_img.shape[1] * shown_scale)))
        th = max(1, int(round(hand_img.shape[0] * shown_scale)))
        shown = cv2.resize(
            hand_img, (tw, th), interpolation=cv2.INTER_LINEAR)
        x0 = x_anchor if x_anchor is not None else img.shape[1] - tw - 12
        y0 = img.shape[0] - th - 60
        img[y0:y0 + th, x0:x0 + tw] = shown
        cv2.putText(
            img, side.upper(), (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 195, 185), 1, cv2.LINE_AA)
    return img


class BimanualViewer(Live3DViewer):
    def __init__(self, *args, separation_m=0.30,
                 display_rotations=None, palm_target_bases=None,
                 left_root_rotvec_sign=None,
                 display_config_path=DEFAULT_BIMANUAL_DISPLAY_CONFIG,
                 recording_enabled=True, recording_state="idle",
                 **kwargs):
        self.recording_enabled = bool(recording_enabled)
        self.recording_state = str(recording_state)
        self._record_action_requested: str | None = None
        self.side_status = {
            "left": {"fps": 0.0, "device_fps": 0.0, "missing": [], "safety": False,
                     "contact": None},
            "right": {"fps": 0.0, "device_fps": 0.0, "missing": [], "safety": False,
                      "contact": None},
        }
        self.tactile_frames = {"left": None, "right": None}
        self.tactile_calibrating_sides = {"left": True, "right": True}
        # Pressure samples can be projected onto the animated MANO surface as
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
        state_labels = {
            "disabled": "DISABLED",
            "idle": "READY - press button or SPACE",
            "recording": "RECORDING",
            "saved": "SAVED - ready for next take",
        }
        lines = [
            f"Recording: {state_labels.get(self.recording_state, self.recording_state)}",
            f"Bimanual recorded frames: {self.frame_count}",
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
            # Solve rate first, device arrival rate right behind it: when they
            # differ, the gap is the host dropping frames, not the glove
            # withholding them, and that is the whole point of showing both.
            lines.append(
                f"{side.upper()}: {status['fps']:5.1f} FPS  "
                f"DEV {float(status.get('device_fps') or 0.0):5.1f} Hz  "
                f"IMU {16-len(status['missing'])}/16"
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
        self._draw_record_button(img)
        return img

    @staticmethod
    def _record_button_rect() -> tuple[int, int, int, int]:
        return W - 305, 18, W - 22, 66

    def _draw_record_button(self, img: np.ndarray) -> None:
        x0, y0, x1, y1 = self._record_button_rect()
        if not self.recording_enabled:
            color, label = (75, 75, 75), "RECORDING DISABLED"
        elif self.recording_state == "recording":
            color, label = (45, 55, 220), "STOP & SAVE"
        elif self.recording_state == "saved":
            color, label = (55, 145, 75), "START NEW RECORDING"
        else:
            color, label = (55, 155, 70), "START RECORDING"
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (150, 165, 190), 1)
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        origin = (
            x0 + max(8, (x1 - x0 - text_size[0]) // 2),
            y0 + (y1 - y0 + text_size[1]) // 2,
        )
        cv2.putText(img, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (245, 245, 245), 2, cv2.LINE_AA)
        if self.recording_enabled and self.recording_state != "saved":
            cv2.putText(
                img, "SPACE = start / stop & save", (x0 + 18, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 195, 185), 1,
                cv2.LINE_AA)

    def _request_record_toggle(self) -> None:
        if not self.recording_enabled:
            return
        self._record_action_requested = (
            "stop" if self.recording_state == "recording" else "start")

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x0, y0, x1, y1 = self._record_button_rect()
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._request_record_toggle()
                return
        super()._on_mouse(event, x, y, flags, param)

    def _key_command(self, key: int):
        if key == ord(" "):
            self._request_record_toggle()
            return
        super()._key_command(key)

    def consume_record_action(self) -> str | None:
        action = self._record_action_requested
        self._record_action_requested = None
        return action

    def set_recording_state(self, state: str, frame_count: int = 0) -> None:
        if state not in {"disabled", "idle", "recording", "saved"}:
            raise ValueError(f"invalid recording state: {state}")
        self.recording_state = state
        self.frame_count = int(frame_count)
        self._redraw_requested = True

    def set_tactile_side(self, side, processed):
        self.tactile_frames[side] = (None if processed is None else
                                     np.asarray(processed, np.float32).reshape(16, 16))
        self.tactile_calibrating_sides[side] = processed is None
        self._redraw_requested = True

    def _draw_tactile(self, img):
        if (self.display_mode == "hand" and self._pressure_mappers
                and self.pressure_display_mode != "none"):
            draw_args = (
                img, self._camera(), self._controlled_display_mesh_slots(),
                self.tactile_frames, self._pressure_mappers,
                {"left": False, "right": False},
            )
            if self.pressure_display_mode == "tiles":
                draw_pressure_tiles(
                    *draw_args, threshold=float(self.tactile_threshold))
            else:
                draw_pressure_color_points(
                    *draw_args, threshold=float(self.tactile_threshold))
        draw_bimanual_tactile_overlay(
            img,
            self.tactile_frames,
            threshold=self.tactile_threshold,
            scale=self.tactile_scale,
        )


def make_result(runtimes, projector):
    hands3d = np.stack([runtimes["left"].latest_joints,
                        runtimes["right"].latest_joints]).astype(np.float32)
    smoothed = np.stack([runtimes["left"].latest_smoothed,
                         runtimes["right"].latest_smoothed]).astype(np.float32)
    hands2d = np.stack([projector.project(smoothed[index])[0] for index in range(2)])
    ready = [len(runtimes[side].missing) == 0 for side in ("left", "right")]
    return {
        "hands2d": hands2d,
        "hands3d": hands3d,
        "smoothed": smoothed,
        "labels": ["Left", "Right"],
        "presents": [True, True],
        "propagated": [False, False],
        "stage2": ready,
        "reprojection_error": [float("nan"), float("nan")],
    }


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
    parser.add_argument("--out", type=Path)
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
    parser.add_argument("--checkpoint-frames", type=int, default=600)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--record-on-start", action="store_true",
        help="Start recording when data is ready; default waits for the GUI button")
    parser.add_argument("--no-display", action="store_true")
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
    if args.no_save and args.record_on_start:
        parser.error("--no-save and --record-on-start cannot be combined")
    if args.no_display and not (args.no_save or args.record_on_start):
        parser.error("--no-display requires --record-on-start or --no-save")
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
    ports = {"left": bindings.left_port, "right": bindings.right_port}
    serials = {
        "left": bindings.left_serial,
        "right": bindings.right_serial,
    }
    for side in ("left", "right"):
        print(
            f"{side} device: serial={serials[side]}  "
            f"port={ports[side]}")
    print(f"Left IMU calibration: {calib_paths['left'].resolve()}")
    print(f"Right IMU calibration: {calib_paths['right'].resolve()}")
    runtimes = {}
    runtime_backend = None
    registry = {}
    for side in ("left", "right"):
        if not calib_paths[side].is_file():
            raise RuntimeError(
                f"Missing {side} calibration: {calib_paths[side]}")
        config = load_calibration(calib_paths[side], side)
        try:
            solver = HandSolver(side, calib_paths[side], args.geometry)
            selected_backend = solver.backend
            if runtime_backend is None:
                runtime_backend = selected_backend
            elif runtime_backend["backend"] != selected_backend["backend"]:
                raise RuntimeError("Left and right solver backends do not match")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            raise RuntimeError(f"{side} calibration is not usable: {exc}") from exc
        stream = RawImuStream(
            ports[side], usb_vid=config.usb_vid, usb_pid=config.usb_pid)
        runtime = HandRuntime(
            side=side, config=config, solver=solver,
            stream=stream,
            tactile_stream=stream.tactile_stream(),
            smoother=JointPositionSmoother(0.035),
            tactile_pre=TactilePreprocessor(
                base_gate=0.0, dynamic_noise_ratio=0.0, temporal_smooth=0.15,
                spatial_filter=False, calibration_frames=100, bypass_gates=False),
            missing=[f"IMU-{index}" for index in range(16)],
        )
        runtimes[side] = runtime
        registry[side] = {
            "usb_serial": serials[side],
            "hardware_id": config.hardware_id,
            "channel_to_hand": config.channel_to_hand,
        }
    for runtime in runtimes.values():
        runtime.stream.start()
        runtime.tactile_stream.start()

    if runtime_backend["backend"] == "hand_pinky_plus_2mm":
        print("Pipeline: 32 raw IMUs -> protected HAND solver -> 42 keypoints")
    elif runtime_backend["backend"] == "retargeted_direct":
        print("Pipeline: 32 raw IMUs -> calibrated retargeting -> 42 keypoints")
    elif runtime_backend["backend"] == "fitted_direct":
        print("Pipeline: 32 raw IMUs -> fitted direct FK -> 42 keypoints")
    else:
        print("Pipeline: 32 raw IMUs -> protected direct FK -> 42 keypoints")

    projector = SkeletonRenderer()
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

    output_path = args.out or (
        PROJECT_ROOT / "data/keypoints_21_bimanual" /
        datetime.now().strftime("%Y%m%d_%H%M%S") / "chunk-000.parquet")
    recording_session: BimanualRecordingSession | None = None
    started = time.monotonic()
    next_sample = started
    processed_frame_index = 0
    fps_times = {"left": [], "right": []}
    # Device arrival rate, tracked beside the solve rate so a host that cannot
    # keep up is not mistaken for a glove that is not delivering.
    device_totals = {"left": 0, "right": 0}
    device_samples = {"left": [], "right": []}
    try:
        while True:
            now = time.monotonic()
            for side, runtime in runtimes.items():
                for tactile_frame in runtime.tactile_stream.poll():
                    runtime.latest_tactile_raw = tactile_frame.samples.copy()
                    processed, _ = runtime.tactile_pre.process(
                        tactile_frame.samples)
                    runtime.latest_tactile = processed
                    if viewer:
                        viewer.set_tactile_side(side, processed)
                raw_frames = runtime.stream.poll()
                if raw_frames:
                    # Count arrivals before the backlog is discarded below;
                    # past this point the device rate is unobservable.
                    device_totals[side] += len(raw_frames)
                    device_samples[side].append((now, device_totals[side]))
                    device_samples[side] = [
                        sample for sample in device_samples[side]
                        if now - sample[0] <= 1.0]
                    # Real-time: drop the buffered backlog and keep only the
                    # newest frame, so the view never lags behind the USB rate.
                    raw_frames = raw_frames[-1:]
                for raw_frame in raw_frames:
                    if raw_frame.sequence == runtime.last_seq:
                        continue
                    runtime.last_seq = raw_frame.sequence
                    runtime.latest_raw_imu = raw_frame.quaternions_xyzw.copy()
                    runtime.latest_raw_present = raw_frame.present_mask.copy()
                    runtime.latest_raw_valid = raw_frame.valid_mask.copy()
                    runtime.latest_raw_device_timestamp_us = (
                        raw_frame.device_timestamp_us)
                    keypoints = runtime.solver.process(raw_frame)
                    sample_s = keypoints.timestamp_us / 1_000_000.0
                    runtime.latest_joints = keypoints.joints_m
                    runtime.latest_smoothed = runtime.smoother.update(
                        keypoints.joints_m, sample_s)
                    runtime.latest_imu = keypoints.imu_xyzw
                    runtime.latest_host_s = now
                    runtime.missing = [
                        f"IMU-{index}" for index in np.flatnonzero(
                            ~np.isfinite(keypoints.sensor_age_s))
                    ]
                    runtime.safety = keypoints.status.safety_intervened
                    runtime.contact = keypoints.status.active_contact
                    fps_times[side].append(now)
                    fps_times[side] = fps_times[side][-60:]
                    samples = device_samples[side]
                    span = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
                    runtime.device_fps = (
                        (samples[-1][1] - samples[0][1]) / span
                        if span > 0.0 else 0.0)

            ready = all(runtime.latest_joints is not None for runtime in runtimes.values())
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
                fixed_rotations = fixed_bimanual_display_rotations(
                    initial, palm_target_bases)
                recording_metadata = {
                    "mode": "bimanual",
                    "sides": ["left", "right"],
                    "sample_fps": float(args.fps),
                    "devices": {
                        side: {
                            "usb_serial": registry[side]["usb_serial"],
                            "hardware_id": registry[side]["hardware_id"],
                            "channel_to_hand": registry[side]["channel_to_hand"],
                        }
                        for side in ("left", "right")
                    },
                    "calibration_files": {
                        side: str(calib_paths[side].resolve())
                        for side in ("left", "right")
                    },
                    "view": view_state_metadata(view_state),
                    "display": {
                        "recorded_source": "raw",
                        "separation_m": separation_m,
                        "palm_target_bases": palm_target_bases,
                        "fixed_rotation_matrices": np.stack([
                            rotation.as_matrix() for rotation in fixed_rotations
                        ]),
                        "stabilize_palm_bases": False,
                        "left_root_rotvec_sign": left_root_rotvec_sign,
                        "default_config": str(args.display_config.resolve()),
                    },
                    "kinematics": {
                        "type": runtime_backend["backend"],
                        "geometry": str(Path(args.geometry).resolve()),
                    },
                    "public_interfaces": [
                        "RawImuStream", "TactileStream", "HandSolver.process"
                    ],
                }
                recording_session = BimanualRecordingSession(
                    output_path, args.episode_index, args.task_index,
                    args.checkpoint_frames,
                    serials={
                        side: registry[side]["usb_serial"]
                        for side in ("left", "right")
                    },
                    metadata=recording_metadata,
                    enabled=not args.no_save,
                )
                if args.record_on_start:
                    recording_session.start(now)
                    print(
                        f"Automatic bimanual recording started: {recording_session.output_path}")
                if not args.no_display:
                    backend_labels = {
                        "hand_pinky_plus_2mm": "HAND pinky +2 mm LOCAL",
                        "retargeted_direct": "measured direct / calibrated HAND orientation",
                        "fitted_direct": "fitted project direct FK",
                        "direct_fk": "direct FK",
                    }
                    backend_label = backend_labels[runtime_backend["backend"]]
                    viewer = BimanualViewer(
                        view_state, initial,
                        f"STM32 bimanual live 21 joints / {backend_label}",
                        separation_m=separation_m,
                        palm_target_bases=palm_target_bases,
                        left_root_rotvec_sign=left_root_rotvec_sign,
                        display_config_path=None,
                        recording_enabled=not args.no_save,
                        recording_state=recording_session.state,
                        display_mode=args.display_mode,
                        initial_mesh_vertices=_latest_solver_mesh_vertices(
                            runtimes["right"].solver),
                        redraw_interval_s=1.0 / max(args.fps, 1.0))
                startup_initialized = True
            if fresh and now >= next_sample:
                result = make_result(runtimes, projector)
                if recording_session is not None:
                    recording_session.add(now, result, runtimes)
                if viewer:
                    statuses = {}
                    for side in ("left", "right"):
                        times = fps_times[side]
                        fps = ((len(times)-1) / max(times[-1]-times[0], 1e-6)
                               if len(times) >= 2 else 0.0)
                        statuses[side] = {
                            "fps": fps,
                            "device_fps": float(
                                getattr(runtimes[side], "device_fps", 0.0)),
                            "missing": runtimes[side].missing,
                            "safety": runtimes[side].safety,
                            "contact": runtimes[side].contact,
                        }
                    viewer.update_both(
                        runtimes["left"].latest_joints,
                        runtimes["right"].latest_joints,
                        (recording_session.frame_count
                         if recording_session is not None else 0), statuses,
                        mesh_slots=[
                            _latest_solver_mesh_vertices(runtimes["left"].solver),
                            _latest_solver_mesh_vertices(runtimes["right"].solver),
                        ])
                processed_frame_index += 1
                next_sample = now + 1.0 / max(args.fps, 1.0)
                if args.max_frames and processed_frame_index >= args.max_frames:
                    break
            elif not ready and now - started > args.startup_timeout:
                missing = [side for side, runtime in runtimes.items()
                           if runtime.latest_joints is None]
                raise RuntimeError(f"Timed out waiting for bimanual data: {missing}")

            if viewer:
                keep_running = viewer.tick()
                action = viewer.consume_record_action()
                if action == "start" and recording_session is not None:
                    if recording_session.start(time.monotonic()):
                        viewer.set_recording_state("recording", 0)
                        print(
                            f"Bimanual recording started: {recording_session.output_path}")
                elif action == "stop" and recording_session is not None:
                    saved = recording_session.stop_and_save(
                        view_state_metadata(view_state))
                    viewer.set_recording_state(
                        "saved", recording_session.frame_count)
                    if saved is not None:
                        print(f"Bimanual recording saved: {saved}")
                    else:
                        print("No valid bimanual frames; no file was created.")
                if not keep_running:
                    break
            time.sleep(0.002)
    finally:
        for runtime in runtimes.values():
            runtime.stream.stop()
        if (recording_session is not None
                and recording_session.state == "recording"):
            saved = recording_session.stop_and_save(
                view_state_metadata(view_state))
            if saved is not None:
                print(f"Bimanual recording saved: {saved}")
        if viewer:
            viewer.close()
    print(f"Bimanual run finished: {processed_frame_index} preview frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
