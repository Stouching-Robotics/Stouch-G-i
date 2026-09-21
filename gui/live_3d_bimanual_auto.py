#!/usr/bin/env python3
"""Adaptive two-STM32 gloves -> HAND2mm-calibrated skeletons -> live 3D.

Runs exactly like ``live_3d_bimanual`` (same persistent display config: wrist
separation, palm bases and viewpoint are shared), but does not require both
gloves to be plugged in:

* right only -> show the right hand at its bimanual anchor position
* left  only -> show the left  hand at its bimanual anchor position
* both      -> identical to the original bimanual live view

A side that is not connected is simply not drawn; the connected hand keeps the
original left/right wrist separation.  An error is raised only when no glove at
all is detected.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import multiprocessing
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
    # Keep mutable configuration, calibration choices and recordings beside
    # the portable exe.  PyInstaller's _MEIPASS is temporary in one-file mode
    # and anything written there disappears when the process exits.
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    BUNDLE_ROOT = PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)
DEFAULT_BIMANUAL_DISPLAY_CONFIG = PROJECT_ROOT / "config" / "bimanual_display_config.json"
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "calibration"
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from gui import APP_VERSION  # noqa: E402
from gui.calibration_bindings import (  # noqa: E402
    DEFAULT_CALIBRATION_BINDINGS,
    load_calibration_bindings,
    remembered_calibration,
    save_calibration_binding,
)
from gui.calibration_selector import select_calibration_files  # noqa: E402
from gui.live_3d import (  # noqa: E402
    TACTILE_SCALE_MAX, TACTILE_SCALE_MIN,
    Live3DViewer, LiveViewState, VIEW3D_CONFIG_PATH,
    align_left_to_right_reference, apply_hand_display_rotation,
    hand_display_basis, _latest_solver_mesh_vertices,
    place_hands_at_wrist_anchors, put_text, text_size)
from gui.rendering.tactile import (  # noqa: E402
    HAND_REGIONS_LEFT, HAND_REGIONS_RIGHT, render_hand,
)
from gui.rendering.tactile_overlay import (  # noqa: E402
    draw_bimanual_tactile_overlay as draw_full_matrix_tactile_overlay,
    tactile_card_rects, tactile_cell_for_scale, tactile_grip_rects,
    tactile_scale_for_cell,
)
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
from glove_io.bluetooth_dongle import reboot_bluetooth_dongle  # noqa: E402
from glove_io.device_registry import (  # noqa: E402
    detect_unbound_glove_links, link_kind_for_device, set_glove_serial,
)
from glove_io.tactile_processing import TactilePreprocessor  # noqa: E402
from sdk import (  # noqa: E402
    DeviceManager, DeviceNotFoundError, HandSolver, RawImuStream,
    get_version)
from common.frame_pacing import recent_batch  # noqa: E402
from common.i18n import L, set_lang  # noqa: E402
from common.usb_cdc import (  # noqa: E402
    USB_FULL_SEQUENCE_FROM,
    display_firmware_version, firmware_real_version, mag_gate_met)

DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "glove_devices.json"
DEFAULT_GEOMETRY_PATH = (
    BUNDLE_ROOT / "assets/hand_geometry/hand_measured_runtime_v1.json")

_FORCE_REGION_NAMES = ("thumb", "index", "middle", "ring", "pinky", "palm")
_FORCE_CURVE_RGB = {
    "thumb": (255, 107, 108),
    "index": (255, 190, 85),
    "middle": (78, 214, 128),
    "ring": (72, 165, 255),
    "pinky": (184, 118, 255),
    "palm": (54, 184, 235),
}
_FORCE_CURVE_BGR = {
    name: tuple(reversed(color)) for name, color in _FORCE_CURVE_RGB.items()
}
_FORCE_REGION_SHORT = {
    "thumb": "THM",
    "index": "IDX",
    "middle": "MID",
    "ring": "RNG",
    "pinky": "PNK",
    "palm": "PALM",
}
_FORCE_HISTORY_SECONDS = 10.0
_FORCE_HISTORY_MAX_SAMPLES = 1200
# Baseline acquisition waits a fixed settle delay after the button is pressed
# before collecting the median frames, so a still-settling hand cannot seed a
# bad empty-glove baseline.
_BASELINE_SETTLE_DELAY_S = 2.0


def _tactile_force_totals(
        frame: np.ndarray, side: str, threshold: float = 0.0) -> np.ndarray:
    """Sum the processed tactile cells in each anatomical hand region."""

    values = np.asarray(frame, dtype=np.float32).reshape(16, 16)
    values = np.maximum(
        0.0, np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
    if threshold > 0.0:
        values = np.where(values >= float(threshold), values, 0.0)
    regions = HAND_REGIONS_LEFT if side == "left" else HAND_REGIONS_RIGHT
    return np.asarray([
        float(values[np.ix_(regions[name]["rows"], regions[name]["cols"])].sum())
        for name in _FORCE_REGION_NAMES
    ], dtype=np.float32)


def _nice_force_axis_max(value: float) -> float:
    """Round an autoscaled force axis up to a readable 1/2/5 multiple."""

    value = max(1.0, float(value))
    exponent = 10.0 ** math.floor(math.log10(value))
    fraction = value / exponent
    step = 1.0 if fraction <= 1.0 else 2.0 if fraction <= 2.0 else 5.0
    if fraction > 5.0:
        step = 10.0
    return step * exponent


def _compact_force_value(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _render_force_trend_card(
        side: str, data, scale: float, now: float) -> tuple[np.ndarray, np.ndarray]:
    """Render six independent force axes in a compact 3-row x 2-column card."""

    card_w = min(520, max(400, int(round(720.0 * float(scale)))))
    card_h = min(330, max(225, int(round(400.0 * float(scale)))))
    top_color = np.asarray((48, 35, 21), np.float32)
    bottom_color = np.asarray((22, 27, 38), np.float32)
    mix = np.linspace(0.0, 1.0, card_h, dtype=np.float32)[:, None]
    rows = (top_color[None, :] + (bottom_color - top_color)[None, :] * mix)
    card = np.empty((card_h, card_w, 3), np.uint8)
    card[...] = rows.astype(np.uint8)[:, None, :]

    accent = (240, 201, 76) if side == "left" else (84, 180, 255)
    cv2.circle(card, (18, 17), 9, accent, -1, cv2.LINE_AA)
    cv2.putText(card, "L" if side == "left" else "R", (14, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 28, 38), 1,
                cv2.LINE_AA)
    cv2.putText(card, "FORCE TREND  /  REGION SUM", (34, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (221, 228, 239), 1,
                cv2.LINE_AA)
    cv2.line(card, (10, 33), (card_w - 11, 33), (84, 72, 61), 1,
             cv2.LINE_AA)

    timestamps = np.empty((0,), dtype=np.float64)
    totals = np.empty((0, len(_FORCE_REGION_NAMES)), dtype=np.float32)
    if data is not None:
        timestamps, totals = data

    rows_top = 38
    rows_bottom = card_h - 9
    row_gap = 3
    column_gap = 4
    grid_rows, grid_columns = 3, 2
    row_h = (rows_bottom - rows_top - row_gap * (grid_rows - 1)) / grid_rows
    column_w = (card_w - 10 - column_gap * (grid_columns - 1)) / grid_columns
    history_start = now - _FORCE_HISTORY_SECONDS
    for region_index, name in enumerate(_FORCE_REGION_NAMES):
        grid_row, grid_column = divmod(region_index, grid_columns)
        x0 = int(round(5 + grid_column * (column_w + column_gap)))
        x1 = int(round(x0 + column_w))
        y0 = int(round(rows_top + grid_row * (row_h + row_gap)))
        y1 = int(round(y0 + row_h))
        cv2.rectangle(card, (x0, y0), (x1, y1), (31, 38, 51), -1)
        cv2.rectangle(card, (x0, y0), (x1, y1), (61, 70, 86), 1)

        color = _FORCE_CURVE_BGR[name]
        cv2.putText(card, _FORCE_REGION_SHORT[name], (x0 + 5, y0 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        series = (totals[:, region_index] if totals.size
                  else np.empty((0,), dtype=np.float32))
        peak = float(np.max(series)) if series.size else 0.0
        y_max = _nice_force_axis_max(peak * 1.05)
        if series.size:
            current_text = "NOW " + _compact_force_value(float(series[-1]))
            (text_w, _), _ = cv2.getTextSize(
                current_text, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            cv2.putText(card, current_text, (x1 - text_w - 5, y0 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (173, 188, 209), 1,
                        cv2.LINE_AA)

        plot_left, plot_right = x0 + 50, x1 - 5
        plot_top = y0 + 17
        plot_bottom = max(plot_top + 3, y1 - 12)
        for division in range(3):
            ratio = division / 2.0
            y = int(round(plot_bottom - ratio * (plot_bottom - plot_top)))
            cv2.line(card, (plot_left, y), (plot_right, y),
                     (79, 87, 103) if division == 0 else (48, 58, 74), 1,
                     cv2.LINE_AA)
        for division in range(3):
            ratio = division / 2.0
            x = int(round(plot_left + ratio * (plot_right - plot_left)))
            cv2.line(card, (x, plot_top), (x, plot_bottom),
                     (48, 58, 74), 1, cv2.LINE_AA)

        max_text = _compact_force_value(y_max)
        cv2.putText(card, max_text, (x0 + 4, plot_top + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (151, 164, 184), 1,
                    cv2.LINE_AA)
        cv2.putText(card, "0", (x0 + 35, plot_bottom + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (151, 164, 184), 1,
                    cv2.LINE_AA)
        cv2.putText(card, "-10s", (plot_left, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, (139, 153, 174), 1,
                    cv2.LINE_AA)
        (zero_w, _), _ = cv2.getTextSize(
            "0s", cv2.FONT_HERSHEY_SIMPLEX, 0.24, 1)
        cv2.putText(card, "0s", (plot_right - zero_w, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, (139, 153, 174), 1,
                    cv2.LINE_AA)

        if series.size:
            xs = plot_left + np.clip(
                (timestamps - history_start) / _FORCE_HISTORY_SECONDS,
                0.0, 1.0) * (plot_right - plot_left)
            ys = plot_bottom - np.clip(
                series / y_max, 0.0, 1.0) * (plot_bottom - plot_top)
            points = np.rint(np.column_stack([xs, ys])).astype(np.int32)
            if len(points) >= 2:
                cv2.polylines(card, [points], False, color, 2, cv2.LINE_AA)
            elif len(points) == 1:
                cv2.circle(card, tuple(points[0]), 2, color, -1, cv2.LINE_AA)

    mask = _rounded_card_mask(card_h, card_w)
    inner = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    card[mask > inner] = np.asarray((100, 84, 70), np.uint8)
    card[1:3, 13:card_w - 13] = accent
    return card, mask


def _draw_force_trend_overlay(
        img: np.ndarray, snapshot: dict, sides, scale: float) -> np.ndarray:
    """Draw force cards in exactly the left/right tactile-panel positions."""

    now = time.monotonic()
    for side in ("left", "right"):
        if side not in sides:
            continue
        card, mask = _render_force_trend_card(
            side, snapshot.get(side), scale, now)
        card_h, card_w = card.shape[:2]
        x0 = 12 if side == "left" else img.shape[1] - card_w - 12
        y0 = img.shape[0] - 60 - card_h
        roi = img[y0:y0 + card_h, x0:x0 + card_w]
        cv2.copyTo(card, mask, roi)
    return img




def _show_frozen_error(message: str) -> None:
    """Show startup failures that would otherwise be invisible in a GUI exe."""
    if not getattr(sys, "frozen", False):
        return
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv[:1])
        QMessageBox.critical(None, "Stouch Glove Live", str(message))
    except Exception as dialog_error:
        print(f"[Dialog error] {dialog_error}", file=sys.stderr)


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
    # The right hand goes through the same per-frame remap slot as the left,
    # but its default is the no-op sign set: the right is the reference side
    # of the shared display frame and needs no handedness mirror by default.
    right_root_sign = np.asarray(
        payload.get("right_root_rotvec_sign", [1.0, 1.0, 1.0]),
        dtype=float).reshape(3)
    if (not np.isfinite(right_root_sign).all()
            or not np.all(np.isin(right_root_sign, [-1.0, 1.0]))):
        raise RuntimeError(
            f"Right root rotation signs must be -1/+1: {right_root_sign.tolist()}")
    result["right_root_rotvec_sign"] = right_root_sign
    return result


def save_bimanual_display_config(
        path: Path, palm_target_bases: np.ndarray, separation_m: float,
        source_session: str = "runtime-relearn",
        view: dict | None = None,
        left_root_rotvec_sign: np.ndarray | None = None,
        right_root_rotvec_sign: np.ndarray | None = None) -> Path:
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
    if right_root_rotvec_sign is not None:
        right_sign = np.asarray(right_root_rotvec_sign, dtype=float).reshape(3)
        if (not np.isfinite(right_sign).all()
                or not np.all(np.isin(right_sign, [-1.0, 1.0]))):
            raise RuntimeError(
                f"Right root rotation signs must be -1/+1: {right_sign.tolist()}")
        payload["right_root_rotvec_sign"] = right_sign.tolist()
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
        hands: np.ndarray, target_bases: np.ndarray) -> list[R | None]:
    """Compute one startup rotation per hand; later root motion is preserved.

    A slot whose joints are not finite (a glove that is not connected) gets a
    ``None`` entry instead of a rotation, so the base viewer simply does not
    display-rotate that hand.
    """
    slots = np.asarray(hands, dtype=np.float32).reshape(2, 21, 3)
    targets = _validate_palm_target_bases(target_bases)
    rotations: list[R | None] = []
    for slot in range(2):
        if not np.isfinite(slots[slot]).all():
            rotations.append(None)
            continue
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


# Per-channel "flagged IMU" detection.  Every IMU frame carries the firmware's
# own 16-bit per-channel health flag (``RawImuFrame.present_mask``, one bit per
# physical channel); a clear bit means that channel's BNO055 is re-initialising,
# quarantined, or absent.  The host reads that flag on *every* frame and derives
# nothing of its own from the quaternions, so a channel that recovers -- either
# when the firmware's re-init succeeds or when the glove is power-cycled --
# stops being reported the moment its bit comes back.  The verdict is
# deliberately not sticky: a latched host-side one outlived the fault it
# described and kept warning about an IMU that was working again.
# One clear frame is routine (a dropped read), so a channel has to stay clear
# for this many consecutive frames before it is called out.
_FLAGGED_IMU_FRAMES = 3


class ImuFaultDetector:
    """Report the physical IMU channels the firmware currently flags as bad.

    ``present_mask`` is the frame's 16-bit health flag in firmware physical
    order (index 0-15); a flagged channel is reported by its physical channel
    index, which maps directly to the sensor position: 0 = wrist, 1-3 = thumb,
    4-6 = index, 7-9 = middle, 10-12 = ring, 13-15 = pinky.
    """

    def __init__(self, n_channels: int = 16):
        self._n_channels = int(n_channels)
        self._clear_streak = np.zeros(self._n_channels, dtype=np.int32)

    def update(self, present_mask: np.ndarray) -> list[int]:
        """Feed one frame's 16-bit health flag; return flagged channel indices."""

        flagged = ~np.asarray(present_mask, dtype=bool).reshape(self._n_channels)
        self._clear_streak = np.where(flagged, self._clear_streak + 1, 0)
        return [
            int(index)
            for index in np.flatnonzero(self._clear_streak >= _FLAGGED_IMU_FRAMES)
        ]


# How big an orientation step in one frame counts as a cliff rather than motion.
#
# The BNO055 fuses accelerometer, gyroscope and magnetometer.  When the field it
# sits in changes abruptly the heading it reports *steps*: the correction the
# magnetometer was applying is replaced by a different one, and the reported
# attitude moves discontinuously.  A wrist snapping as fast as it can turns
# roughly 10-15 deg per 1/80 s frame, so the bar sits above that.
#
# This is the one number to tune on hardware.  Set it too low and ordinary fast
# hand motion asks the user to draw circles; too high and a real disturbance is
# missed.  The detector returns the step it measured, so a tuning session can
# read the numbers out of a real one, but the live view reports nothing at all:
# it acts on the verdict and stays quiet about it.
MAG_TRIGGER_JUMP_DEG = 30.0


def _quat_step_deg(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Angle in degrees between two (N, 4) quaternion arrays, per row.

    ``q`` and ``-q`` are the same rotation, so the dot product is taken as an
    absolute value before the half-angle.  Non-finite rows come back as NaN,
    which the caller drops.
    """

    dot = np.abs(np.sum(np.asarray(before, dtype=np.float64)
                        * np.asarray(after, dtype=np.float64), axis=1))
    with np.errstate(invalid="ignore"):
        return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


class ImuJumpDetector:
    """Report a cliff-edge step in the raw quaternion stream.

    Per physical IMU, the angle between this frame's quaternion and the previous
    frame's.  The largest of those is the frame's step: a disturbance moves every
    IMU that sits in the same field, and a single sensor that jumps on its own is
    worth reporting too.

    Raw quaternions rather than the solved pose, because the solve smooths and
    temporal smoothing is built precisely to hide a step.

    ``update`` returns the step it measured alongside whether it crossed the bar,
    so a caller that wants the numbers a threshold has to be chosen from can have
    them; the live view takes only the verdict and stays quiet about the step.
    Only channels *this* frame marks valid take part: an implausible reading --
    zeros or a non-finite row -- is dropped rather than compared.  The previous
    frame's mask is not kept, so a channel that was implausible last frame and is
    plausible now is compared against that stale reading instead.
    """

    def __init__(self, threshold_deg: float = MAG_TRIGGER_JUMP_DEG,
                 n_channels: int = 16):
        self.threshold_deg = float(threshold_deg)
        self._n_channels = int(n_channels)
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the last frame; the next one has nothing to be compared to."""

        self._previous = None

    def update(self, quaternions: np.ndarray,
               valid_mask: np.ndarray) -> tuple[bool, float]:
        """Feed one raw IMU frame; return ``(jumped, step_degrees)``."""

        current = np.asarray(quaternions, dtype=np.float64).reshape(
            self._n_channels, 4)
        valid = np.asarray(valid_mask, dtype=bool).reshape(self._n_channels)
        previous, self._previous = self._previous, current.copy()
        if previous is None:
            return False, 0.0

        steps = _quat_step_deg(previous, current)
        usable = valid & np.isfinite(steps)
        if not usable.any():
            return False, 0.0
        step = float(np.max(steps[usable]))
        return step >= self.threshold_deg, step


class MagCalibrationMonitor:
    """Decide when to ask for magnetometer circles, and when the spot is bad.

    Per hand, driven by two streams: the type-0x05 calibration levels and the
    raw attitude.  The gate is three quarters of the channels the newest 0x05
    frame carried reading 3 (``mag_gate_met``, decided on the snapshot because
    that is where the freshness mask is).

    **A glove below the gate is not prompted for.**  The trigger is a cliff-edge
    step in the attitude stream (:class:`ImuJumpDetector`): the magnetometer
    correction being replaced by a different one moves the reported attitude
    discontinuously, so a step is what "the field around the glove changed"
    looks like from outside.  On a trigger the gate is read, and if it fails the
    user gets ``CIRCLE_PROMPT_S`` to draw circles; reaching the gate inside the
    window stops the prompt on that frame.  Still short when the window closes
    means circles were not enough, which in practice means the ambient field is
    too disturbed for the sensor to separate hard iron from the earth's field.

    Prompting on the step rather than on the level is what keeps the card quiet:
    a glove that has just been powered on sits below the gate for its first
    10-30 s of motion (the BNO055 re-calibrates from scratch on every power-on
    reset, MAG_TELEMETRY.md 3), and a monitor that spoke up for that would hold
    a card over the view at every single start.

    The bad-environment verdict is announced, not repeated: it is shown for
    ``ENVIRONMENT_PROMPT_S`` and then the monitor falls silent.  A message that
    never goes away is worse than no message -- it is the same countdown a user
    has already failed to satisfy.  A fresh trigger re-arms it, so carrying the
    glove somewhere else is asked about again.

    Judging is held off until ``judge_ready`` -- IMU warm-up and the startup
    hand-direction capture.  The fusion is settling through those, so a step
    would say nothing about the field.

    The magnetic-drift detection that used to live here is gone: type 0x05 no
    longer carries the field vectors, so there is no |B| to record.
    """

    # 画圆窗口；超时仍未达标判为环境干扰。
    CIRCLE_PROMPT_S = 30.0
    # 环境干扰结论只提示这么久，之后彻底闭嘴（不达标也不再重复）。
    ENVIRONMENT_PROMPT_S = 3.0

    def __init__(self):
        # When the current "below the gate" streak began.
        self._needs_since: float | None = None
        # The newest frame's (at 3, carried), shown next to the countdown.
        self._counts = (0, 0)
        self._prompt: dict | None = None

    def _reset(self) -> None:
        """Forget the streak and re-arm: the next trigger starts a new window.

        Used both when the gate is met and while judging is held off, so a
        re-capture or a recovered glove is announced again if it drops later.
        """

        self._needs_since = None
        self._counts = (0, 0)
        self._prompt = None

    def _phase(self, now: float) -> dict | None:
        """The prompt implied by the streak and the clock, re-derived per call.

        Split out because it is a pure function of time while the streak runs:
        the frame sequence only advances every 2 s, but the counts are seconds,
        so a prompt recomputed only on a new frame would hold the 3 s message
        for up to 4 s and tick its countdown in 2 s steps.
        """

        if self._needs_since is None:
            self._prompt = None
            return None
        elapsed = now - self._needs_since
        ready, carried = self._counts
        if elapsed < self.CIRCLE_PROMPT_S:
            self._prompt = {"kind": "circles",
                            "remaining_s": self.CIRCLE_PROMPT_S - elapsed,
                            "ready": ready, "carried": carried}
        elif elapsed < self.CIRCLE_PROMPT_S + self.ENVIRONMENT_PROMPT_S:
            self._prompt = {"kind": "environment", "remaining_s": 0.0,
                            "ready": ready, "carried": carried}
        else:
            # Said its piece; stay quiet until the gate is met again.
            self._prompt = None
        return self._prompt

    def update(self, now: float, ready: bool | None, counts=(0, 0),
               triggered: bool = False,
               judge_ready: bool = True) -> dict | None:
        """Act on this frame; return the prompt, if any.

        The gate itself is decided where the freshness mask is -- in
        :class:`MagTelemetrySnapshot` -- and arrives here as ``ready``, with
        ``counts`` = ``(at 3, carried)`` for the same frame.  ``ready is None``
        means there is nothing to judge by: firmware that sends no type 0x05.

        ``triggered`` is a cliff-edge step in the attitude stream
        (:class:`ImuJumpDetector`) -- the event this whole thing waits for.  A
        glove sitting below the gate is *not* prompted for; only a glove that
        just moved discontinuously is, because that is what a change in the
        field it sits in looks like.  A second trigger restarts the window, so
        a user who carries the glove somewhere else is asked again.

        The countdown is re-derived from the clock on every call rather than
        advanced per frame: it is a function of time, and the caller is not the
        only clock.  ``judge_ready`` is False while warm-up or the startup
        direction capture is still running, where the fusion is settling and a
        step would not mean anything about the field.

        Returns ``None``, or a plain ``{"kind", "remaining_s", "ready",
        "carried"}`` dict -- primitives only, since it is published through a
        process queue.
        """

        if not judge_ready:
            # Holding off is not a verdict about the glove, so nothing is
            # prompted and the window starts fresh once it ends.
            self._reset()
            return None

        if ready is None:
            # No 0x05 on this firmware; the caller keeps its own prompt.
            return self._prompt

        if ready:
            # Gate met: stop prompting on this frame, however it started.
            self._reset()
            return None

        if triggered:
            # A step in the attitude is the only thing that asks for circles, so
            # this is what starts or restarts the window.
            self._needs_since = now
        elif self._needs_since is None:
            # Below the gate and nothing happened: stay quiet.  This is the
            # state a glove is in for its whole first minute, and prompting for
            # it is what made the card feel permanent.
            return None

        self._counts = counts
        return self._phase(now)


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
    latest_mesh_vertices: np.ndarray | None = None
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
    # No warm-up fields here on purpose.  Warm-up is reported per frame in
    # ``keypoints.status.details`` and published from there; it is not something
    # this object tracks or knows.  It used to carry ``warming_up`` /
    # ``warmup_completed`` / ``warmup_timed_out`` / ``warmup_remaining_s``
    # defaults that nothing ever wrote, and reading one of them as if it meant
    # something held the magnetometer gate shut forever -- ``getattr`` with a
    # default makes a field nobody assigns look like a real ``False``.
    fps: float = 0.0
    tactile_version: int = 0


class _AbsentHandRuntime:
    """Stand-in runtime for a side whose glove is not connected.

    Keeps ``runtimes`` addressable as ``runtimes["left"]`` / ``runtimes["right"]``
    so the recorder and result builders can treat both slots uniformly; every
    sensor field stays ``None`` and the keypoints become NaN.
    """

    def __init__(self, side: str):
        self.side = side
        self.missing = [f"NO-DEVICE-{side.upper()}"]

    latest_joints = None
    latest_mesh_vertices = None
    latest_smoothed = None
    latest_imu = None
    latest_tactile = None
    latest_tactile_raw = None
    latest_raw_imu = None
    latest_raw_present = None
    latest_raw_valid = None
    latest_mag_levels = None
    latest_mag_unread = None
    latest_mag_prompt = None
    latest_raw_device_timestamp_us = None
    last_seq = -1
    safety = False
    contact = None
    broken = []
    warming_up = False
    warmup_remaining_s = 0.0
    warmup_completed = False
    warmup_timed_out = False
    latest_host_s = 0.0
    fps = 0.0
    device_fps = 0.0
    # Delivery shape plus the parser's cumulative error tally, as published by
    # the hand process.  ``None`` until the first solve message arrives.
    latest_link_health: dict | None = None
    tactile_version = 0
    firmware_version = None
    latency = None


class PublishedHandState:
    """Parent-side mirror of one hand process's published ``latest_*`` state.

    Each hand's solver process copy-on-publishes through an inter-process
    queue; the main loop drains it keep-newest under ``state_lock`` and the
    recorder thread reads the same fields under the same lock.  Every field
    starts at the same "no data yet" value as :class:`_AbsentHandRuntime`, so
    both classes can back ``runtimes`` interchangeably.
    """

    def __init__(self, side: str):
        self.side = side
        self.latest_joints = None
        self.latest_mesh_vertices = None
        self.latest_smoothed = None
        self.latest_imu = None
        self.latest_raw_imu = None
        self.latest_raw_present = None
        self.latest_raw_valid = None
        # Per-IMU BNO055 magnetometer levels, or ``None`` on firmware that
        # sends no type 0x05.  A never-read channel is itself ``None``, so
        # "unknown" and "level 0" stay distinguishable.
        self.latest_mag_levels = None
        # The IMUs the newest 0x05 frame did *not* report this round.  Not a
        # fault: the device carries about twelve of the sixteen per frame and
        # which ones varies, so these are the channels the gate's denominator
        # left out.
        self.latest_mag_unread = None
        # The calibration monitor's verdict for this hand: ``None`` when there
        # is nothing to prompt, otherwise the plain dict it published.
        self.latest_mag_prompt = None
        self.latest_raw_device_timestamp_us = None
        self.latest_tactile = None
        self.latest_tactile_raw = None
        self.missing = []
        self.broken = []
        self.safety = False
        self.contact = None
        self.warming_up = False
        self.warmup_remaining_s = 0.0
        self.warmup_completed = False
        self.warmup_timed_out = False
        self.fps = 0.0
        self.device_fps = 0.0
        self.latest_link_health = None
        self.latest_host_s = 0.0
        self.tactile_version = 0
        self.firmware_version = None
        self.latency = None


def _rounded_card_mask(height: int, width: int, radius: int = 12) -> np.ndarray:
    """Return an antialiased-looking uint8 mask for a small rounded card."""

    height, width = int(height), int(width)
    radius = max(1, min(int(radius), height // 2, width // 2))
    mask = np.zeros((height, width), np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    for center in (
            (radius, radius), (width - radius - 1, radius),
            (radius, height - radius - 1),
            (width - radius - 1, height - radius - 1)):
        cv2.circle(mask, center, radius, 255, -1, cv2.LINE_AA)
    return mask


def draw_bimanual_tactile_overlay(
        img: np.ndarray,
        tactile_frames: dict[str, np.ndarray | None],
        threshold: float = 0.0,
        scale: float = 0.6,
        baseline_corrected: bool = True,
        linear_fitted: bool = False) -> np.ndarray:
    """Compatibility wrapper for the full-matrix tactile UI."""
    return draw_full_matrix_tactile_overlay(
        img, tactile_frames, threshold=threshold, scale=scale,
        baseline_corrected=baseline_corrected,
        linear_fitted=linear_fitted)


def _magnetometer_prompt_card(side_status) -> tuple[str, str] | None:
    """The calibration monitor's card, or ``None`` when it has nothing to say.

    Driven by the type-0x05 calibration levels: the monitor owns the deadline
    and the per-frame counts, so every glove that reports levels gets this card.
    Firmware older than v1.2.13 sends no 0x05 and gets no card at all -- its
    only magnetometer signal was bit 15 of the sequence word, which is not a
    verdict (it is the counter's own high bit) and is no longer read here.
    """

    prompts = {side: side_status.get(side, {}).get("mag_prompt")
               for side in ("left", "right")}
    prompts = {side: prompt for side, prompt in prompts.items() if prompt}
    if not prompts:
        return None

    if len(prompts) == 2:
        hand_text = L("双手", "BOTH HANDS")
    elif "left" in prompts:
        hand_text = L("左手", "LEFT HAND")
    else:
        hand_text = L("右手", "RIGHT HAND")

    # The most advanced verdict names the card; the countdown below then comes
    # from the worst hand.
    if any(prompt["kind"] == "environment" for prompt in prompts.values()):
        return (
            L(f"{hand_text}磁场环境干扰较大",
              f"{hand_text} MAGNETIC ENVIRONMENT TOO DISTURBED"),
            L("画圆 30 秒后磁力计仍未完成校准，请更换环境",
              "Circles for 30 s did not finish the calibration - move to "
              "another location"),
        )

    remaining = min(prompt["remaining_s"] for prompt in prompts.values())
    # The counts come from the same frame the gate was read from, so the
    # denominator is that frame's own channel count rather than all 16 -- the
    # device only carries about twelve of the sixteen at a time, and a number
    # out of 16 next to a gate that is not out of 16 would just be wrong.
    counts = []
    for side, prompt in prompts.items():
        label = L("左", "L") if side == "left" else L("右", "R")
        counts.append(f"{label} {prompt['ready']}/{prompt['carried']}")
    return (
        L(f"{hand_text}磁力计尚未校准", f"{hand_text} MAGNETOMETER NOT CALIBRATED"),
        L(f"磁力计就绪 {' · '.join(counts)} — 请缓慢画圆"
          f"（剩 {remaining:.0f} 秒）",
          f"Magnetometer ready {' · '.join(counts)} - draw slow circles "
          f"({remaining:.0f} s left)"),
    )


class BimanualViewer(Live3DViewer):
    STARTUP_ALIGNMENT_DURATION_S = 3.0
    STARTUP_ALIGNMENT_SAMPLE_COUNT = 12

    def __init__(self, *args, separation_m=0.30,
                 display_rotations=None, palm_target_bases=None,
                 left_root_rotvec_sign=None,
                 right_root_rotvec_sign=None,
                 display_config_path=DEFAULT_BIMANUAL_DISPLAY_CONFIG,
                 recording_enabled=True, recording_state="idle",
                 present_sides=None, mag_judge_event=None, **kwargs):
        self._bimanual_slots_cache: np.ndarray | None = None
        self._bimanual_mesh_slots_cache: list[np.ndarray | None] | None = None
        self._display_times: list[float] = []
        self._display_fps = 0.0
        self.present_sides = set(present_sides or ())
        # The solver processes own the magnetometer monitor and its countdown,
        # so the hold-off has to be published to them (see
        # ``_publish_mag_judge_gate``).  ``None`` means no such process to tell.
        self._mag_judge_event = mag_judge_event
        self._mag_judge_published: bool | None = None
        self._mag_warmup_seen = False
        # After the IMU warm-up, give the user three seconds to hold each
        # connected hand in the desired straight-ahead pose.  The captured
        # palm direction is compared with the calibrated target basis and the
        # resulting display-only correction is retained for the session.
        self._startup_alignment_started_s: float | None = None
        self._startup_alignment_remaining_s = self.STARTUP_ALIGNMENT_DURATION_S
        self._startup_alignment_completed = False
        self._startup_alignment_corrections = [R.identity(), R.identity()]
        self._startup_alignment_samples = {
            side: deque(maxlen=self.STARTUP_ALIGNMENT_SAMPLE_COUNT)
            for side in ("left", "right")
        }
        self.recording_enabled = bool(recording_enabled)
        self.recording_state = str(recording_state)
        self._record_action_requested: str | None = None
        self.side_status = {
            "left": {"fps": 0.0, "device_fps": 0.0, "link_health": None,
                     "missing": [], "safety": False,
                     "contact": None, "firmware_version": None, "latency": None,
                     "connected": "left" in self.present_sides, "stale": False},
            "right": {"fps": 0.0, "device_fps": 0.0, "link_health": None,
                      "missing": [], "safety": False,
                      "contact": None, "firmware_version": None, "latency": None,
                      "connected": "right" in self.present_sides, "stale": False},
        }
        self.tactile_frames = {"left": None, "right": None}
        # Bluetooth-dongle reboot feedback shown in the HUD (set from the
        # background worker spawned by the on-canvas "Reboot Dongle" button).
        self._dongle_reboot_status = ""
        self.tactile_calibrating_sides = {"left": True, "right": True}
        # Pressure-matrix baseline control (glove-test v1.4 style): a toggle
        # switches between the raw ADC map and the baseline-corrected map, and
        # enabling it auto-collects a fresh empty-glove baseline (median of
        # ``_baseline_target`` raw frames) that the recollect button can refresh.
        self.tactile_raw_frames = {"left": None, "right": None}
        self.tactile_baseline = {"left": None, "right": None}
        self.use_baseline = False
        self._baseline_collecting = False
        self._baseline_buffer = {"left": [], "right": []}
        self._baseline_done = {"left": False, "right": False}
        self._baseline_target = 40
        # Two-phase acquisition: wait a fixed settle delay after the button is
        # pressed, then collect ``_baseline_target`` frames for the median.
        self._baseline_waiting = False
        self._baseline_wait_until = 0.0
        # The fixed tactile-card area is selectable and starts empty.  It can
        # show either the pressure matrix or six independent force trends.
        self.tactile_panel_mode = "off"
        # Set while the user drags a card's resize handle: the corner they
        # grabbed, and the geometry the drag is measured against.
        self._tactile_resize: dict | None = None
        # Samples are retained continuously so selecting the force view
        # immediately shows the preceding 10 seconds.
        self._force_history = {
            side: deque(maxlen=_FORCE_HISTORY_MAX_SAMPLES)
            for side in ("left", "right")
        }
        # Full-hand pressure visualization can be enabled in one of two
        # mutually exclusive modes; it is off by default so startup shows the
        # mesh without pressure points.
        self.pressure_display_mode = "none"
        # Neutral pressure-cell anchors are built once after the MANO faces are
        # loaded.  They are evaluated on the current 778-vertex mesh at draw
        # time, so no pressure work is added to the solver or IPC paths.
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
            if right_root_rotvec_sign is None:
                right_root_rotvec_sign = saved_display["right_root_rotvec_sign"]
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
        self._right_root_rotvec_sign = np.asarray(
            right_root_rotvec_sign
            if right_root_rotvec_sign is not None else [1.0, 1.0, 1.0],
            dtype=float).reshape(3)
        grid_forward = self._palm_target_bases[1, :, 1].copy()
        grid_normal = np.cross(self._anchor_axis, grid_forward)
        grid_normal /= max(float(np.linalg.norm(grid_normal)), 1e-9)
        self._bimanual_grid_basis = np.stack([
            self._anchor_axis, grid_forward, grid_normal], axis=1)
        super().__init__(*args, hand_side="right",
                         display_rotations=fixed_rotations,
                         show_title_in_canvas=False,
                         show_baseline=True,
                         show_surface_color=True,
                         tactile_panel_modes=self._tactile_panel_mode_options(),
                         tactile_panel_mode=self.tactile_panel_mode,
                         show_tactile_toggle=False,
                         show_orientation_button=True,
                         show_selector_button=True,
                         show_dongle_reboot_button=True, **kwargs)
        # The baseline controls follow the tactile panel, and the panel starts
        # off, so they start hidden too -- same rule as
        # ``_on_tactile_panel_mode`` applies when the user picks a view.
        self._qcanvas.set_baseline_controls_visible(self.tactile_panel_mode != "off")
        if self._mesh_faces is not None:
            for side in self.present_sides:
                try:
                    self._pressure_mappers[side] = PressureHandMapper(
                        side, self._mesh_faces,
                        BUNDLE_ROOT / "assets" / "hand")
                except (OSError, RuntimeError, ValueError) as exc:
                    print(
                        f"[Pressure view warning] Cannot prepare {side} hand map: {exc}",
                        file=sys.stderr)
        self._relative_reference_slots = self._display_slots().copy()
        # Window title bar is updated once the firmware version is known.
        self._window_title_shown = self._qcanvas.windowTitle()
        # Last (left, right) firmware pair for which the terminal mismatch
        # warning was printed; None while versions match or are unknown.
        self._fw_mismatch_reported = None
        self._redraw_requested = True

    def _hud_lines(self):
        fw_parts = []
        latency_parts = []
        for side in ("left", "right"):
            status = self.side_status[side]
            # Only a glove that is actually here has a version or a round trip
            # to report.  An absent side would contribute a placeholder that
            # reads like a fault, and a hand that has not answered yet has
            # nothing to say either, so both are simply left out.
            if not status.get("connected"):
                continue
            side_tag = L("左 ", "left ") if side == "left" else L("右 ", "right ")
            version = status.get("firmware_version")
            if version is not None:
                fw_parts.append(
                    side_tag + f"V{display_firmware_version(version)}")
            latency = status.get("latency")
            # A missing measurement is normal on firmware older than v1.2.11,
            # which does not answer the probe; show a dash rather than 0 ms.
            latency_parts.append(
                side_tag + (f"{latency.rtt_ms:.2f} ms"
                            if latency is not None else "—"))
        lines = []
        if fw_parts:
            lines.append(L("固件版本: ", "Firmware: ") + " / ".join(fw_parts))
        if latency_parts:
            lines.append(L("延迟: ", "Latency: ") + " / ".join(latency_parts))
        lines.extend([
            L(f"界面: {self._display_fps:5.1f} FPS",
              f"GUI: {self._display_fps:5.1f} FPS"),
            L(f"双手已录制帧数: {self.frame_count}",
              f"Bimanual recorded frames: {self.frame_count}"),
        ])
        for side in ("left", "right"):
            status = self.side_status[side]
            side_name = L("左手", "LEFT") if side == "left" else L("右手", "RIGHT")
            if not status.get("connected"):
                lines.append(f"{side_name}: {L('未连接', 'NOT CONNECTED')}")
                continue
            # Only the countdown is worth the space; a finished warm-up is the
            # normal state and needs no announcement.
            warmup_text = ""
            if status.get("warming_up"):
                warmup_text = (
                    L("  预热", "  WARM-UP")
                    + f" {status.get('warmup_remaining_s', 0.0):.1f}s")
            # Solve rate first, device arrival rate right behind it: when they
            # differ, the gap is the host dropping frames, not the glove
            # withholding them, and that is the whole point of showing both.
            device_fps = float(status.get("device_fps") or 0.0)
            lines.append(
                f"{side_name}: {status['fps']:5.1f} FPS  "
                + L(f"设备 {device_fps:5.1f} Hz  ", f"DEV {device_fps:5.1f} Hz  ")
                + f"IMU {16-len(status['missing'])}/16"
                + (L("  蓝牙", "  BT")
                   if status.get("link") == "bluetooth" else "")
                + warmup_text
                + (L("  安全", "  SAFETY") if status["safety"] else ""))
            broken = [int(channel) for channel in (status.get("broken") or [])]
            if broken:
                channels = "、".join(
                    f"{L('第', 'CH')}{channel}{L('路', '')}" for channel in broken)
                lines.append(f"{side_name}: {channels} {L('坏掉了', 'BROKEN')}")
            # A batched link explains an FPS that sits under the device rate, so
            # it earns a row only when that gap is actually there -- a
            # Bluetooth link is batched all the time, and saying so while it
            # keeps up would be noise.  A wired CDC link delivers one frame per
            # poll and stays silent here either way.
            health = status.get("link_health") or {}
            burst_mean = float(health.get("burst_mean") or 0.0)
            solve_fps = float(status.get("fps") or 0.0)
            if burst_mean >= 1.5 and device_fps - solve_fps > 0.15 * device_fps:
                lines.append(
                    f"{side_name}: "
                    + L(f"链路攒批 {burst_mean:.1f} 帧/批  "
                        f"最大间隔 {health.get('gap_max_s', 0.0):.2f}s",
                        f"Batched {burst_mean:.1f} frames/poll  "
                        f"max gap {health.get('gap_max_s', 0.0):.2f}s"))
            # The link-loss readout is deliberately not shown: ``dropped`` is
            # still counted and is still what the dual-dongle interference
            # warning below is judged from, it is just no longer a HUD row.
            recoveries = int(health.get("recoveries") or 0)
            if recoveries:
                lines.append(
                    f"{side_name}: "
                    + L(f"蓝牙链路已自动恢复 {recoveries} 次",
                        f"Bluetooth link auto-recovered {recoveries} time(s)"))
            # The stale-pose readout is deliberately not shown either:
            # ``status["stale"]`` still marks the frame, it is just no longer a
            # HUD row.
            # The magnetometer calibration readout is deliberately not shown:
            # the per-IMU levels and the "x/16 calibrated" count stay in the
            # monitor and remain what the prompt card is judged from, they are
            # just no longer a HUD row.
        bluetooth = [
            self.side_status[side] for side in ("left", "right")
            if self.side_status[side].get("connected")
            and self.side_status[side].get("link") == "bluetooth"
        ]
        if len(bluetooth) == 2:
            device_rates = [float(item.get("device_fps") or 0.0)
                            for item in bluetooth]
            loss_rates = []
            for item in bluetooth:
                health = item.get("link_health") or {}
                dropped = int(health.get("dropped") or 0)
                sent = int(health.get("frames") or 0) + dropped
                loss_rates.append(dropped / sent if sent else 0.0)
            # Verified signature on the real pair: either hand alone is 60 Hz
            # with zero loss, while together one stays near 60 and the other
            # accumulates CRC/sequence loss.  Say what that means rather than
            # presenting it as an unexplained low solver rate.
            if (min(device_rates, default=60.0) < 40.0
                    and max(loss_rates, default=0.0) >= 0.10):
                lines.append(L(
                    "检测到双蓝牙接收器互扰：请将两个接收器拉开，"
                    "或插到不同 USB 控制器",
                    "Dual Bluetooth receiver interference detected: separate "
                    "the dongles or use different USB controllers"))
        if self._dongle_reboot_status:
            lines.append(
                L("Dongle: ", "Dongle: ") + self._dongle_reboot_status)
        return lines

    def _dongle_reboot_button_label(self):
        return L("重启 dongle", "Reboot Dongle")

    def _on_reboot_dongle(self):
        """Send ``AT+REBOOT`` to the Bluetooth dongle from a worker thread.

        Runs off the GUI thread so the 3-D view keeps updating while the
        dongle replies and drops its CDC link; the outcome is echoed back to
        the HUD via ``self._dongle_reboot_status``.
        """

        self._dongle_reboot_status = L("正在发送 AT+REBOOT…", "sending AT+REBOOT…")
        self._redraw_requested = True

        def _worker():
            try:
                reply = reboot_bluetooth_dongle()
            except Exception as exc:  # surfaced to the user, not fatal
                self._dongle_reboot_status = (
                    L("重启失败: ", "reboot failed: ") + str(exc))
                print(f"[Dongle] AT+REBOOT failed: {exc}", file=sys.stderr)
            else:
                reply_text = reply or L("(无回复)", "(no reply)")
                self._dongle_reboot_status = (
                    L("重启成功 — 回复: ", "reboot ok — reply: ") + reply_text)
                print(f"[Dongle] AT+REBOOT reply: {reply!r}")
            self._redraw_requested = True

        threading.Thread(target=_worker, daemon=True, name="dongle-reboot").start()

    def update_both(self, left, right, frame_count, statuses,
                    mesh_slots: list[np.ndarray | None] | None = None):
        slots = np.stack([
            np.asarray(left, np.float32).reshape(21, 3),
            np.asarray(right, np.float32).reshape(21, 3),
        ])
        self._kpts_slots = slots
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
        # AT+REBOOT only means something to a Bluetooth dongle, so its button
        # appears with a dongle link and stays out of the bar on a wired glove
        # (where there is nothing to send the command to).  The button's
        # behaviour is unchanged -- this is visibility only.
        self._qcanvas.set_dongle_reboot_visible(
            any(statuses.get(side, {}).get("link") == "bluetooth"
                for side in ("left", "right")))
        # Extend the window title with the per-side firmware version once it
        # is known, e.g. "Stouch Glove V1.0 (SDK v0.3.0)  hardware: L V1 / R V1".
        fw_versions = {
            side: statuses.get(side, {}).get("firmware_version")
            for side in ("left", "right")
        }
        fw_parts = [
            f"{side[0].upper()} V{display_firmware_version(version)}"
            for side, version in fw_versions.items()
            if version is not None
        ]
        title = self._window_title_shown
        if fw_parts:
            base = title.split("  hardware: ")[0]
            title = f"{base}  hardware: {' / '.join(fw_parts)}"
        if title != self._window_title_shown:
            self._window_title_shown = title
            self._qcanvas.setWindowTitle(title)
        # Terminal warning when both hands are connected but their firmware
        # versions differ; printed once per (left, right) real-version pair.
        # The hand-side digit in the patch (``1.2.101`` vs ``1.2.102``) is not a
        # version mismatch, so compare the real version with that digit removed.
        left_fw, right_fw = fw_versions["left"], fw_versions["right"]
        left_real = firmware_real_version(left_fw) if left_fw is not None else None
        right_real = firmware_real_version(right_fw) if right_fw is not None else None
        if (left_real is not None and right_real is not None
                and left_real != right_real
                and (left_real, right_real) != self._fw_mismatch_reported):
            print(
                f"[WARN] 左右手固件版本不一致: "
                f"左手 V{display_firmware_version(left_fw)} / "
                f"右手 V{display_firmware_version(right_fw)}")
            self._fw_mismatch_reported = (left_real, right_real)
        elif left_real is not None and right_real is not None and left_real == right_real:
            self._fw_mismatch_reported = None
        self._bimanual_slots_cache = None
        self._bimanual_mesh_slots_cache = None
        # Actual GUI display rate = update_both arrivals over the last 1 s
        # (sliding count), independent of the solver's per-hand FPS.
        now = time.perf_counter()
        self._advance_startup_alignment(now)
        self._display_times.append(now)
        self._display_times = [
            t for t in self._display_times if now - t <= 1.0]
        if len(self._display_times) >= 2:
            window_s = self._display_times[-1] - self._display_times[0]
            self._display_fps = (
                (len(self._display_times) - 1) / window_s
                if window_s > 0.0 else 0.0)
        else:
            self._display_fps = 0.0
        self._redraw_requested = True

    def _startup_alignment_ready(self) -> bool:
        """Start only after every connected hand has left IMU warm-up."""
        if not self.present_sides:
            return False
        for side in self.present_sides:
            slot = 0 if side == "left" else 1
            status = self.side_status.get(side, {})
            if (status.get("warming_up")
                    or not status.get("warmup_completed")):
                return False
            if not np.isfinite(self._kpts_slots[slot]).all():
                return False
        return True

    def _publish_mag_judge_gate(self) -> None:
        """Tell the solver processes whether the magnetometer may be judged.

        Held off until IMU warm-up *and* the startup direction capture are both
        done.  Both steps have the glove deliberately still, so its MAG levels
        are low for reasons that have nothing to do with the ambient field, and
        the monitor's 30 s window would be spent before the user was ever asked
        to draw anything.

        Both facts live here, not in the solver process: warm-up arrives in
        ``statuses`` from the solve loop, and the capture is this object's own.
        The solver process only gets the verdict, because the monitor that owns
        the countdown is down there next to the 0x05 cache.

        Warm-up is latched once seen.  The solver stops reporting it after the
        fact, and a gate that re-closed the moment the key went missing would
        take the magnetometer prompt down with it.

        Written only when the value changes, so this is not a per-frame syscall.
        """

        event = self._mag_judge_event
        if event is None:
            return
        warmed = bool(self.present_sides)
        for side in self.present_sides:
            status = self.side_status.get(side, {})
            if status.get("warming_up") or not status.get("warmup_completed"):
                warmed = False
        if warmed:
            self._mag_warmup_seen = True
        ready = bool(self._startup_alignment_completed) and self._mag_warmup_seen
        if ready == self._mag_judge_published:
            return
        self._mag_judge_published = ready
        if ready:
            event.set()
        else:
            event.clear()

    def _orientation_recollect_label(self) -> str:
        return L("重新采集手方向", "Recapture hand direction")

    def _on_recollect_orientation(self) -> None:
        """Restart the three-second direction capture without touching recording."""

        self._startup_alignment_started_s = None
        self._startup_alignment_remaining_s = self.STARTUP_ALIGNMENT_DURATION_S
        self._startup_alignment_completed = False
        # Drop the previous correction back to identity so the hand model
        # returns to its just-started (uncorrected) pose while the new
        # direction is captured.  Samples are then taken from the raw pose,
        # exactly like the initial startup capture, and the correction is
        # recomputed fresh rather than composed on top of the old one.
        self._startup_alignment_corrections = [R.identity(), R.identity()]
        for samples in self._startup_alignment_samples.values():
            samples.clear()
        self._bimanual_slots_cache = None
        self._bimanual_mesh_slots_cache = None
        if self._startup_alignment_ready():
            self._startup_alignment_started_s = time.perf_counter()
        self._redraw_requested = True

    def _advance_startup_alignment(self, now_s: float) -> None:
        """Run the post-warm-up three-second orientation capture."""
        # Published on every pass, including the early returns below: an
        # in-progress capture is exactly a state the magnetometer gate has to
        # stay held off in.
        self._publish_mag_judge_gate()
        if self._startup_alignment_completed:
            return
        if self._startup_alignment_started_s is None:
            if not self._startup_alignment_ready():
                return
            self._startup_alignment_started_s = float(now_s)

        elapsed = max(0.0, float(now_s) - self._startup_alignment_started_s)
        self._startup_alignment_remaining_s = max(
            0.0, self.STARTUP_ALIGNMENT_DURATION_S - elapsed)

        # Cache is currently clear, so this resolves the normal calibrated
        # display pose before any startup correction has been committed.
        slots = self._display_slots()
        for side in self.present_sides:
            slot = 0 if side == "left" else 1
            if np.isfinite(slots[slot]).all():
                try:
                    basis = hand_display_basis(slots[slot])
                    if np.isfinite(basis).all():
                        self._startup_alignment_samples[side].append(basis)
                except ValueError:
                    pass

        if self._startup_alignment_remaining_s > 0.0:
            return
        if any(not self._startup_alignment_samples[side]
               for side in self.present_sides):
            return

        for side in self.present_sides:
            slot = 0 if side == "left" else 1
            matrices = np.stack(
                list(self._startup_alignment_samples[side]), axis=0)
            captured_basis = R.from_matrix(matrices).mean().as_matrix()
            target_basis = np.asarray(
                self._palm_target_bases[slot], dtype=float).reshape(3, 3)
            # Maps the direction held at the end of the guide back onto the
            # calibrated straight-ahead direction.  Subsequent motion is thus
            # expressed relative to the user's startup pose.
            correction = target_basis @ captured_basis.T
            self._startup_alignment_corrections[slot] = R.from_matrix(
                correction)

        self._startup_alignment_completed = True
        self._bimanual_slots_cache = None
        self._bimanual_mesh_slots_cache = None
        self._relative_reference_slots = self._display_slots().copy()
        print("[Orientation] Startup hand-direction alignment captured.")

    def _display_slots(self):
        # Cached per frame: the draw path resolves it several times, and each
        # call pays scipy Rotation + anchor math.
        if self._bimanual_slots_cache is None:
            slots = super()._display_slots()
            # Both hands get the per-frame root-rotation remap, each with its
            # own sign set.  This is a handedness/sign correction of the root
            # rotation in the shared display frame (previously only the left
            # hand was remapped); the right defaults to the no-op sign set
            # because it is the reference side.  It never reduces how far a
            # palm axis deviates from its target basis -- root drift removal
            # is stabilize_bimanual_palm_bases, used by replay only.
            for slot in (0, 1):
                if np.isfinite(slots[slot]).all():
                    signs = (self._left_root_rotvec_sign if slot == 0
                             else self._right_root_rotvec_sign)
                    slots[slot] = remap_hand_root_motion(
                        slots[slot], self._palm_target_bases[slot], signs)
            self._bimanual_slots_cache = place_hands_at_wrist_anchors(
                slots, separation_m=self.separation_m,
                anchor_axis=self._anchor_axis)
            for slot, correction in enumerate(
                    self._startup_alignment_corrections):
                hand = self._bimanual_slots_cache[slot]
                if np.isfinite(hand).all():
                    wrist = hand[0].copy()
                    self._bimanual_slots_cache[slot] = (
                        correction.apply(hand - wrist) + wrist
                    ).astype(np.float32)
        return self._bimanual_slots_cache

    def _display_mesh_slots(self) -> list[np.ndarray | None]:
        if self._bimanual_mesh_slots_cache is None:
            source_slots = super()._display_slots()
            source_mesh_slots = super()._display_mesh_slots()
            final_slots = self._display_slots()
            result: list[np.ndarray | None] = [None, None]
            for slot, source_mesh in enumerate(source_mesh_slots):
                if source_mesh is None:
                    continue
                if (not np.isfinite(source_slots[slot]).all()
                        or not np.isfinite(final_slots[slot]).all()):
                    continue
                source_basis = hand_display_basis(source_slots[slot])
                final_basis = hand_display_basis(final_slots[slot])
                rotation = final_basis @ source_basis.T
                source_wrist = source_slots[slot, 0]
                final_wrist = final_slots[slot, 0]
                # Continue from the base viewer's display-space mesh.  It has
                # already received the same fixed startup rotation as
                # source_slots; using the raw solver vertices here leaves that
                # rotation out and makes the full hand drift away from bones.
                mesh = np.asarray(
                    source_mesh, dtype=np.float32).reshape(778, 3)
                result[slot] = (
                    (mesh - source_wrist) @ rotation.T + final_wrist
                ).astype(np.float32)
            self._bimanual_mesh_slots_cache = result
        return self._bimanual_mesh_slots_cache

    def _grid_basis(self):
        return self._bimanual_grid_basis

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
        self._draw_startup_alignment_guide(img)
        self._draw_magnetometer_guide(img)
        return img

    def _draw_magnetometer_guide(self, img: np.ndarray) -> None:
        """Draw the magnetometer prompt card, if there is one to draw.

        The wording lives in :func:`_magnetometer_prompt_card` so it can be
        asserted without a canvas.
        """

        card = _magnetometer_prompt_card(self.side_status)
        if card is None:
            return
        title, subtitle = card
        title_scale = 0.92
        subtitle_scale = 0.58
        title_w, title_h = text_size(title, title_scale, 2)
        subtitle_w, _ = text_size(subtitle, subtitle_scale, 1)
        # The count wording is longer than the old prompt, and English runs much
        # wider than Chinese, so size the card to its contents instead of
        # clipping at the historical 620 px -- and if a line still cannot fit,
        # shrink that line: the card is centred, so an over-wide one would start
        # at a negative x and spill off the left edge.
        max_w = self.canvas_w - 32
        if title_w + 48 > max_w:
            title_scale *= (max_w - 48) / title_w
            title_w, title_h = text_size(title, title_scale, 2)
        if subtitle_w + 48 > max_w:
            subtitle_scale *= (max_w - 48) / subtitle_w
            subtitle_w, _ = text_size(subtitle, subtitle_scale, 1)
        card_w = max(620, title_w + 48, subtitle_w + 48)
        card_h = 112
        x0 = (self.canvas_w - card_w) // 2
        y0 = 168 if not self._startup_alignment_completed else 28
        x1, y1 = x0 + card_w, y0 + card_h
        roi = img[y0:y1, x0:x1]
        panel = np.full_like(roi, (35, 31, 48))
        cv2.addWeighted(panel, 0.94, roi, 0.06, 0.0, dst=roi)
        accent = (70, 170, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), accent, 3, cv2.LINE_AA)
        put_text(img, title,
                 (x0 + (card_w - title_w) // 2, y0 + 42 + title_h // 2),
                 title_scale, (250, 244, 232), 2)
        put_text(img, subtitle,
                 (x0 + (card_w - subtitle_w) // 2, y0 + 83),
                 subtitle_scale, (230, 218, 202), 1)

    def _draw_startup_alignment_guide(self, img: np.ndarray) -> None:
        """Draw the pre-capture instruction prominently at the upper centre."""
        if self._startup_alignment_completed:
            return

        card_w, card_h = 640, 126
        x0 = (self.canvas_w - card_w) // 2
        y0 = 28
        x1, y1 = x0 + card_w, y0 + card_h
        roi = img[y0:y1, x0:x1]
        panel = np.empty_like(roi)
        panel[:] = (28, 36, 54)
        cv2.addWeighted(panel, 0.92, roi, 0.08, 0.0, dst=roi)

        active = self._startup_alignment_started_s is not None
        accent = (185, 125, 245) if active else (225, 165, 75)
        cv2.rectangle(img, (x0, y0), (x1, y1), accent, 3, cv2.LINE_AA)

        if active:
            title = L("请将手摆正并保持", "STRAIGHTEN AND HOLD YOUR HANDS")
            subtitle = L(
                f"将在 {self._startup_alignment_remaining_s:.1f} 秒后获取当前方向",
                f"Capturing current orientation in {self._startup_alignment_remaining_s:.1f} seconds")
        else:
            title = L("IMU 预热中", "IMU WARM-UP")
            subtitle = L(
                "预热完成后，请在 3 秒内将手摆正",
                "When ready, straighten your hands during the 3-second countdown")

        title_scale = 1.0
        title_w, title_h = text_size(title, title_scale, 2)
        put_text(
            img, title,
            (x0 + (card_w - title_w) // 2, y0 + 48 + title_h // 2),
            title_scale, (248, 244, 238), 2)
        subtitle_scale = 0.62
        subtitle_w, _ = text_size(subtitle, subtitle_scale, 1)
        put_text(
            img, subtitle,
            (x0 + (card_w - subtitle_w) // 2, y0 + 96),
            subtitle_scale, (220, 211, 198), 1)

    def _record_button_rect(self) -> tuple[int, int, int, int]:
        # Right-anchored, so it tracks the canvas width.  Draw and hit-test
        # both go through here, which is what keeps them in agreement.
        return self.canvas_w - 305, 18, self.canvas_w - 22, 66

    def _draw_record_button(self, img: np.ndarray) -> None:
        x0, y0, x1, y1 = self._record_button_rect()
        if not self.recording_enabled:
            color, label = (75, 75, 75), L("录制已禁用", "RECORDING DISABLED")
        elif self.recording_state == "recording":
            color, label = (45, 55, 220), L("停止并保存", "STOP & SAVE")
        elif self.recording_state == "saved":
            color, label = (55, 145, 75), L("开始新录制", "START NEW RECORDING")
        else:
            color, label = (55, 155, 70), L("开始录制", "START RECORDING")
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (150, 165, 190), 1)
        size = text_size(label, 0.62, 2)
        origin = (
            x0 + max(8, (x1 - x0 - size[0]) // 2),
            y0 + (y1 - y0 + size[1]) // 2,
        )
        put_text(img, label, origin, 0.62, (245, 245, 245), 2)
        if self.recording_enabled and self.recording_state != "saved":
            put_text(
                img, L("空格 = 开始 / 停止并保存", "SPACE = start / stop & save"),
                (x0 + 18, y1 + 20), 0.42, (200, 195, 185), 1)

    def _request_record_toggle(self) -> None:
        if not self.recording_enabled:
            return
        self._record_action_requested = (
            "stop" if self.recording_state == "recording" else "start")

    def _resizable_tactile_frames(self):
        """The frames whose card is on screen with a resize handle.

        The two matrix panels are the resizable ones and each is drawn from a
        different source, so the answer depends on the selected view.
        """
        if self.tactile_panel_mode == "matrix":
            return self.tactile_frames
        if self.tactile_panel_mode == "linear_matrix":
            return self.tactile_raw_frames
        return {}

    def _tactile_grip_at(self, x, y):
        """The resize handle under the cursor, ready to drive a drag.

        The cards are composite images inside the frame, not widgets, so their
        geometry belongs to the overlay renderer; asking it here keeps one
        source of truth for where a card is and therefore for where its
        handle is.
        """
        frames = self._resizable_tactile_frames()
        if not frames:
            return None
        # Must be the canvas the overlay was actually drawn into -- the overlay
        # is rendered from the real buffer's shape -- or the handles would be
        # computed for one size and drawn at another, and stop responding.
        shape = (self.canvas_h, self.canvas_w)
        handles = tactile_grip_rects(shape, frames, self.tactile_scale)
        cards = tactile_card_rects(shape, frames, self.tactile_scale)
        start_cell = tactile_cell_for_scale(self.tactile_scale, shape)
        for side, (x0, y0, x1, y1) in handles.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                card = cards[side]
                # The drag is measured from the card's outer edge and its
                # bottom -- the two sides that stay put while it grows.
                anchor_x = float(card[0] if side == "left" else card[2])
                anchor_y = float(card[3])
                return {
                    "anchor_x": anchor_x,
                    "anchor_y": anchor_y,
                    "span_x": max(abs(float(x) - anchor_x), 1.0),
                    "span_y": max(anchor_y - float(y), 1.0),
                    "start_cell": start_cell,
                }
        return None

    def _drag_tactile_resize(self, x, y) -> None:
        """Resize the cards so the handle keeps following the cursor.

        Both directions count: the handle travels sideways as the card widens
        and upwards as it grows taller, and whichever way the cursor was
        pulled furthest sets the size.  Sliding only sideways therefore
        resizes, and so does pulling only upwards.

        The size is chosen in cell steps rather than by scaling the current
        scale: on a 1280x720 canvas a card runs out of frame after about a
        quarter more growth, so a scale-driven drag would stall while the
        cursor was still travelling.  The scale is derived back out of the
        cell, because it is the only size the painter is handed.
        """
        drag = self._tactile_resize
        ratio = max(
            abs(float(x) - drag["anchor_x"]) / drag["span_x"],
            max(0.0, drag["anchor_y"] - float(y)) / drag["span_y"],
        )
        cell = int(round(drag["start_cell"] * ratio))
        scale = float(np.clip(
            tactile_scale_for_cell(cell, (self.canvas_h, self.canvas_w)),
            TACTILE_SCALE_MIN, TACTILE_SCALE_MAX))
        if scale == self.tactile_scale:
            return
        self.tactile_scale = scale
        self._redraw_requested = True

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x0, y0, x1, y1 = self._record_button_rect()
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._request_record_toggle()
                return
            grip = self._tactile_grip_at(x, y)
            if grip is not None:
                # Claim the press: a resize must not also start a camera pan.
                self._tactile_resize = grip
                return
        elif event == cv2.EVENT_MOUSEMOVE and self._tactile_resize is not None:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self._drag_tactile_resize(x, y)
                return
            # No button down: the release landed outside the canvas.
            self._tactile_resize = None
        elif event == cv2.EVENT_LBUTTONUP and self._tactile_resize is not None:
            self._tactile_resize = None
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

    def consume_selector_request(self) -> bool:
        requested = self.request_selector
        self.request_selector = False
        return requested

    def set_recording_state(self, state: str, frame_count: int = 0) -> None:
        if state not in {"disabled", "idle", "recording", "saved"}:
            raise ValueError(f"invalid recording state: {state}")
        self.recording_state = state
        self.frame_count = int(frame_count)
        self._redraw_requested = True

    def set_tactile_side(self, side, processed, raw=None):
        # The overlay is driven by the raw ADC map so the baseline toggle can
        # switch between raw and baseline-corrected display (glove-test style);
        # the force curves use ``processed`` so their zero level and denoising
        # remain stable regardless of the pressure-map display mode.
        raw16 = (None if raw is None else
                 np.asarray(raw, np.float32).reshape(16, 16))
        self.tactile_raw_frames[side] = raw16
        self.tactile_calibrating_sides[side] = raw16 is None
        if self._baseline_collecting and raw16 is not None:
            self._baseline_sample(side, raw16)
        self.tactile_frames[side] = self._tactile_display_value(side)
        self._append_force_sample(side, processed)
        # No _redraw_requested here: the tactile stream bumps at ~70 Hz and
        # forcing a redraw per bump made the draw rate track the tactile rate
        # instead of the display gate (visible as jitter while moving).  The
        # panel is picked up by the next scheduled gate frame instead.

    def _tactile_display_value(self, side):
        raw = self.tactile_raw_frames.get(side)
        if raw is None:
            return None
        if self.use_baseline and self.tactile_baseline.get(side) is not None:
            return np.maximum(0.0, raw - self.tactile_baseline[side])
        return raw

    def _append_force_sample(self, side: str, processed) -> None:
        """Append one six-region force sample from the processed matrix."""

        if processed is None or side not in self._force_history:
            return
        totals = _tactile_force_totals(
            processed, side, threshold=float(self.tactile_threshold))
        now = time.monotonic()
        history = self._force_history[side]
        history.append((now, totals))
        cutoff = now - _FORCE_HISTORY_SECONDS
        while history and history[0][0] < cutoff:
            history.popleft()

    def _force_curve_snapshot(self):
        """Return stable numpy copies for the force window's paint pass."""

        now = time.monotonic()
        cutoff = now - _FORCE_HISTORY_SECONDS
        result = {}
        for side, history in self._force_history.items():
            visible = [(stamp, values) for stamp, values in history
                       if stamp >= cutoff]
            if visible:
                result[side] = (
                    np.asarray([item[0] for item in visible], np.float64),
                    np.stack([item[1] for item in visible]).astype(
                        np.float32, copy=False),
                )
        return result

    def _clear_force_history(self) -> None:
        for history in self._force_history.values():
            history.clear()

    def _on_tactile_thr(self, value) -> None:
        previous = self.tactile_threshold
        super()._on_tactile_thr(value)
        # Historical samples cannot be re-thresholded because only their six
        # totals are retained.  Start a clean time series after a threshold
        # change instead of mixing two definitions in one curve.
        if self.tactile_threshold != previous:
            self._clear_force_history()

    def _finish_baseline_if_complete(self):
        if not self._baseline_collecting:
            return
        if all(self._baseline_done.get(side) for side in self.present_sides):
            self._baseline_collecting = False
            # Refresh the corrected map immediately for sides whose raw data is
            # already available, without waiting for their next tactile frame.
            for side in ("left", "right"):
                if self.tactile_raw_frames.get(side) is not None:
                    self.tactile_frames[side] = self._tactile_display_value(side)
        self._update_baseline_status()

    def _baseline_sample(self, side, raw16):
        """Advance the baseline acquisition by one raw frame.

        Phase 1 waits a fixed settle delay after the button is pressed so a
        still-settling hand cannot seed a bad empty-glove baseline.  Phase 2
        then collects ``_baseline_target`` frames for the median and runs to
        completion once it starts.
        """
        if self._baseline_waiting:
            if time.monotonic() < self._baseline_wait_until:
                self._update_baseline_status()
                return
            self._baseline_waiting = False

        if self._baseline_done.get(side, False):
            return
        self._baseline_buffer[side].append(raw16)
        if len(self._baseline_buffer[side]) >= self._baseline_target:
            stack = np.stack(
                self._baseline_buffer[side][-self._baseline_target:], axis=0)
            self.tactile_baseline[side] = np.median(
                stack, axis=0).astype(np.float32)
            self._baseline_done[side] = True
        self._finish_baseline_if_complete()

    def _start_baseline_collection(self):
        self._baseline_collecting = True
        self._baseline_waiting = True
        self._baseline_wait_until = time.monotonic() + _BASELINE_SETTLE_DELAY_S
        self._baseline_buffer = {"left": [], "right": []}
        self._baseline_done = {"left": False, "right": False}
        self._update_baseline_status()

    def _update_baseline_status(self):
        qcanvas = getattr(self, "_qcanvas", None)
        if qcanvas is not None:
            qcanvas.set_baseline_status(self._baseline_status_text())
            if self._baseline_waiting:
                qcanvas.show_baseline_overlay(
                    L(f"{_BASELINE_SETTLE_DELAY_S:.0f}s后开始基线采集，请保持无接触",
                      f"{_BASELINE_SETTLE_DELAY_S:.0f}s before baseline collection, please keep hands free"))
            else:
                qcanvas.hide_baseline_overlay()

    def _baseline_status_text(self) -> str:
        if self.use_baseline:
            if self._baseline_collecting:
                if self._baseline_waiting:
                    return L(
                        f"等待 {_BASELINE_SETTLE_DELAY_S:.0f} 秒后开始采集…",
                        f"Waiting {_BASELINE_SETTLE_DELAY_S:.0f} s before collecting…")
                counts = [len(self._baseline_buffer.get(side, []))
                          for side in self.present_sides]
                n = min(counts) if counts else 0
                return L(
                    f"正在采集基线… {n}/{self._baseline_target}",
                    f"Collecting baseline… {n}/{self._baseline_target}")
            return L(
                f"已使用基线（{self._baseline_target}帧 · 中位数）",
                f"Baseline active ({self._baseline_target} frames · median)")
        return L("未使用基线", "Baseline off")

    def _on_use_baseline_toggled(self, checked) -> None:
        self.use_baseline = bool(checked)
        qcanvas = getattr(self, "_qcanvas", None)
        if self.use_baseline:
            if qcanvas is not None:
                qcanvas.set_baseline_button_enabled(True)
            self._start_baseline_collection()
        else:
            self._baseline_collecting = False
            self._baseline_waiting = False
            self._baseline_buffer = {"left": [], "right": []}
            self._baseline_done = {"left": False, "right": False}
            if qcanvas is not None:
                qcanvas.set_baseline_button_enabled(False)
            for side in ("left", "right"):
                self.tactile_frames[side] = self._tactile_display_value(side)
            self._update_baseline_status()
        self._redraw_requested = True

    def _on_recollect_baseline(self) -> None:
        if self.use_baseline:
            self._start_baseline_collection()
            self._redraw_requested = True

    def _baseline_toggle_label(self) -> str:
        return L("使用基线", "Use baseline")

    def _baseline_recollect_label(self) -> str:
        return L("重新采集基线", "Re-collect baseline")

    def _tactile_panel_label(self) -> str:
        return L("触觉面板", "Tactile panel")

    def _tactile_panel_mode_options(self) -> list[tuple[str, str]]:
        # "linear_matrix" (分区线性拟合) is deliberately kept out of the menu:
        # the feature is whole -- the guard in ``_on_tactile_panel_mode``,
        # ``_resizable_tactile_frames`` and the draw branch all still answer to
        # it -- it is simply not offered.  Putting the one entry back below is
        # the whole of re-enabling it.
        return [
            ("off", L("关闭", "Off")),
            ("force", L("力变化趋势", "Force trends")),
            ("matrix", L("压力矩阵", "Pressure matrix")),
        ]

    def _on_tactile_panel_mode(self, mode: str) -> None:
        mode = str(mode)
        self.tactile_panel_mode = (
            mode
            if mode in {"off", "force", "matrix", "linear_matrix"}
            else "off"
        )
        # The pressure-baseline toggle acts on the tactile panel's own maps, so
        # it rides with the panel: with the panel off there is nothing on screen
        # to correct.  Only visibility changes; ``use_baseline`` is left alone.
        self._qcanvas.set_baseline_controls_visible(
            self.tactile_panel_mode != "off")
        # Any resize in flight was aimed at a card that is no longer there.
        self._tactile_resize = None
        self._redraw_requested = True

    def _pressure_display_mode_label(self) -> str:
        return L("压力显示", "Pressure display")

    def _pressure_display_mode_options(self) -> list[tuple[str, str]]:
        return [
            ("none", L("无", "None")),
            ("points", L("压力点", "Pressure points")),
            ("tiles", L("方片", "Tiles")),
        ]

    def _on_pressure_display_mode(self, mode: str) -> None:
        mode = str(mode)
        self.pressure_display_mode = (
            mode if mode in {"none", "points", "tiles"} else "none")
        self._redraw_requested = True

    def _draw_tactile(self, img):
        if self.display_mode == "hand" and self._pressure_mappers:
            camera = self._camera()
            mesh_slots = self._controlled_display_mesh_slots()
            baseline_ready = {
                side: bool(self.use_baseline
                           and self.tactile_baseline.get(side) is not None)
                for side in ("left", "right")
            }
            if self.pressure_display_mode == "tiles":
                draw_pressure_tiles(
                    img,
                    camera,
                    mesh_slots,
                    self.tactile_frames,
                    self._pressure_mappers,
                    baseline_ready,
                    threshold=float(self.tactile_threshold),
                )
            elif self.pressure_display_mode == "points":
                draw_pressure_color_points(
                    img,
                    camera,
                    mesh_slots,
                    self.tactile_frames,
                    self._pressure_mappers,
                    baseline_ready,
                    threshold=float(self.tactile_threshold),
                )
            # "none" -> draw nothing on the mesh surface
        if self.tactile_panel_mode == "matrix":
            draw_bimanual_tactile_overlay(
                img,
                self.tactile_frames,
                threshold=self.tactile_threshold,
                scale=self.tactile_scale,
                baseline_corrected=self.use_baseline,
            )
        elif self.tactile_panel_mode == "linear_matrix":
            # The fit always receives the raw 12-bit ADC matrix.  Its fixed
            # 2048 -> 0 response conversion lives in pressure_linear_fit and
            # is independent of this viewer's optional baseline toggle.
            draw_bimanual_tactile_overlay(
                img,
                self.tactile_raw_frames,
                threshold=self.tactile_threshold,
                scale=self.tactile_scale,
                baseline_corrected=False,
                linear_fitted=True,
            )
        elif self.tactile_panel_mode == "force":
            _draw_force_trend_overlay(
                img, self._force_curve_snapshot(), self.present_sides,
                self.tactile_scale)


def _synthesize_single_hand_bases(initial: np.ndarray) -> np.ndarray:
    """Best-effort display bases when only one hand is present and no saved
    config exists: the connected hand keeps its own basis and the empty slot
    gets a thumb-side mirror, so it still lands at a bimanual wrist anchor."""
    slots = np.asarray(initial, dtype=np.float32).reshape(2, 21, 3)
    bases: list[np.ndarray | None] = [None, None]
    for slot in range(2):
        if np.isfinite(slots[slot]).all():
            bases[slot] = hand_display_basis(slots[slot])
    if bases[0] is None and bases[1] is not None:
        mirrored = bases[1].copy()
        mirrored[:, 0] *= -1.0
        mirrored[:, 2] = np.cross(mirrored[:, 0], mirrored[:, 1])
        mirrored[:, 2] /= max(float(np.linalg.norm(mirrored[:, 2])), 1e-9)
        bases[0] = mirrored
    elif bases[1] is None and bases[0] is not None:
        mirrored = bases[0].copy()
        mirrored[:, 0] *= -1.0
        mirrored[:, 2] = np.cross(mirrored[:, 0], mirrored[:, 1])
        mirrored[:, 2] /= max(float(np.linalg.norm(mirrored[:, 2])), 1e-9)
        bases[1] = mirrored
    return np.stack(bases)


class _LinkHealth:
    """Per-second view of how the link delivers frames, not just how many.

    ``device_fps`` already answers "how many frames arrived"; this answers "in
    what shape", which is what tells a link that is slow apart from one that is
    merely bursty.  A Bluetooth dongle hands over one batch per RF event, so a
    ``burst_mean`` above 1 there is the number that shows the solver is being
    fed bursts; on a wired CDC link it stays at 1.
    """

    def __init__(self) -> None:
        self.bursts: list[tuple[float, int]] = []
        self.gaps: list[tuple[float, float]] = []
        self._last_arrival: float | None = None

    def observe(self, now: float, frames: int) -> None:
        gap = 0.0 if self._last_arrival is None else now - self._last_arrival
        self._last_arrival = now
        self.bursts.append((now, int(frames)))
        self.gaps.append((now, gap))
        self.bursts = [item for item in self.bursts if now - item[0] <= 1.0]
        self.gaps = [item for item in self.gaps if now - item[0] <= 1.0]

    def snapshot(self, now: float, counters: dict) -> dict:
        bursts = [frames for _, frames in self.bursts]
        silent_s = 0.0 if self._last_arrival is None else now - self._last_arrival
        # A link that has stopped delivers no gap event of its own, so the
        # current silence has to be folded in -- reporting 0.0 for a dead link
        # would read as healthy.
        gap_max = max((gap for _, gap in self.gaps), default=0.0)
        return {
            "burst_max": max(bursts, default=0),
            "burst_mean": (sum(bursts) / len(bursts)) if bursts else 0.0,
            "gap_max_s": max(gap_max, silent_s),
            **counters,
        }


def make_result(runtimes, projector, present_sides):
    def slot_of(side: str, attr: str) -> np.ndarray:
        value = getattr(runtimes[side], attr)
        if value is None:
            return np.full((21, 3), np.nan, np.float32)
        return np.asarray(value, np.float32).reshape(21, 3)

    hands3d = np.stack([slot_of("left", "latest_joints"),
                        slot_of("right", "latest_joints")]).astype(np.float32)
    smoothed = np.stack([slot_of("left", "latest_smoothed"),
                         slot_of("right", "latest_smoothed")]).astype(np.float32)
    hands2d = np.stack([projector.project(smoothed[index])[0] for index in range(2)])
    ready = [len(runtimes[side].missing) == 0 for side in ("left", "right")]
    presents = [side in present_sides for side in ("left", "right")]
    return {
        "hands2d": hands2d,
        "hands3d": hands3d,
        "smoothed": smoothed,
        "labels": ["Left", "Right"],
        "presents": presents,
        "propagated": [False, False],
        "stage2": ready,
        "reprojection_error": [float("nan"), float("nan")],
    }


def _snapshot(runtimes, present_sides, state_lock):
    """Read the published per-side state under ``state_lock``.

    The hand processes copy-on-publish every array, so the returned
    references are stable: state is only ever replaced with fresh copies,
    never mutated in place, so the main thread can use it without a further
    copy.
    """
    with state_lock:
        snap = {}
        for side in ("left", "right"):
            runtime = runtimes[side]
            snap[side] = {
                "joints": runtime.latest_joints,
                "mesh_vertices": getattr(runtime, "latest_mesh_vertices", None),
                "smoothed": runtime.latest_smoothed,
                "tactile": runtime.latest_tactile,
                "tactile_raw": runtime.latest_tactile_raw,
                "missing": list(runtime.missing),
                "broken": list(getattr(runtime, "broken", [])),
                "safety": bool(runtime.safety),
                "contact": runtime.contact,
                "warming_up": bool(getattr(runtime, "warming_up", False)),
                "warmup_remaining_s": float(
                    getattr(runtime, "warmup_remaining_s", 0.0)),
                "warmup_completed": bool(
                    getattr(runtime, "warmup_completed", False)),
                "warmup_timed_out": bool(
                    getattr(runtime, "warmup_timed_out", False)),
                "fps": float(getattr(runtime, "fps", 0.0)),
                "device_fps": float(getattr(runtime, "device_fps", 0.0)),
                "link_health": getattr(runtime, "latest_link_health", None),
                "host_s": float(runtime.latest_host_s),
                "tactile_version": int(
                    getattr(runtime, "tactile_version", 0)),
                "raw_present": (
                    None if getattr(runtime, "latest_raw_present", None) is None
                    else np.asarray(runtime.latest_raw_present, dtype=bool).reshape(16).copy()),
                "mag_levels": getattr(runtime, "latest_mag_levels", None),
                "mag_unread": getattr(runtime, "latest_mag_unread", None),
                "mag_prompt": getattr(runtime, "latest_mag_prompt", None),
                "firmware_version": getattr(
                    runtime, "firmware_version", None),
                "latency": getattr(runtime, "latency", None),
            }
    return snap


def _snap_joints_or_nan(snap, side) -> np.ndarray:
    value = snap[side]["joints"]
    if value is None:
        return np.full((21, 3), np.nan, np.float32)
    return np.asarray(value, np.float32).reshape(21, 3)


def _snap_smoothed_or_nan(snap, side) -> np.ndarray:
    value = snap[side].get("smoothed")
    if value is None:
        return _snap_joints_or_nan(snap, side)
    return np.asarray(value, np.float32).reshape(21, 3)


def _snap_mesh_or_none(snap, side) -> np.ndarray | None:
    value = snap[side].get("mesh_vertices")
    if value is None:
        return None
    mesh = np.asarray(value, np.float32)
    if mesh.shape != (778, 3) or not np.isfinite(mesh).all():
        return None
    return mesh


def _publish_latest(state_queue, message) -> None:
    """Publish without allowing stale IPC state to block a solver process."""

    try:
        state_queue.put_nowait(message)
        return
    except queue.Full:
        pass
    try:
        state_queue.get_nowait()
    except queue.Empty:
        # A multiprocessing.Queue feeder can briefly hold the item after its
        # capacity semaphore is acquired.  Dropping this update is preferable
        # to blocking the latency-sensitive solver; the next update follows.
        return
    try:
        state_queue.put_nowait(message)
    except queue.Full:
        pass


def hand_solver_process(side, port, usb_vid, usb_pid, calib_path,
                        geometry_path, solve_rate, state_queue, stop_event,
                        mag_judge_event=None):
    """One process per hand: stream + solve + publish to the parent.

    The parent is the consumer: it drains ``state_queue`` keep-newest and
    mirrors the fields into a :class:`PublishedHandState` under its own lock.
    Giving each hand its own process gives it its own GIL and its own core,
    so solve bursts no longer preempt the GUI thread.  Fatal errors are
    shipped as a ``{"type": "fatal"}`` message before the process exits.
    """
    try:
        runtime = HandRuntime(
            side=side,
            config=load_calibration(Path(calib_path), side),
            solver=HandSolver(side, Path(calib_path), Path(geometry_path)),
            stream=RawImuStream(
                str(port), usb_vid=int(usb_vid), usb_pid=int(usb_pid)),
            tactile_stream=None,
            smoother=JointPositionSmoother(0.015),
            tactile_pre=TactilePreprocessor(
                base_gate=0.0, dynamic_noise_ratio=0.0, temporal_smooth=0.15,
                spatial_filter=False, calibration_frames=100, bypass_gates=False),
            missing=[f"IMU-{index}" for index in range(16)],
        )
        runtime.tactile_stream = runtime.stream.tactile_stream()
        runtime.stream.start()
        runtime.tactile_stream.start()
    except BaseException as exc:
        _publish_latest(state_queue, {
            "type": "fatal",
            "error": f"{type(exc).__name__}: {exc}",
        })
        return
    print(f"[{side}] solver process started (pid {os.getpid()})")
    # IMU and tactile packets share the transport sequence counter, so using
    # ``raw_frame.sequence % 4`` does *not* keep three of four IMU samples; in
    # the usual alternating packet stream it drops about half of them.  Use
    # device time as a fractional, non-blocking rate budget instead.  A frame
    # is either solved immediately or discarded; there is no sleep, backlog,
    # or replay of stale poses.
    solve_rate = min(max(float(solve_rate), 1.0), 80.0)
    solve_credit = 0.0
    last_device_us: int | None = None
    solve_times: list[float] = []
    # Device arrival rate, tracked separately from the solve rate.  The two
    # diverge whenever a solve cannot keep up with the USB rate, and showing
    # only the solve rate makes a busy host look like a slow glove.
    device_frame_total = 0
    device_samples: list[tuple[float, int]] = []
    # Delivery shape (burst size, arrival gaps) and the transport's own parser
    # error counters, which nothing read before.  Together they decide whether
    # a low FPS is the link's fault or the host discarding frames.
    link_health = _LinkHealth()
    fault_detector = ImuFaultDetector()
    # One calibration monitor per hand: it both reads the 0x05 cache and owns
    # the countdown, so it lives here where the frames arrive rather than being
    # fed through the parent at the display rate.  The jump detector sits beside
    # it for the same reason -- it needs every raw attitude frame, and the
    # parent only sees the ones that get published.
    mag_monitor = MagCalibrationMonitor()
    mag_jump_detector = ImuJumpDetector()
    try:
        while not stop_event.is_set():
            saw_data = False
            tactile_frames = runtime.tactile_stream.poll()
            if tactile_frames:
                # Tactile is a live overlay, not a history playback.  Publishing
                # only the newest sample prevents stale pressure frames from
                # delaying the latency-sensitive IMU solve after a USB backlog.
                # Unlike the IMU frames below, keeping one here loses nothing a
                # consumer could have seen: ``latest_tactile`` is replaced by
                # each frame and the overlay redraws from it, so earlier frames
                # of the same poll would only be overwritten.
                tactile_frames = tactile_frames[-1:]
            for tactile_frame in tactile_frames:
                saw_data = True
                tactile_raw = tactile_frame.samples.copy()
                processed, _ = runtime.tactile_pre.process(
                    tactile_frame.samples)
                runtime.latest_tactile = (
                    processed.copy() if processed is not None else None)
                runtime.tactile_version += 1
                _publish_latest(state_queue, {
                    "type": "tactile",
                    "tactile": runtime.latest_tactile,
                    "tactile_raw": tactile_raw,
                    "tactile_version": runtime.tactile_version,
                })
            raw_frames = runtime.stream.poll()
            if raw_frames:
                # Count arrivals before the backlog is discarded below; past
                # this point the device rate is unobservable.
                arrival = time.perf_counter()
                device_frame_total += len(raw_frames)
                device_samples.append((arrival, device_frame_total))
                device_samples = [
                    sample for sample in device_samples
                    if arrival - sample[0] <= 1.0]
                # Observed before the burst is trimmed below: how the link
                # grouped the frames is the whole point, and it is only visible
                # on the raw poll result.
                link_health.observe(arrival, len(raw_frames))
                # Real-time: drop only the part of the batch that is history by
                # now, keep the rest.  Collapsing a whole batch to its last
                # frame (what this did before) made a bursty link's solved rate
                # equal its batch rate; the device-time credit accumulator below
                # is what throttles the rate, and it cannot throttle frames that
                # were discarded before it ever saw them.
                raw_frames = recent_batch(raw_frames)
            for raw_frame in raw_frames:
                if raw_frame.sequence == runtime.last_seq:
                    continue
                saw_data = True
                runtime.last_seq = raw_frame.sequence
                if runtime.stream.firmware_version is not None:
                    runtime.firmware_version = runtime.stream.firmware_version
                real_version = firmware_real_version(
                    runtime.firmware_version) if runtime.firmware_version else None
                uses_mag_telemetry = bool(
                    real_version is not None
                    and real_version >= USB_FULL_SEQUENCE_FROM)
                magnetic = (runtime.stream.mag_telemetry
                            if uses_mag_telemetry else None)
                # Everything the publish at the bottom of this loop reads must be
                # defined before the branch that fills it.  A key only the
                # telemetry path can produce became an UnboundLocalError rather
                # than a missing field when it was defined inside that branch,
                # which crashed the solver process on firmware without type 0x05.
                # ``py_compile`` does not see it and neither did any offline test.
                mag_levels = None
                mag_unread: tuple[int, ...] = ()
                mag_prompt = None
                # Fed every frame, whatever the firmware: the step is what
                # triggers the magnetometer check, and a step is only meaningful
                # once there is a previous frame to compare against.  How big
                # the step was is not reported anywhere -- not on screen and not
                # in the terminal -- only whether it crossed the bar.
                mag_triggered, _ = mag_jump_detector.update(
                    raw_frame.quaternions_xyzw, raw_frame.valid_mask)
                if uses_mag_telemetry:
                    # ``mag_levels``, not ``levels_raw``: the wire carries the
                    # whole CALIB_STAT byte, and MAG is its low two bits -- 0x03
                    # for a calibrated channel (MAG_TELEMETRY.md 1.4).  Comparing
                    # ``levels_raw`` against 3 would count that as 0, and reading
                    # the high bits instead returns SYS, which sits near 0 on a
                    # calibrated glove, so the gate would never open.  A
                    # never-read channel stays ``None`` in both tuples, so
                    # "unknown" is not level 0.
                    mag_levels = None if magnetic is None else magnetic.mag_levels
                    # Half of the channels *this 0x05 frame carried* must read 3.
                    # That frame's mask is the only denominator the telemetry
                    # licenses -- there is no usable "is this IMU working" signal
                    # -- and it is what keeps the glove from being charged for
                    # channels nobody read this round.
                    mag_counts = ((0, 0) if magnetic is None
                                  else magnetic.gate_counts)
                    # ``None`` is "no 0x05 at all" rather than "not calibrated",
                    # which the monitor must not treat as something to prompt for.
                    mag_gate = (None if magnetic is None
                                or not any(magnetic.level_seen)
                                else magnetic.ready)
                    mag_unread = (() if magnetic is None
                                  else magnetic.unread_channels)
                    # Judging waits for IMU warm-up and the startup direction
                    # capture: the fusion is settling through both, so a step in
                    # the attitude says nothing about the field and a trigger
                    # arriving then is discarded rather than acted on.  The
                    # parent owns both facts and publishes the combined verdict
                    # -- warm-up rides in the solve status and the capture is the
                    # viewer's, neither of which is here.
                    judge_ready = (mag_judge_event is None
                                   or mag_judge_event.is_set())
                    mag_prompt = None if magnetic is None else mag_monitor.update(
                        time.perf_counter(), mag_gate, mag_counts,
                        triggered=mag_triggered, judge_ready=judge_ready)
                # Read this frame's firmware health flag afresh; the previous
                # frame's verdict is never carried over.
                broken_channels = fault_detector.update(raw_frame.present_mask)
                device_us = int(raw_frame.device_timestamp_us)
                if last_device_us is None:
                    # Publish the first pose immediately.
                    last_device_us = device_us
                elif solve_rate < 80.0:
                    # The firmware timestamp is a wrapping uint32 counter.
                    elapsed_us = (device_us - last_device_us) & 0xFFFFFFFF
                    last_device_us = device_us
                    if elapsed_us > 250_000:
                        # Reconnect/timestamp reset: do not accumulate a burst
                        # of historical solve credit.
                        solve_credit = 0.0
                    else:
                        solve_credit = min(
                            solve_credit
                            + elapsed_us * solve_rate / 1_000_000.0,
                            2.0,
                        )
                        if solve_credit < 1.0:
                            continue
                        solve_credit -= 1.0
                else:
                    last_device_us = device_us
                keypoints = runtime.solver.process(raw_frame)
                sample_s = keypoints.timestamp_us / 1_000_000.0
                joints = np.asarray(keypoints.joints_m, np.float32).copy()
                smoothed = np.asarray(
                    runtime.smoother.update(keypoints.joints_m, sample_s),
                    np.float32).copy()
                imu = np.asarray(keypoints.imu_xyzw, np.float32).copy()
                missing = [
                    f"IMU-{index}" for index in np.flatnonzero(
                        ~np.isfinite(keypoints.sensor_age_s))]
                # perf_counter (QPC) is system-wide monotonic, so the host_s
                # published here is directly comparable across the process
                # boundary in the parent.
                now = time.perf_counter()
                solve_times.append(now)
                solve_times = [
                    t for t in solve_times if now - t <= 1.0]
                if len(solve_times) >= 2:
                    solve_window_s = solve_times[-1] - solve_times[0]
                    measured_fps = (
                        (len(solve_times) - 1) / solve_window_s
                        if solve_window_s > 0.0 else 0.0)
                else:
                    measured_fps = 0.0
                # Frames delivered over the same one-second window.  Counted
                # rather than timed because the transport hands over a burst
                # per poll, so a per-batch timestamp would read as a stall.
                if len(device_samples) >= 2:
                    device_window_s = (
                        device_samples[-1][0] - device_samples[0][0])
                    device_fps = (
                        (device_samples[-1][1] - device_samples[0][1])
                        / device_window_s
                        if device_window_s > 0.0 else 0.0)
                else:
                    device_fps = 0.0
                _publish_latest(state_queue, {
                    "type": "solve",
                    "joints": joints,
                    "mesh_vertices": _latest_solver_mesh_vertices(runtime.solver),
                    "smoothed": smoothed,
                    "imu": imu,
                    "raw_imu": raw_frame.quaternions_xyzw.copy(),
                    "raw_present": raw_frame.present_mask.copy(),
                    "raw_valid": raw_frame.valid_mask.copy(),
                    # Plain tuples of int/None only: this crosses a process
                    # queue, so no snapshot object may travel with it.
                    "mag_levels": mag_levels,
                    # Plain tuple of ints: the channels this round did
                    # not report, which is why the count is out of twelve.
                    # Empty on a glove with no 0x05 at all.
                    "mag_unread": mag_unread,
                    # None, or the monitor's plain {"kind", "remaining_s",
                    # "ready", "carried"} dict, which is what the card renders
                    # its counts from.
                    "mag_prompt": mag_prompt,
                    "raw_device_timestamp_us": raw_frame.device_timestamp_us,
                    "missing": missing,
                    "broken": broken_channels,
                    "safety": bool(keypoints.status.safety_intervened),
                    "contact": keypoints.status.active_contact,
                    "warming_up": bool(
                        keypoints.status.details.get("warming_up", False)),
                    "warmup_remaining_s": float(
                        keypoints.status.details.get("warmup_remaining_s")
                        or 0.0),
                    "warmup_completed": bool(
                        keypoints.status.details.get("warmup_completed", False)),
                    "warmup_timed_out": bool(
                        keypoints.status.details.get("warmup_timed_out", False)),
                    "fps": measured_fps,
                    "device_fps": device_fps,
                    "host_s": now,
                    "firmware_version": getattr(runtime, "firmware_version", None),
                    # Refreshed by the transport's own reader thread every ~2 s;
                    # ``None`` until the device answers, and permanently
                    # ``None`` on firmware older than v1.2.11.
                    "latency": getattr(runtime.stream, "latency", None),
                    # Plain dict of numbers: this crosses the process queue.
                    # ``link_errors`` counts bytes the parser rejected,
                    # ``link_drops`` frames the link never delivered; neither
                    # rises for frames this loop deliberately discards, so
                    # together they say where the frames went.
                    "link_health": link_health.snapshot(
                        now, {
                            **(getattr(runtime.stream, "link_errors", None)
                               or {}),
                            **(getattr(runtime.stream, "link_drops", None)
                               or {}),
                        }),
                })
            if not saw_data:
                stop_event.wait(0.001)
    except BaseException as exc:  # shipped to the parent, not swallowed
        _publish_latest(state_queue, {
            "type": "fatal",
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        runtime.stream.stop()


def _drain_hand_queue(state_queue, state) -> dict | None:
    """Apply pending child messages to the parent-side mirror state.

    Called by the main loop with ``state_lock`` held (the recorder thread
    reads the same fields under the same lock).  Messages apply keep-newest,
    so a slow GUI frame only ever sees the latest published values.  Returns
    the fatal-error message when the hand process crashed, else ``None``.
    """
    fatal = None
    latest_by_kind = {}
    # The queue itself is bounded to eight messages.  Also cap each drain so
    # producers cannot keep this loop alive indefinitely while the 60 Hz GUI
    # deadline passes.
    for _ in range(8):
        try:
            message = state_queue.get_nowait()
        except queue.Empty:
            break
        kind = message.get("type")
        if kind == "fatal":
            fatal = message
        elif kind in ("tactile", "solve"):
            latest_by_kind[kind] = message
    tactile_message = latest_by_kind.get("tactile")
    if tactile_message is not None:
        message = tactile_message
        state.latest_tactile = message["tactile"]
        state.latest_tactile_raw = message["tactile_raw"]
        state.tactile_version = message["tactile_version"]
    solve_message = latest_by_kind.get("solve")
    if solve_message is not None:
        message = solve_message
        state.latest_joints = message["joints"]
        state.latest_mesh_vertices = message.get("mesh_vertices")
        state.latest_smoothed = message["smoothed"]
        state.latest_imu = message["imu"]
        state.latest_raw_imu = message["raw_imu"]
        state.latest_raw_present = message["raw_present"]
        state.latest_raw_valid = message["raw_valid"]
        state.latest_mag_levels = message.get("mag_levels")
        state.latest_mag_unread = message.get("mag_unread")
        state.latest_mag_prompt = message.get("mag_prompt")
        state.latest_raw_device_timestamp_us = (
            message["raw_device_timestamp_us"])
        state.missing = message["missing"]
        state.broken = list(message.get("broken") or [])
        state.safety = message["safety"]
        state.contact = message["contact"]
        state.warming_up = message["warming_up"]
        state.warmup_remaining_s = message["warmup_remaining_s"]
        state.warmup_completed = message["warmup_completed"]
        state.warmup_timed_out = message["warmup_timed_out"]
        state.fps = message["fps"]
        state.device_fps = float(message.get("device_fps") or 0.0)
        state.latest_link_health = message.get("link_health")
        state.latest_host_s = message["host_s"]
        state.firmware_version = message["firmware_version"]
        state.latency = message.get("latency")
    return fatal


def recorder_loop(runtimes, present_sides, projector, args,
                  state_lock, rec_lock, stop_event, shared):
    """Sample ``latest_*`` onto the recording session at a fixed wall-clock
    cadence (``args.record_fps``, default 60 Hz), independent of the solve loop.

    Each hand's solver process can be busy for many milliseconds inside a poll
    burst, so cadence checks bolted onto the acquire path would lag the 60 Hz
    slots by up to one solve per burst.  This thread owns the sampling clock
    instead: it wakes every
    ``1/record_fps`` seconds and records the newest published state, so recorded
    timestamps stay spaced by exactly ``interval`` regardless of burstiness.

    Lock discipline: ``state_lock`` is held while reading the published state
    (via :func:`make_result`) and while :meth:`BimanualRecordingSession.add`
    reads the same fields; ``rec_lock`` guards session start/stop/add.  The
    nesting here is always ``rec_lock`` then ``state_lock``; no thread acquires
    them in the reverse order, so there is no lock ordering to violate.
    """
    record_fps = float(getattr(args, "record_fps", None) or args.fps)
    interval = 1.0 / max(record_fps, 1.0)
    next_record = 0.0
    prev_recording_state = None
    try:
        while not stop_event.is_set():
            # perf_counter (QPC): time.monotonic ticks at ~15.6 ms on Windows
            # here, which quantized the recording gate to ~32 Hz.
            now = time.perf_counter()
            with rec_lock:
                session = shared.session
            if session is not None and session.state == "recording":
                if prev_recording_state != "recording":
                    next_record = now  # anchor the first sample of this take
                prev_recording_state = "recording"
                if now >= next_record:
                    with state_lock:
                        ready = all(
                            runtimes[side].latest_joints is not None
                            for side in present_sides)
                        fresh = ready and all(
                            now - runtimes[side].latest_host_s
                            <= args.max_frame_age
                            for side in present_sides)
                        result = (
                            make_result(runtimes, projector, present_sides)
                            if fresh else None)
                    if result is not None:
                        with rec_lock:
                            session = shared.session
                            if (session is not None
                                    and session.state == "recording"):
                                with state_lock:
                                    session.add(now, result, runtimes)
                # Fixed-timestep accumulator.  Resetting from `now` added the
                # loop's wake-up overshoot to every interval and drifted the
                # rate ~8% slow; advancing the deadline keeps the long-term
                # rate exactly record_fps and skips any beat the slow loop
                # missed instead of bursting duplicate rows.
                while now >= next_record:
                    next_record += interval
            else:
                prev_recording_state = (
                    session.state if session is not None else None)
            stop_event.wait(
                max(interval - (time.perf_counter() - now), 0.001))
    except BaseException as exc:  # surfaced to the main loop, not swallowed
        shared.recorder_error = (type(exc), exc, exc.__traceback__)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", "--calib-dir", type=Path,
                        help="Folder shortcut for the newest left/right calibrations")
    parser.add_argument("--left-calib", type=Path,
                        help="Explicit left-hand IMU calibration JSON")
    parser.add_argument("--right-calib", type=Path,
                        help="Explicit right-hand IMU calibration JSON")
    parser.add_argument(
        "--select-calibration", action="store_true",
        help="Force the calibration selection window (re-pick even for a "
             "glove that already has a remembered calibration file)")
    parser.add_argument(
        "--no-select-calibration", action="store_true",
        help="Skip the selection window; use --left-calib/--right-calib, "
             "remembered bindings, or the auto-picked defaults")
    parser.add_argument(
        "--calibration-bindings", type=Path,
        default=DEFAULT_CALIBRATION_BINDINGS,
        help="JSON file remembering each glove's calibration file by USB serial")
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_PATH,
                        help="Hand geometry JSON")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--fps", type=float, default=60.0,
        help="Preview refresh rate (Hz), default 60; recording uses --record-fps")
    parser.add_argument(
        "--record-fps", type=float, default=60.0,
        help="Recording sample rate (Hz), default 60")
    parser.add_argument(
        "--solve-fps", type=float, default=0.0,
        help="Per-hand solve decimation target in Hz; 0 = auto (60). "
             "Each hand solves in its own process on its own "
             "core, so the GUI no longer competes for the GIL; lower this "
             "only on CPU-constrained machines.")
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--max-frame-age", type=float, default=0.25)
    parser.add_argument(
        "--link-diag", action="store_true",
        help="Print one link-diagnostics line per hand per second: how many "
             "frames each poll delivered, the longest arrival gap, and the "
             "frame parser's error counters.  Off by default so a normal run "
             "stays quiet; this is what tells a slow link apart from a bursty "
             "one when FPS sits below the device rate.")
    parser.add_argument("--display-config", type=Path,
                        default=DEFAULT_BIMANUAL_DISPLAY_CONFIG,
                        help="Persistent palm bases and wrist separation")
    parser.add_argument("--view-config", type=Path,
                        default=PROJECT_ROOT / "config" / "view_config_3d_bimanual.json")
    parser.add_argument("--view-reference-config", type=Path,
                        default=PROJECT_ROOT / "config" / "view_config_3d.json")
    parser.add_argument("--relearn-display-default", action="store_true",
                        help="Relearn and overwrite display defaults from the first frame")
    parser.add_argument("--separation", type=float, default=None,
                        help="Override wrist separation in metres for this run")
    parser.add_argument("--yaw", type=float, default=None)
    parser.add_argument("--elev", type=float, default=None)
    parser.add_argument("--roll", type=float, default=None)
    parser.add_argument("--dist", type=float, default=None)
    parser.add_argument("--display-mode", default="hand", choices=("bones", "hand"),
                        help="Initial 3D display mode: bones or full gray hand mesh")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh",
                        help="UI language: zh (default) or en")
    parser.add_argument("--checkpoint-frames", type=int, default=600)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--record-on-start", action="store_true",
        help="Start recording when data is ready; default waits for the GUI button")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)
    if args.no_save and args.record_on_start:
        parser.error("--no-save and --record-on-start cannot be combined")
    if args.no_display and not (args.no_save or args.record_on_start):
        parser.error("--no-display requires --record-on-start or --no-save")
    return args


def _run_live_session(args) -> int | str:
    """Run one live-viewer session.

    Returns an int exit code, ``"calibrate"`` when the in-app Calibrate
    button was pressed (the launcher then switches to the calibration screen),
    or ``"selector"`` when the in-app "back to calibration selector" button was
    pressed (the launcher re-opens the calibration-file picker).
    """
    # Resolve each bound hand independently: a missing glove is skipped, not
    # fatal.  Only when no glove at all is detected do we abort.
    device_manager = DeviceManager(args.registry)
    # A glove link that no registry entry claims is probed once for the hand
    # side its firmware reports.  This is what makes a freshly plugged Bluetooth
    # dongle usable on the first launch: the dongle hides the glove's STM32
    # serial, so there is no other way to tell which hand is behind it.
    try:
        unbound_links = detect_unbound_glove_links(args.registry)
    except Exception as exc:  # probing is best-effort, never fatal
        print(f"[WARN] unbound glove probe failed: {exc}", file=sys.stderr)
        unbound_links = {}
    present_sides: list[str] = []
    ports: dict[str, str] = {}
    serials: dict[str, str] = {}
    links: dict[str, str] = {}
    auto_detected: dict[str, str] = {}
    used_ports: set[str] = set()
    for side in ("left", "right"):
        try:
            port, serial = device_manager.resolve_port(side)
        except DeviceNotFoundError:
            discovered = unbound_links.get(side)
            if discovered is None:
                print(f"{side} device: NOT CONNECTED (skipped)")
                continue
            port, serial = discovered
            auto_detected[side] = serial
            print(
                f"{side} device: firmware reports this hand; unbound "
                f"link={link_kind_for_device(port) or 'usb'}  port={port}")
        # A bimanual runtime must never open one physical CDC port twice.  This
        # is a defensive guard for stale registries, unusual USB drivers, or a
        # future resolver fallback: the second logical side is absent rather
        # than being fed the first hand's frames.
        if port in used_ports:
            print(
                f"{side} device: DUPLICATE PORT {port} (skipped; "
                "already assigned to another hand)")
            continue
        used_ports.add(port)
        present_sides.append(side)
        ports[side] = port
        serials[side] = serial
        link = link_kind_for_device(port) or "usb"
        links[side] = link
        # A "bluetooth" link means the serial below is the dongle's, not the
        # glove's: the glove behind it is not a USB device, so the dongle is
        # what the registry and the calibration bindings key on.
        print(
            f"{side} device: link={link}  serial={serial}  port={port}")
    if not present_sides:
        print("No STM32 glove connected; opening the calibration selector.", file=sys.stderr)
    else:
        print(f"Connected gloves: {', '.join(present_sides)}")
    # Persist what the probe just worked out, so the next launch resolves the
    # link by serial and skips the (up to two-second) port probe entirely.
    for side, serial in auto_detected.items():
        try:
            set_glove_serial(side, serial, args.registry)
        except Exception as exc:  # a read-only registry must not stop the run
            print(f"[WARN] could not remember {side} link: {exc}", file=sys.stderr)
        else:
            print(f"{side} device: remembered {side} -> serial {serial}")

    # Apply remembered calibration bindings first (serial -> calibration file),
    # so a previously-bound glove reuses its file on the next launch without a
    # prompt.  Then prompt only for sides that still lack a calibration (or
    # every connected side when --select-calibration forces a re-pick).
    bindings = load_calibration_bindings(args.calibration_bindings)
    for side in present_sides:
        attr = "left_calib" if side == "left" else "right_calib"
        if getattr(args, attr) is not None:
            continue
        remembered = remembered_calibration(bindings, serials.get(side))
        if remembered is not None:
            setattr(args, attr, remembered)
            print(
                f"{side} calibration: remembered {remembered} "
                f"(serial {serials[side]})")

    # Prompt for sides lacking a calibration; --select-calibration forces the
    # picker for every connected side.  The picker itself offers a Calibrate
    # button that switches to the calibration program (returning "calibrate").
    if not args.no_select_calibration and not args.no_display:
        pick_sides = tuple(
            side for side in present_sides
            if (args.left_calib if side == "left" else args.right_calib) is None)
        if args.select_calibration:
            # Explicit re-pick: re-offer the selector for every connected side.
            pick_sides = tuple(present_sides)
        if not pick_sides and not present_sides:
            # No glove connected: still open the selector (with its Calibrate
            # button) so the user can plug in / bind gloves or go calibrate.
            pick_sides = ("left", "right")
        if pick_sides:

            def recheck_connection() -> dict[str, bool]:
                """Re-detect USB presence for the Refresh button."""
                status = {}
                used_refresh_ports: set[str] = set()
                for side in ("left", "right"):
                    try:
                        port, _ = device_manager.resolve_port(side)
                        status[side] = port not in used_refresh_ports
                        used_refresh_ports.add(port)
                    except DeviceNotFoundError:
                        status[side] = False
                return status

            try:
                select_dir = args.calibration_dir or DEFAULT_CALIBRATION_DIR
                selected = select_calibration_files(
                    select_dir, pick_sides,
                    title="Select Adaptive Bimanual IMU Calibrations",
                    connected={
                        side: side in present_sides
                        for side in ("left", "right")
                    },
                    refresh_status=recheck_connection,
                    calibrate_button=True)
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"[Calibration selection error] {exc}", file=sys.stderr)
                return 2
            if selected == "calibrate":
                return "calibrate"
            if selected is None:
                print("Calibration selection cancelled.")
                return 130
            # Apply picks for both hands: the refresh button may discover a
            # hand that was absent at startup, whose pick must not be dropped.
            for side in ("left", "right"):
                picked = selected.get(side)
                if picked is None:
                    continue
                if side == "left":
                    args.left_calib = picked
                else:
                    args.right_calib = picked
                serial = serials.get(side)
                if serial:
                    save_calibration_binding(
                        serial, picked, args.calibration_bindings)
                    print(f"{side} calibration bound: {serial} -> {picked}")
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
    right_root_rotvec_sign = np.asarray(
        saved_display["right_root_rotvec_sign"]
        if saved_display is not None else [1.0, 1.0, 1.0],
        dtype=float).reshape(3)
    if args.separation is not None:
        separation_m = float(args.separation)
    elif saved_display is not None:
        separation_m = float(saved_display["separation_m"])
    else:
        separation_m = 0.30

    runtimes: dict[str, object] = {}
    runtime_backend = None
    registry: dict[str, dict] = {}
    configs: dict[str, CalibrationSettings] = {}
    for side in ("left", "right"):
        if side not in present_sides:
            runtimes[side] = _AbsentHandRuntime(side)
            registry[side] = {
                "usb_serial": "",
                "hardware_id": f"stm32-glove-{side}",
                "channel_to_hand": [],
            }
            continue
        if not calib_paths[side].is_file():
            raise RuntimeError(
                f"Missing {side} calibration: {calib_paths[side]}")
        config = load_calibration(calib_paths[side], side)
        configs[side] = config
        try:
            # Parent-side probe: validate the calibration and lock the solver
            # backend so both hands use the same pipeline.  The live solver
            # runs in its own process (hand_solver_process) and is not shared.
            solver = HandSolver(side, calib_paths[side], args.geometry)
            selected_backend = solver.backend
            if runtime_backend is None:
                runtime_backend = selected_backend
            elif runtime_backend["backend"] != selected_backend["backend"]:
                raise RuntimeError("Left and right solver backends do not match")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            raise RuntimeError(f"{side} calibration is not usable: {exc}") from exc
        runtimes[side] = PublishedHandState(side)
        registry[side] = {
            "usb_serial": serials[side],
            "hardware_id": config.hardware_id,
            "channel_to_hand": config.channel_to_hand,
        }

    if runtime_backend is None:
        # No glove connected: the pipeline label and recording metadata below
        # still need a backend name. hand_pinky_plus_2mm is the fixed local
        # backend (algorithm/runtime_backend.py), so default to it rather than
        # dereferencing None.
        runtime_backend = {"backend": "hand_pinky_plus_2mm"}

    print(f"Left IMU calibration: {calib_paths['left'].resolve()}")
    print(f"Right IMU calibration: {calib_paths['right'].resolve()}")

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
    # perf_counter (QPC) everywhere: time.monotonic ticks at ~15.6 ms on
    # Windows here, which quantized the preview/recording gates to ~32 Hz.
    # 1 ms timer resolution so time.sleep() actually sleeps what it is asked
    # (the Windows default quantum is ~15.6 ms, which capped the main loop at
    # ~30-60 Hz and limited recording to ~31.5 Hz).
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    solve_rate = (float(args.solve_fps)
                  if args.solve_fps > 0
                  else 60.0)
    args.solve_fps = solve_rate  # resolved value, read by hand_solver_process
    print(f"Host software v{APP_VERSION}, SDK V{get_version()}")
    print(f"Solve rate: {solve_rate:g} Hz per hand "
          f"({len(present_sides)} hand(s) connected)")
    started = time.perf_counter()
    processed_frame_index = 0
    state_lock = threading.Lock()
    rec_lock = threading.Lock()
    stop_event = multiprocessing.Event()
    # Held off until the viewer's startup direction capture finishes, then set
    # for the rest of the session (cleared again if the user re-captures).  The
    # magnetometer monitor lives in the solver process and owns the 30 s
    # window, so it has to hear this from the GUI side.
    mag_judge_event = multiprocessing.Event()
    shared = SimpleNamespace(session=None, recorder_error=None)
    hand_procs: dict[str, multiprocessing.Process] = {}
    hand_queues: dict[str, multiprocessing.Queue] = {}
    recorder = None
    last_tactile_versions = {side: -1 for side in present_sides}
    # Last pose actually drawn per hand.  A hand that goes stale keeps its last
    # drawn pose while the other keeps moving, so a stalled link freezes one
    # hand instead of blanking the frame or freezing both.
    last_shown: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    # ``--link-diag`` bookkeeping: the transport's parser counters are
    # cumulative, so the interesting number is the per-second delta.
    prev_link_errors: dict[str, dict[str, int] | None] = {
        side: None for side in present_sides}
    # Sequence-gap loss, kept apart from the parser errors above: those two are
    # different faults (bytes rejected in the parser vs frames the link never
    # delivered) and the whole point of printing both per second is that they
    # have different fixes.
    prev_link_drops: dict[str, dict[str, int] | None] = {
        side: None for side in present_sides}
    session_result: int | str = 0
    try:
        # One solver process per hand: each owns its USB stream, tactile
        # preprocessing and HAND solver on its own core/GIL, so solve bursts
        # no longer preempt the GUI thread.  Published state flows back
        # through a per-hand queue, drained keep-newest by the main loop.
        for side in present_sides:
            config = configs[side]
            # Live state is replaceable: cap IPC backlog so a slow GUI frame
            # cannot turn into visible pose latency on the following frames.
            hand_queues[side] = multiprocessing.Queue(maxsize=8)
            hand_procs[side] = multiprocessing.Process(
                target=hand_solver_process,
                args=(side, ports[side], config.usb_vid, config.usb_pid,
                      calib_paths[side], args.geometry,
                      solve_rate, hand_queues[side], stop_event,
                      mag_judge_event),
                name=f"solver-{side}",
                daemon=True,
            )
            hand_procs[side].start()
        recorder = threading.Thread(
            target=recorder_loop,
            args=(runtimes, present_sides, projector, args,
                  state_lock, rec_lock, stop_event, shared),
            name="recorder", daemon=True)
        recorder.start()
        next_display = started
        display_interval = 1.0 / max(float(args.fps), 1.0)
        next_link_diag = started
        while True:
            now = time.perf_counter()
            for side in present_sides:
                with state_lock:
                    fatal = _drain_hand_queue(
                        hand_queues[side], runtimes[side])
                if fatal is not None:
                    raise RuntimeError(
                        f"{side} solver process failed: {fatal['error']}")
                if not hand_procs[side].is_alive():
                    raise RuntimeError(
                        f"{side} solver process exited unexpectedly "
                        f"(exitcode={hand_procs[side].exitcode})")
            snap = _snapshot(runtimes, present_sides, state_lock)
            if args.link_diag and now >= next_link_diag:
                next_link_diag = now + 1.0
                for side in present_sides:
                    health = snap[side]["link_health"] or {}
                    errors = {
                        key: int(health.get(key) or 0)
                        for key in ("crc_errors", "length_errors",
                                    "discarded_bytes")}
                    previous = prev_link_errors[side] or errors
                    prev_link_errors[side] = errors
                    loss = {
                        key: int(health.get(key) or 0)
                        for key in ("frames", "dropped")}
                    previous_loss = prev_link_drops[side] or loss
                    prev_link_drops[side] = loss
                    latency = snap[side].get("latency")
                    # ``None`` on firmware older than v1.2.11, which does not
                    # answer the probe; a dash says so rather than 0 ms.
                    latency_text = ("--" if latency is None
                                    else f"{latency.rtt_ms:.2f}ms")
                    # English, like every other terminal line in this file; the
                    # on-screen HUD is what carries the bilingual labels.
                    # ``loss+`` is frames the device sent that never arrived, in
                    # the same whole-device unit as ``fram+``; ``dropped+`` is
                    # bytes the parser binned, and the two are unrelated.
                    print(
                        f"[{side}] FPS {snap[side]['fps']:5.1f}  "
                        f"DEV {snap[side]['device_fps']:5.1f} Hz  "
                        f"burst {float(health.get('burst_mean') or 0.0):4.1f}"
                        f"/poll (max {int(health.get('burst_max') or 0)})  "
                        f"gap<={float(health.get('gap_max_s') or 0.0):.3f}s  "
                        f"fram+{loss['frames'] - previous_loss['frames']} "
                        f"loss+{loss['dropped'] - previous_loss['dropped']} "
                        f"crc+{errors['crc_errors'] - previous['crc_errors']} "
                        f"len+{errors['length_errors'] - previous['length_errors']} "
                        f"dropped+{errors['discarded_bytes'] - previous['discarded_bytes']}B  "
                        f"recoveries {int(health.get('recoveries') or 0)}  "
                        f"lat {latency_text}",
                        flush=True)
            if viewer:
                # Tactile frames arrive at ~70 Hz from the hand processes;
                # push them to the overlay whenever a new one is published.
                for side in present_sides:
                    if snap[side]["tactile_version"] != last_tactile_versions[side]:
                        last_tactile_versions[side] = snap[side]["tactile_version"]
                        viewer.set_tactile_side(
                            side, snap[side]["tactile"],
                            snap[side]["tactile_raw"])
            ready = all(
                snap[side]["joints"] is not None
                for side in present_sides)
            # Freshness is judged per hand.  An ``all()`` over both hands meant
            # one hand stalling past ``--max_frame_age`` froze the other with
            # it, which is what made a single bad link look like a hung
            # application.  On a Bluetooth link that is the common case rather
            # than the exception: the two dongles share the 2.4 GHz band, so
            # one hand dropping past the age limit while the other runs on is
            # ordinary.  ``ready`` stays an ``all()`` -- a hand that has never
            # produced a pose is not something to freeze.
            fresh_sides = {
                side for side in present_sides
                if now - snap[side]["host_s"] <= args.max_frame_age}
            any_fresh = ready and bool(fresh_sides)
            if ready and not startup_initialized:
                initial = np.stack([
                    _snap_joints_or_nan(snap, "left"),
                    _snap_joints_or_nan(snap, "right"),
                ]).astype(np.float32)
                if palm_target_bases is None:
                    both_present = len(present_sides) == 2
                    if both_present:
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
                            left_root_rotvec_sign=left_root_rotvec_sign,
                            right_root_rotvec_sign=right_root_rotvec_sign)
                        # save_bimanual_display_config is the only path allowed
                        # to change these persistent left/right-root handedness
                        # corrections.
                        print(f"Bimanual display config saved: {saved_path}")
                    else:
                        # One hand only and no saved config yet: derive the
                        # present hand's basis and mirror it into the empty
                        # slot so the connected hand still sits at its bimanual
                        # anchor.  A partial config is never persisted.
                        palm_target_bases = _synthesize_single_hand_bases(initial)
                        print(
                            "[WARN] Only one hand connected and no saved display "
                            "config; bimanual anchors derived from the connected "
                            "hand and not persisted.",
                            file=sys.stderr)
                fixed_rotations = fixed_bimanual_display_rotations(
                    initial, palm_target_bases)
                recording_metadata = {
                    "mode": "bimanual_auto",
                    "app_version": APP_VERSION,
                    "sides": list(present_sides),
                    "sample_fps": float(
                        getattr(args, "record_fps", None) or args.fps),
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
                            (rotation.as_matrix() if rotation is not None
                             else np.eye(3))
                            for rotation in fixed_rotations
                        ]),
                        "stabilize_palm_bases": False,
                        "left_root_rotvec_sign": left_root_rotvec_sign,
                        "right_root_rotvec_sign": right_root_rotvec_sign,
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
                with rec_lock:
                    shared.session = recording_session
                if args.record_on_start:
                    with rec_lock:
                        recording_session.start(now)
                    print(
                        f"Automatic adaptive recording started: {recording_session.output_path}")
                if not args.no_display:
                    viewer = BimanualViewer(
                        view_state, initial,
                        "Stouch Glove",
                        separation_m=separation_m,
                        palm_target_bases=palm_target_bases,
                        left_root_rotvec_sign=left_root_rotvec_sign,
                        right_root_rotvec_sign=right_root_rotvec_sign,
                        display_config_path=None,
                        recording_enabled=not args.no_save,
                        recording_state=recording_session.state,
                        present_sides=present_sides,
                        display_mode=args.display_mode,
                        mag_judge_event=mag_judge_event,
                        redraw_interval_s=1.0 / max(float(args.fps), 1.0))
                startup_initialized = True
            if any_fresh and now >= next_display:
                if viewer:
                    statuses = {}
                    for side in ("left", "right"):
                        if side in present_sides:
                            statuses[side] = {
                                "fps": snap[side]["fps"],
                                "device_fps": snap[side]["device_fps"],
                                "link_health": snap[side]["link_health"],
                                "missing": snap[side]["missing"],
                                "broken": snap[side]["broken"],
                                "safety": snap[side]["safety"],
                                "contact": snap[side]["contact"],
                                "warming_up": snap[side]["warming_up"],
                                "warmup_remaining_s": snap[side]["warmup_remaining_s"],
                                "warmup_completed": snap[side]["warmup_completed"],
                                "warmup_timed_out": snap[side]["warmup_timed_out"],
                                "raw_present": snap[side]["raw_present"],
                                "mag_levels": snap[side]["mag_levels"],
                                "mag_unread": snap[side]["mag_unread"],
                                "mag_prompt": snap[side]["mag_prompt"],
                                "firmware_version": snap[side]["firmware_version"],
                                "latency": snap[side]["latency"],
                                "link": links.get(side, "usb"),
                                "connected": True,
                                # Abnormal, so the HUD shows it: this hand's
                                # pose above is the last good one, not this
                                # frame's.
                                "stale": side not in fresh_sides,
                            }
                        else:
                            statuses[side] = {
                                "fps": 0.0, "device_fps": 0.0,
                                "link_health": None,
                                "missing": snap[side]["missing"],
                                "broken": [],
                                "safety": False, "contact": None,
                                "warming_up": False,
                                "warmup_remaining_s": 0.0,
                                "warmup_completed": False,
                                "warmup_timed_out": False,
                                "raw_present": None,
                                "mag_levels": None,
                                "mag_unread": None,
                                "mag_prompt": None,
                                "firmware_version": None,
                                "latency": None,
                                "link": "usb",
                                "connected": False,
                                "stale": False,
                            }
                    frame_count = 0
                    with rec_lock:
                        if shared.session is not None:
                            frame_count = shared.session.frame_count
                    # Refresh only the hands that are fresh.  A stale hand keeps
                    # its last drawn pose, so it freezes in place while the
                    # other keeps moving -- the alternative, a NaN slot, makes
                    # the hand vanish from the frame, which reads as a tracking
                    # failure rather than as a link that stopped delivering.
                    # A hand absent from ``present_sides`` still renders as NaN:
                    # absent and stale are different things.
                    for side in present_sides:
                        if side in fresh_sides:
                            last_shown[side] = (
                                _snap_smoothed_or_nan(snap, side),
                                _snap_mesh_or_none(snap, side))
                    shown = {}
                    for side in ("left", "right"):
                        entry = last_shown.get(side)
                        if entry is None:
                            entry = (_snap_smoothed_or_nan(snap, side),
                                     _snap_mesh_or_none(snap, side))
                        shown[side] = entry
                    viewer.update_both(
                        shown["left"][0], shown["right"][0],
                        frame_count, statuses,
                        mesh_slots=[shown["left"][1], shown["right"][1]])
                processed_frame_index += 1
                # Fixed-timestep accumulator.  Resetting from `now` added the
                # loop's wake-up overshoot to every interval and drifted the
                # rate ~8% slow; advancing the deadline keeps the long-term
                # rate exactly args.fps and skips any beat the slow loop
                # missed instead of bursting duplicates.
                while now >= next_display:
                    next_display += display_interval
                if args.max_frames and processed_frame_index >= args.max_frames:
                    break
            elif not ready and now - started > args.startup_timeout:
                missing = [side for side in present_sides
                           if snap[side]["joints"] is None]
                raise RuntimeError(f"Timed out waiting for hand data: {missing}")

            if viewer:
                keep_running = viewer.tick()
                action = viewer.consume_record_action()
                if action == "start":
                    with rec_lock:
                        session = shared.session
                        if (session is not None
                                and session.start(time.perf_counter())):
                            viewer.set_recording_state("recording", 0)
                            print(
                                f"Adaptive recording started: {session.output_path}")
                elif action == "stop":
                    with rec_lock:
                        session = shared.session
                        if session is not None:
                            saved = session.stop_and_save(
                                view_state_metadata(view_state))
                            viewer.set_recording_state(
                                "saved", session.frame_count)
                            if saved is not None:
                                print(f"Adaptive recording saved: {saved}")
                            else:
                                print(
                                    "No valid bimanual frames; no file was created.")
                if viewer.consume_selector_request():
                    session_result = "selector"
                    break
                if not keep_running:
                    break
            if shared.recorder_error is not None:
                _, error_value, error_tb = shared.recorder_error
                raise error_value.with_traceback(error_tb)
            # Deadline-aware pacing: sleep only until the next display slot
            # (clamped 1-10 ms) instead of a fixed 10 ms.  When the loop runs
            # behind the 65 Hz gate the fixed sleep was pure overhead; the
            # floor avoids busy-waiting when there is nothing due.
            time.sleep(max(
                0.001, min(0.010, next_display - time.perf_counter())))
    finally:
        stop_event.set()
        for proc in hand_procs.values():
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
        if recorder is not None:
            recorder.join(timeout=3.0)
        with rec_lock:
            session = shared.session
            if (session is not None
                    and session.state == "recording"):
                saved = session.stop_and_save(
                    view_state_metadata(view_state))
                if saved is not None:
                    print(f"Adaptive recording saved: {saved}")
        if viewer:
            viewer.close()
    elapsed = time.perf_counter() - started
    print(
        f"Adaptive bimanual run finished: {processed_frame_index} preview "
        f"frames in {elapsed:.1f}s "
        f"({processed_frame_index / max(elapsed, 1e-6):.1f} GUI FPS)")
    return session_result


def _run_calibration_session(args) -> int | str:
    """Run one calibration-GUI session; returns ``"done"`` or an exit code."""
    from gui import imu_calibration_gui
    calib_args = imu_calibration_gui._parse_args([])
    calib_args.registry = Path(args.registry)
    calib_args.lang = args.lang
    return imu_calibration_gui.run_calibration_session(calib_args)


def main(argv=None):
    # Required so multiprocessing children bootstrap correctly when the app
    # is packaged (PyInstaller / py2exe); a no-op for a plain `python` run.
    multiprocessing.freeze_support()
    args = _parse_args(argv)
    set_lang(args.lang)
    while True:
        # The live session opens the calibration-file selector (with a
        # Calibrate button).  Selecting files -> live viewer -> exit code;
        # Calibrate -> run the calibration program -> loop back to the selector.
        result = _run_live_session(args)
        if result == "calibrate":
            result = _run_calibration_session(args)
            if result == "done":
                continue
            return result
        if result == "selector":
            # The in-app "back to selector" button: force a re-pick and loop
            # back into a fresh live session (which re-opens the selector).
            args.select_calibration = True
            continue
        return result


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _show_frozen_error(str(exc))
        exit_code = 1
    raise SystemExit(exit_code)
