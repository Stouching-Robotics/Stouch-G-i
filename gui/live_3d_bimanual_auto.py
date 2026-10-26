#!/usr/bin/env python3
"""Adaptive two-STM32 gloves -> HAND2mm-calibrated skeletons -> live 3D.

Data acquisition is the SDK's: each hand gets its own process calling
``sdk.open_glove``, so nothing here builds a stream-and-solver pipeline of its
own any more and neither hand's solver shares an interpreter with the other's.
What arrives is the client edition's frame -- solved joint positions in metres
and finger force in newtons -- so the raw IMU quaternions, the raw pressure ADC
matrix and the magnetometer registers are not on this side of the boundary at
all.  Everything the viewer draws, including the pressure panel, is computed
from those two arrays.

Runs exactly like
``live_3d_bimanual`` (same persistent display config: wrist separation, palm
bases and viewpoint are shared), but does not require both gloves to be
plugged in:

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
import copy
import json
import multiprocessing
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
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
    hand_display_basis, place_hands_at_wrist_anchors, put_text, text_size)
from gui.rendering.tactile import (  # noqa: E402
    HAND_REGIONS_LEFT, HAND_REGIONS_RIGHT,
)
from gui.rendering.tactile_overlay import (  # noqa: E402
    draw_bimanual_tactile_overlay as draw_full_matrix_tactile_overlay,
    tactile_card_rects, tactile_cell_for_scale, tactile_grip_rects,
    tactile_scale_for_cell,
)
from gui.rendering.tactile_pressure_hand import (  # noqa: E402
    PressureHandMapper, draw_pressure_color_points, draw_pressure_tiles,
)
# A display-side helper borrowed from the recorder module: a joint low-pass.
# It touches no sensor data; it lives there because the recorder is its other
# caller.  ``view_state_metadata`` is the same idea for the persistent display
# config's viewpoint, and lives with the session metadata it is written beside.
from sdk.io.recording import JointPositionSmoother  # noqa: E402
from sdk.io.session import view_state_metadata  # noqa: E402
from sdk.device import (  # noqa: E402
    DeviceManager, GloveHandMismatchError, detect_unbound_glove_links,
    link_kind_for_device,
    reboot_bluetooth_dongle, set_glove_serial,
)
from sdk.calibration import find_latest_hand2mm_calibration  # noqa: E402
from sdk import (  # noqa: E402
    DeviceNotFoundError, StreamTimeoutError,
    display_firmware_version, firmware_real_version, get_version, open_glove)
from common.i18n import L, set_lang  # noqa: E402


DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "glove_devices.json"

# Floor for the raw-ADC pressure view.  A resting 16x16 sensor sits at ~2000
# with several hundred counts of spread between cells, and a contact drags its
# neighbours *down*, not up -- so without a floor the cells around a press sink
# into a darker colour band and read as a second, larger contact.  Clamping to
# this value pins all of them to one flat colour that matches an unloaded cell,
# which leaves only the pressed cells visibly changed.  The newton view needs no
# floor: it has a real zero.  The same convention, and the same number, as the
# development viewer's ADC panel.
_DISPLAY_ADC_FLOOR = 2000.0

# Baseline calibration, the same rule as the admin edition and the standalone
# pressure-matrix tools: press 重新采集基线 with the glove unloaded, average
# _BASELINE_FRAMES raw scans per cell as the zero, then draw
# ``max(0, adc - baseline)`` on the ADC card.  No settle delay -- the first
# frame after the press already counts.  The newton card is not baseline-
# corrected: those newtons already have an absolute zero.
_BASELINE_FRAMES = 40


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


@dataclass
class HandRuntime:
    """One hand's live state inside the SDK reader process.

    The link, the stream and the solve belong to the SDK now, so what is kept
    here is only what a frame does not carry: the local joint smoother and the
    newest published values.  Everything on this object is display data --
    joint positions, the solved mesh, and finger force already in newtons.
    """

    side: str
    smoother: JointPositionSmoother
    latest_joints: np.ndarray | None = None
    latest_mesh_vertices: np.ndarray | None = None
    latest_smoothed: np.ndarray | None = None
    latest_uv: np.ndarray | None = None
    latest_force: np.ndarray | None = None
    latest_adc: np.ndarray | None = None
    latest_host_s: float = 0.0
    last_seq: int = -1
    safety: bool = False
    contact: str | None = None
    # No warm-up fields here on purpose.  Warm-up is reported per frame in
    # ``status.details`` and published from there; it is not something this
    # object tracks or knows.  It used to carry ``warming_up`` /
    # ``warmup_completed`` / ``warmup_timed_out`` / ``warmup_remaining_s``
    # defaults that nothing ever wrote, and reading one of them as if it meant
    # something held a gate shut forever -- ``getattr`` with a default makes a
    # field nobody assigns look like a real ``False``.
    fps: float = 0.0
    force_version: int = 0


class _AbsentHandRuntime:
    """Stand-in runtime for a side whose glove is not connected.

    Keeps ``runtimes`` addressable as ``runtimes["left"]`` / ``runtimes["right"]``
    so the result builders can treat both slots uniformly; every sensor field
    stays ``None`` and the keypoints become NaN.
    """

    def __init__(self, side: str):
        self.side = side

    latest_joints = None
    latest_mesh_vertices = None
    latest_smoothed = None
    latest_force = None
    latest_adc = None
    last_seq = -1
    safety = False
    contact = None
    warming_up = False
    warmup_remaining_s = 0.0
    warmup_completed = False
    warmup_timed_out = False
    latest_host_s = 0.0
    fps = 0.0
    device_fps = 0.0
    tactile_fps = 0.0
    # Delivery shape plus the parser's cumulative error tally, as published by
    # the hand process.  ``None`` until the first solve message arrives.
    latest_link_health: dict | None = None
    force_version = 0
    firmware_version = None
    latency = None
    link_kind = "usb"


class PublishedHandState:
    """Parent-side mirror of one hand process's published ``latest_*`` state.

    Each hand's solver process copy-on-publishes through an inter-process
    queue; the main loop drains it keep-newest under ``state_lock``.  Every
    field starts at the same "no data yet" value as :class:`_AbsentHandRuntime`,
    so both classes can back ``runtimes`` interchangeably.
    """

    def __init__(self, side: str):
        self.side = side
        self.latest_joints = None
        self.latest_mesh_vertices = None
        self.latest_smoothed = None
        self.latest_force = None
        self.latest_adc = None
        self.safety = False
        self.contact = None
        self.warming_up = False
        self.warmup_remaining_s = 0.0
        self.warmup_completed = False
        self.warmup_timed_out = False
        self.fps = 0.0
        self.device_fps = 0.0
        self.tactile_fps = 0.0
        self.latest_link_health = None
        self.latest_host_s = 0.0
        self.force_version = 0
        self.firmware_version = None
        self.latency = None
        self.link_kind = "usb"


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


class BimanualViewer(Live3DViewer):
    STARTUP_ALIGNMENT_DURATION_S = 3.0
    STARTUP_ALIGNMENT_SAMPLE_COUNT = 12

    def __init__(self, *args, separation_m=0.30,
                 display_rotations=None, palm_target_bases=None,
                 left_root_rotvec_sign=None,
                 right_root_rotvec_sign=None,
                 display_config_path=DEFAULT_BIMANUAL_DISPLAY_CONFIG,
                 present_sides=None,
                 stop_event=None, **kwargs):
        self._bimanual_slots_cache: np.ndarray | None = None
        self._bimanual_mesh_slots_cache: list[np.ndarray | None] | None = None
        self._display_times: list[float] = []
        self._display_fps = 0.0
        self.present_sides = set(present_sides or ())
        # The session's own shutdown flag, so a worker stops waiting the moment
        # the session ends rather than on its next timeout.
        self._session_stop_event = stop_event
        # Give the user three seconds to hold each connected hand in the
        # desired straight-ahead pose.  The captured palm direction is compared
        # with the calibrated target basis and the resulting display-only
        # correction is retained for the session.
        self._startup_alignment_started_s: float | None = None
        self._startup_alignment_remaining_s = self.STARTUP_ALIGNMENT_DURATION_S
        self._startup_alignment_completed = False
        self._startup_alignment_corrections = [R.identity(), R.identity()]
        self._startup_alignment_samples = {
            side: deque(maxlen=self.STARTUP_ALIGNMENT_SAMPLE_COUNT)
            for side in ("left", "right")
        }
        self.side_status = {
            "left": {"fps": 0.0, "device_fps": 0.0, "tactile_fps": 0.0,
                     "link_health": None,
                     "safety": False, "contact": None,
                     "warming_up": False, "warmup_completed": False,
                     "warmup_remaining_s": 0.0,
                     "firmware_version": None, "latency": None,
                     "connected": "left" in self.present_sides, "stale": False},
            "right": {"fps": 0.0, "device_fps": 0.0, "tactile_fps": 0.0,
                      "link_health": None,
                      "safety": False, "contact": None,
                      "warming_up": False, "warmup_completed": False,
                      "warmup_remaining_s": 0.0,
                      "firmware_version": None, "latency": None,
                      "connected": "right" in self.present_sides, "stale": False},
        }
        self.tactile_frames = {"left": None, "right": None}
        # The raw ADC the forces above were fitted from, for the panel's ADC
        # view.  Kept apart from ``tactile_frames`` because those newtons also
        # drive the on-mesh tiles and points, which have no ADC scale.
        self.tactile_adc_frames = {"left": None, "right": None}
        # Pressure-matrix baseline (the admin edition's convention, shared with
        # the standalone pressure tools).  ``use_baseline`` is the ADC card's
        # view switch -- corrected or raw -- and it only picks what is drawn:
        # the baseline itself is collected when the user presses 重新采集基线
        # with the glove unloaded, and nothing is collected automatically.  It
        # starts off, so a fresh start shows the raw map; until a baseline
        # exists the corrected view falls back to that same raw map.
        self.tactile_raw_frames = {"left": None, "right": None}
        self.tactile_baseline = {"left": None, "right": None}
        self.use_baseline = False
        self._baseline_collecting = False
        self._baseline_buffer = {"left": [], "right": []}
        self._baseline_done = {"left": False, "right": False}
        # The hands a running collection is for: both when both are connected,
        # the one that is connected otherwise.  Set when the run starts.
        self._baseline_sides = ("left", "right")
        self._baseline_target = _BASELINE_FRAMES
        # Bluetooth-dongle reboot feedback shown in the HUD (set from the
        # background worker spawned by the on-canvas "Reboot Dongle" button).
        self._dongle_reboot_status = ""
        # The fixed force-card area is selectable and starts empty; the
        # pressure matrix is the only view it draws.
        self.tactile_panel_mode = "off"
        # Set while the user drags a card's resize handle: the corner they
        # grabbed, and the geometry the drag is measured against.
        self._tactile_resize: dict | None = None
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
        self._qcanvas.set_baseline_controls_visible(
            self.tactile_panel_mode != "off")
        # The canvas builds the switch unchecked; the controller owns the view
        # state, so push it down.  Pushed without emitting: a signal here would
        # arrive as a user click the user never made.
        self._qcanvas.set_baseline_toggle(bool(self.use_baseline))
        self._update_baseline_status()
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
            # A missing measurement is normal on a Bluetooth link and on
            # firmware older than v1.2.11, neither of which answers the probe;
            # show a dash rather than 0 ms.
            latency_parts.append(
                side_tag + (f"{float(latency):.2f} ms"
                            if latency is not None else "—"))
        lines = []
        if fw_parts:
            lines.append(L("固件版本: ", "Firmware: ") + " / ".join(fw_parts))
        if latency_parts:
            lines.append(L("延迟: ", "Latency: ") + " / ".join(latency_parts))
        lines.append(
            L(f"界面: {self._display_fps:5.1f} FPS",
              f"GUI: {self._display_fps:5.1f} FPS"))
        for side in ("left", "right"):
            status = self.side_status[side]
            side_name = L("左手", "LEFT") if side == "left" else L("右手", "RIGHT")
            if not status.get("connected"):
                lines.append(f"{side_name}: {L('未连接', 'NOT CONNECTED')}")
                continue
            if status.get("waiting_for_data"):
                lines.append(f"{side_name}: {L('等待首帧数据', 'WAITING FOR FIRST FRAME')}")
                continue
            # Solve rate first, then each device stream's own arrival rate:
            # when they differ, the gap is the host dropping frames, not the
            # glove withholding them, and that is the whole point of showing
            # all three.  Attitude and pressure run as separate streams at the
            # same nominal rate, so one number cannot stand in for the other --
            # a hand whose IMUs settle would otherwise take its pressure rate
            # with it, and a panel that looks frozen can be a stream that is
            # genuinely delivering at half rate.
            device_fps = float(status.get("device_fps") or 0.0)
            tactile_fps = float(status.get("tactile_fps") or 0.0)
            parts = [
                f"{status['fps']:5.1f} FPS",
                L(f"IMU {device_fps:5.1f} Hz", f"IMU {device_fps:5.1f} Hz"),
                L(f"压力 {tactile_fps:5.1f} Hz", f"TACTILE {tactile_fps:5.1f} Hz"),
            ]
            if status.get("link") == "bluetooth":
                parts.append(L("蓝牙", "BT"))
            # Only the countdown is worth the space; a finished warm-up is the
            # normal state and needs no announcement.
            if status.get("warming_up"):
                parts.append(
                    L("预热", "WARM-UP")
                    + f" {status.get('warmup_remaining_s', 0.0):.1f}s")
            if status["safety"]:
                parts.append(L("安全", "SAFETY"))
            lines.append(f"{side_name}: " + "  ".join(parts))
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

    def _orientation_recollect_label(self) -> str:
        return L("重新采集手方向", "Recapture hand direction")

    def _on_recollect_orientation(self) -> None:
        """Restart the three-second direction capture."""

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
        self._draw_startup_alignment_guide(img)
        return img

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

    def _resizable_tactile_frames(self):
        """The frames whose card is on screen with a resize handle.

        Only the matrix panels draw a resizable card, and only while one of
        them is the selected view.  Both views share the card's geometry, so the
        handles follow whichever matrix that view is showing.
        """
        if self.tactile_panel_mode == "matrix":
            return self.tactile_frames
        if self.tactile_panel_mode == "adc":
            return self.tactile_adc_frames
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

    def consume_selector_request(self) -> bool:
        requested = self.request_selector
        self.request_selector = False
        return requested

    def set_tactile_side(self, side, force_n, adc=None):
        """Store one hand's force matrix, and the ADC it was fitted from.

        ``force_n`` is the frame's ``force_n`` field: a (16, 16) float32 array
        of newtons straight out of the client-edition SDK, with NaN in the cells
        no fitted region covers and ``None`` when the frame carried no pressure
        scan at all.  Nothing is converted or baseline-corrected here -- the
        ADC-to-newton step already happened on the SDK side.

        ``adc`` is that same frame's ``tactile_adc``: the raw 12-bit counts for
        the panel's ADC view.  The raw scan is kept here -- it is what a running
        baseline run averages -- and the ADC card draws
        :meth:`_adc_display_value` of it: floor-clamped raw, or
        ``max(0, adc - baseline)`` when 使用基线 is on.  See
        ``_DISPLAY_ADC_FLOOR`` for why the floor belongs to the display and not
        to the reading.  The newton matrix is deliberately not clamped: it has
        a real zero, and a floor there would be a lie.
        """
        matrix = (None if force_n is None else
                  np.asarray(force_n, np.float32).reshape(16, 16))
        raw16 = (None if adc is None else
                 np.asarray(adc, np.float32).reshape(16, 16))
        self.tactile_frames[side] = matrix
        self.tactile_raw_frames[side] = raw16
        if self._baseline_collecting and raw16 is not None:
            self._baseline_sample(side, raw16)
        self.tactile_adc_frames[side] = self._adc_display_value(side)
        # No _redraw_requested here: the pressure stream bumps faster than the
        # display gate and forcing a redraw per bump made the draw rate track
        # the sensor rate instead of the gate (visible as jitter while moving).
        # The panel is picked up by the next scheduled gate frame instead.

    def _adc_display_value(self, side):
        """The ADC card's picture for one hand: corrected or floor-clamped raw."""

        raw = self.tactile_raw_frames.get(side)
        if raw is None:
            return None
        if self.use_baseline and self.tactile_baseline.get(side) is not None:
            return np.maximum(0.0, raw - self.tactile_baseline[side])
        # Uncalibrated: the floor keeps the dip around a contact from turning
        # into a dark patch of its own.  See ``_DISPLAY_ADC_FLOOR``.
        return np.maximum(_DISPLAY_ADC_FLOOR, raw)

    def _tactile_panel_label(self) -> str:
        return L("触觉面板", "Tactile panel")

    def _tactile_panel_mode_options(self) -> list[tuple[str, str]]:
        # Every mode the panel answers to is offered here, and this list is the
        # only thing that decides which ones a user can reach: the guards
        # (``_on_tactile_panel_mode``), the resize handle
        # (``_resizable_tactile_frames``) and the draw branch in
        # ``_draw_tactile`` are keyed by mode name alone.  Leaving an entry out
        # hides a finished feature rather than disabling it.
        return [
            ("off", L("关闭", "Off")),
            ("matrix", L("压力矩阵 (N)", "Pressure matrix (N)")),
            ("adc", L("压力矩阵 (ADC)", "Pressure matrix (ADC)")),
        ]

    def _on_tactile_panel_mode(self, mode: str) -> None:
        mode = str(mode)
        self.tactile_panel_mode = (
            mode
            if mode in {"off", "matrix", "adc"}
            else "off"
        )
        # The pressure-baseline toggle acts on the ADC card's own map, so it
        # rides with the panel: with the panel off there is nothing on screen
        # to correct.  Only visibility changes; ``use_baseline`` is left alone.
        self._qcanvas.set_baseline_controls_visible(
            self.tactile_panel_mode != "off")
        # Any resize in flight was aimed at a card that is no longer there.
        self._tactile_resize = None
        self._redraw_requested = True

    # ---- pressure baseline -------------------------------------------------

    def _baseline_toggle_label(self) -> str:
        return L("使用基线", "Use baseline")

    def _baseline_recollect_label(self) -> str:
        return L("重新采集基线", "Re-collect baseline")

    def _on_use_baseline_toggled(self, checked) -> None:
        # Checking the box is itself the calibration: one press on 使用基线
        # collects a fresh baseline for the connected hands -- both of them
        # when both are connected, the single one otherwise -- and then shows
        # the corrected map.  The overlay asks the user to keep the glove
        # unloaded while it runs, because the reading it averages becomes zero.
        #
        # Unchecking is the opposite and nothing more: a pure view switch,
        # which neither collects nor discards.  Dropping the baseline there
        # would mean re-calibrating on every glance at the raw map, and the
        # raw map is exactly what someone checks first when a number looks
        # wrong.
        self.use_baseline = bool(checked)
        if self.use_baseline:
            self._start_baseline_collection()
        for side in ("left", "right"):
            self.tactile_adc_frames[side] = self._adc_display_value(side)
        self._update_baseline_status()
        self._redraw_requested = True

    def _on_recollect_baseline(self) -> None:
        # No ``use_baseline`` guard: calibration is not a property of the view
        # in the reference tools, and neither is it here.  Pressing it again
        # mid-collection restarts, which is also the way out of a run that is
        # waiting on a hand that stopped sending frames -- so this button is
        # never greyed out, and no separate "clear" control is needed.
        self._start_baseline_collection()
        self._redraw_requested = True

    def _start_baseline_collection(self) -> None:
        self._baseline_collecting = True
        # One press covers both hands when both are connected, and just the one
        # hand when only one is.  The fallback keeps the run open when it was
        # started before any hand is known: the contributing-frames rule in
        # ``_finish_baseline_if_complete`` then decides who counts.
        targets = tuple(side for side in ("left", "right")
                        if side in self.present_sides)
        self._baseline_sides = targets or ("left", "right")
        self._baseline_buffer = {"left": [], "right": []}
        self._baseline_done = {"left": False, "right": False}
        self._update_baseline_status()

    def _baseline_sample(self, side, raw16) -> None:
        """Advance the baseline calibration by one raw frame.

        The first frame after the button press already counts -- there is no
        settle delay, matching the standalone pressure tools, where the user is
        told to press the button while the glove is unloaded and the next
        ``_BASELINE_FRAMES`` frames are averaged.  Sides are independent: each
        keeps sampling until it has its own ``_baseline_target`` frames, so a
        two-hand run calibrates both hands from one press.
        """
        if self._baseline_done.get(side, False):
            return
        self._baseline_buffer[side].append(raw16)
        if len(self._baseline_buffer[side]) >= self._baseline_target:
            stack = np.stack(
                self._baseline_buffer[side][-self._baseline_target:], axis=0)
            # Mean, not median: the same per-cell average the admin edition
            # computes, so a value read here matches one read there.
            self.tactile_baseline[side] = np.mean(
                stack, axis=0).astype(np.float32)
            self._baseline_done[side] = True
        self._finish_baseline_if_complete()

    def _finish_baseline_if_complete(self) -> None:
        if not self._baseline_collecting:
            return
        # "Every side that contributed frames", not "every side the runtimes
        # announced".  Two failures hide behind the wrong test: with an empty
        # ``present_sides`` the condition is vacuously true, so the collection
        # retires and the status claims a baseline that was never collected
        # (the corrected view would then silently keep showing raw numbers),
        # while an announced side that stops delivering frames before its last
        # sample would hold the collection open forever.
        contributed = [side for side in ("left", "right")
                       if self._baseline_buffer[side]]
        if contributed and all(self._baseline_done.get(side)
                               for side in contributed):
            self._baseline_collecting = False
            # Refresh the corrected map immediately for sides whose raw data is
            # already available, without waiting for their next tactile frame.
            for side in ("left", "right"):
                if self.tactile_raw_frames.get(side) is not None:
                    self.tactile_adc_frames[side] = self._adc_display_value(side)
        self._update_baseline_status()

    def _update_baseline_status(self) -> None:
        qcanvas = getattr(self, "_qcanvas", None)
        if qcanvas is None:
            return
        qcanvas.set_baseline_status(self._baseline_status_text())
        # The collect button is left live even while a run is going: pressing
        # it again restarts the run, which is how a collection that is waiting
        # on a hand that stopped sending frames gets unstuck.
        if self._baseline_collecting:
            qcanvas.show_baseline_overlay(
                L("基线采集中，请保持空载",
                  "Collecting baseline, keep the glove unloaded"))
        else:
            qcanvas.hide_baseline_overlay()

    def _baseline_exists(self) -> bool:
        return any(self.tactile_baseline.get(side) is not None
                   for side in ("left", "right"))

    def _baseline_status_text(self) -> str:
        """The baseline's own state, independent of which map is on screen.

        The reference tools keep these separate -- a 基线: 已校准/未校准 readout
        beside the 原始/校准后 switch -- so that looking at the raw map never
        reads as "no baseline".
        """

        if self._baseline_collecting:
            # The slowest hand taking part that is actually sending, matching
            # the rule that retires the collection: a side that never delivers
            # a frame is not part of the run and must not pin the readout at
            # 0/N forever.
            counts = [len(self._baseline_buffer.get(side, []))
                      for side in self._baseline_sides
                      if self._baseline_buffer.get(side)]
            n = min(counts) if counts else 0
            return L(
                f"基线采集中… {n}/{self._baseline_target}（保持空载）",
                f"Collecting baseline… {n}/{self._baseline_target} (keep free)")
        if self._baseline_exists():
            return L(
                f"已校准（{self._baseline_target}帧 · 均值）",
                f"Calibrated ({self._baseline_target} frames · mean)")
        return L("未校准（显示原始值）", "Not calibrated (raw values)")

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
        # The on-mesh overlays paint ``force_n`` -- newtons already fitted by
        # the SDK -- because the mesh has no ADC scale to colour against, and
        # the newton panel does the same.  The ADC panel paints the other
        # matrix: the raw scan those newtons were fitted from, in sensor counts.
        # ``_adc_display_value`` already subtracted the baseline when 使用基线
        # is on, so the colour bands have to follow the same per-hand answer --
        # a corrected frame is 0..~500 and would all but vanish on the raw
        # scale, while an uncalibrated ~2000 would go solid red on the
        # corrected one.
        #
        # No threshold is forwarded to either card: the client edition has no
        # ADC-count gate (the old tactile_thr slider is gone), and every force
        # value here is 0..5 N, so the ADC scale it counted in has nothing to
        # say.  The floor that separates contact from an idle sensor is
        # ``_NEWTON_CONTACT_FLOOR_N`` inside ``gui.rendering.tactile_pressure_hand``.
        newton_ready = {"left": False, "right": False}
        # Per hand, because one glove can be calibrated while the other is not.
        baseline_ready = {
            side: bool(self.use_baseline
                       and self.tactile_baseline.get(side) is not None)
            for side in ("left", "right")
        }
        if self.display_mode == "hand" and self._pressure_mappers:
            camera = self._camera()
            mesh_slots = self._controlled_display_mesh_slots()
            if self.pressure_display_mode == "tiles":
                draw_pressure_tiles(
                    img, camera, mesh_slots, self.tactile_frames,
                    self._pressure_mappers, newton_ready, newtons=True)
            elif self.pressure_display_mode == "points":
                draw_pressure_color_points(
                    img, camera, mesh_slots, self.tactile_frames,
                    self._pressure_mappers, newton_ready, newtons=True)
            # "none" -> draw nothing on the mesh surface
        if self.tactile_panel_mode == "matrix":
            draw_bimanual_tactile_overlay(
                img,
                self.tactile_frames,
                scale=self.tactile_scale,
                baseline_corrected=False,
                linear_fitted=True,
            )
        elif self.tactile_panel_mode == "adc":
            # The same card fed the scan the newtons were fitted from, so the
            # per-cell numbers are what the sensor reported instead of what the
            # fit made of it -- on the raw band scale, or the corrected one once
            # a baseline exists and 使用基线 is on.
            draw_bimanual_tactile_overlay(
                img,
                self.tactile_adc_frames,
                scale=self.tactile_scale,
                baseline_corrected=baseline_ready,
                linear_fitted=False,
            )


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
                "force": getattr(runtime, "latest_force", None),
                "adc": getattr(runtime, "latest_adc", None),
                "safety": bool(runtime.safety),
                "contact": runtime.contact,
                "warming_up": bool(getattr(runtime, "warming_up", False)),
                "warmup_remaining_s": float(
                    getattr(runtime, "warmup_remaining_s", 0.0)),
                "warmup_completed": bool(
                    getattr(runtime, "warmup_completed", False)),
                "fps": float(getattr(runtime, "fps", 0.0)),
                "device_fps": float(getattr(runtime, "device_fps", 0.0)),
                "tactile_fps": float(getattr(runtime, "tactile_fps", 0.0)),
                "link_health": getattr(runtime, "latest_link_health", None),
                "host_s": float(runtime.latest_host_s),
                "force_version": int(getattr(runtime, "force_version", 0)),
                "firmware_version": getattr(runtime, "firmware_version", None),
                "latency": getattr(runtime, "latency", None),
                "link_kind": getattr(runtime, "link_kind", "usb"),
                "connected": side in present_sides,
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


def _snap_force_or_none(snap, side) -> np.ndarray | None:
    """One hand's force matrix in newtons, or ``None`` if there is nothing."""

    value = snap[side].get("force")
    if value is None:
        return None
    matrix = np.asarray(value, np.float32)
    return matrix if matrix.shape == (16, 16) else None


def _snap_adc_or_none(snap, side) -> np.ndarray | None:
    """One hand's raw pressure ADC matrix, or ``None`` if there is nothing.

    Not floored here: this is still the reading the device sent.  The floor
    belongs to the view, and ``set_tactile_side`` applies it.
    """

    value = snap[side].get("adc")
    if value is None:
        return None
    matrix = np.asarray(value, np.float32)
    return matrix if matrix.shape == (16, 16) else None


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


def _cancel_stopped_worker_queue_joins(stop_event, *output_queues) -> None:
    """Do not flush abandoned preview/results after a requested session stop.

    Queue.put() only feeds a local buffer. At process exit multiprocessing
    normally joins its feeder until the pipe is drained; the parent has already
    stopped consuming these session queues and is waiting in Process.join().
    Drop that unread tail only on an explicit stop. Unexpected failures still
    flush their fatal report while the parent continues to consume messages.
    Recording is owned by the parent and does not use these feeder buffers as
    its save queue. Call this after closing the glove, never instead of it.
    """
    if stop_event.is_set():
        for output_queue in output_queues:
            if output_queue is not None:
                output_queue.cancel_join_thread()


# How long the SDK reader waits for a paired frame before giving the loop back.
# A Bluetooth dongle hands its frames over one RF event at a time, so the window
# is a dongle's own batch latency -- the same 250 ms the solve loop treats as
# the boundary between "current" and "a timestamp jump".
_READ_TIMEOUT_S = 0.25
# Cap on frames drained per turn.  A producer that keeps handing frames over
# must not keep the reader from ever publishing, which is the same guard the
# parent's drain applies.
_MAX_DRAIN = 8


def _open_sdk_link(side, calib_paths, links, serials, registry_path):
    """Open one hand, through the SDK's one entry point.

    Every choice was already made in the parent -- calibration file, link kind
    and serial -- and is handed over here, so opening the glove does not
    re-discover any of it.  Each hand has its own process and so opens its own
    link: ``open_glove`` rather than ``open_bimanual``, which would resolve and
    pair both links inside one interpreter and put the two solvers back on each
    other's GIL.
    """

    return open_glove(
        side,
        calibration=calib_paths[side],
        link=links[side],
        serial_number=serials[side],
        registry=registry_path,
    )


def _read_newest(glove, timeout):
    """The newest frame of this hand, never a replay of a backlog.

    ``read`` hands back one queued frame at a time from the oldest end, so a
    reader that fell behind would publish history rather than the current pose.
    The newest frame is what this keeps, and a frame per hand is all a hand's
    process can publish -- one process reads one glove, so there is no pairing
    to do here.  ``timeout`` bounds the first read only, so an idle glove costs
    nothing while the frames already queued are taken in one go.  Returns
    ``({side: HandFrame}, frames_drained)``; the mapping is one entry wide and
    keeps the publishing loop below keyed by side, which is also how the parent
    mirrors it.
    """

    newest: dict[str, object] = {}
    drained = 0
    wait = timeout
    while True:
        try:
            frame = glove.read(wait)
        except StreamTimeoutError:
            break
        newest[str(frame.side)] = frame
        drained += 1
        # Bounded like the parent's own drain: a producer that keeps handing
        # frames over must not keep this loop from ever publishing.
        if drained >= _MAX_DRAIN:
            break
        wait = 0.0
    return newest, drained


def hand_solver_process(side, calib_paths, links, serials, registry_path,
                        solve_rate, state_queue, stop_event,
                        contact_control=None, root_limits_control=None):
    """One process for one hand's SDK pipeline, publishing to the parent.

    The parent is the consumer: it drains ``state_queue`` keep-newest and
    mirrors the fields into a :class:`PublishedHandState` under its own lock.
    One hand per process, because that is what gives the hand its own GIL: two
    hands in one process solve at roughly 42 Hz each rather than the 60 Hz their
    devices deliver, and under a Bluetooth dongle's batched arrival the backlog
    that builds behind that shared interpreter is what the link then has to
    carry.  Keeping the loop here rather than on the GUI's thread is what stops
    a solve burst from preempting the draw.  Fatal errors are shipped as a
    ``{"type": "fatal", "side": ...}`` message before the process exits.
    """
    try:
        glove = _open_sdk_link(
            side, calib_paths, links, serials, registry_path)
    except BaseException as exc:
        _publish_latest(state_queue, {
            "type": "fatal",
            "side": side,
            "error": f"{type(exc).__name__}: {exc}",
        })
        return
    print(f"[SDK] {side} pipeline started (pid {os.getpid()})")
    # The mesh is read through ``Glove.mesh_vertices_m``, the hand's own public
    # facade: a frame carries the pose and not the mesh, which is the state that
    # lives on the engine.  That property is the newest solved mesh rather than
    # one keyed to a frame, which is what the renderer paints from.
    runtime = HandRuntime(side=side, smoother=JointPositionSmoother(0.015))
    # Solve-rate budget, per hand: the SDK pipeline solves every frame it
    # receives, so this no longer saves the solve -- it keeps what is published
    # (the drawn pose, the HUD's solve rate) at the cadence it had.  Device time
    # is a fractional, non-blocking budget rather than a frame counter, so a
    # burst is thinned out instead of discarded whole.
    solve_rate = min(max(float(solve_rate), 1.0), 80.0)
    solve_credit = 0.0
    last_timestamp_us: int | None = None
    # The last force matrix published for this hand, and the ADC it was fitted
    # from.  The SDK re-attaches the newest pressure scan to every attitude
    # frame until the next one lands, and re-publishing a repeat would drive the
    # parent's force history at the attitude rate instead of the scan rate -- so
    # only a scan that differs from the last published one goes out.  Both
    # matrices are compared, because they are one scan and the parent draws
    # either of them: ``equal_nan`` because the uncovered cells of a force
    # matrix are NaN on purpose, which is also why the ADC has to be in the
    # comparison at all -- a scan that moved only there is unchanged in the
    # force matrix and very much changed on the ADC panel.
    last_force: np.ndarray | None = None
    last_adc: np.ndarray | None = None
    publish_times: list[float] = []
    # Delivery shape (frames per read, arrival gaps) and the transport's own
    # parser error counters, which travel on the frame.  Together they decide
    # whether a low FPS is the link's fault or the host discarding frames.
    link_health = _LinkHealth()
    try:
        while not stop_event.is_set():
            if contact_control is not None:
                strength = contact_control.value
                glove.set_contact_controls(strength >= 0.0, max(0.0, strength))
            if root_limits_control is not None:
                glove.set_root_limits_enabled(bool(root_limits_control.value))
            frames, drained = _read_newest(glove, _READ_TIMEOUT_S)
            if not frames:
                continue
            # perf_counter (QPC) is system-wide monotonic, so the host_s
            # published here is directly comparable across the process
            # boundary in the parent.
            now = time.perf_counter()
            for hand_side, frame in frames.items():
                # The frame's own force in newtons: the finger regions are
                # already fitted, and the cells no fitted region covers arrive
                # as NaN.  ``None`` means this frame carries no pressure scan.
                force = frame.force_n
                if force is not None:
                    force = np.asarray(force, np.float32).reshape(16, 16).copy()
                    # The raw ADC the newtons were fitted from rides the same
                    # message: one scan, one version.  Both matrices are compared
                    # before publishing, because in the ADC view a cell no fitted
                    # region covers is a reading too -- it is NaN in the newton
                    # matrix, so a scan that moved only there would otherwise be
                    # dropped as unchanged.
                    adc = frame.tactile_adc
                    adc = (None if adc is None else
                           np.asarray(adc, np.float32).reshape(16, 16).copy())
                    if (last_force is None
                            or not np.array_equal(
                                force, last_force, equal_nan=True)
                            or (adc is not None
                                and (last_adc is None
                                     or not np.array_equal(adc, last_adc)))):
                        last_force = force
                        last_adc = adc
                        runtime.latest_force = force
                        runtime.latest_adc = adc
                        runtime.force_version += 1
                        _publish_latest(state_queue, {
                            "type": "force",
                            "side": hand_side,
                            "force": force,
                            "adc": adc,
                            "force_version": runtime.force_version,
                        })
                timestamp_us = int(frame.timestamp_us)
                previous_us = last_timestamp_us
                last_timestamp_us = timestamp_us
                if previous_us is None:
                    # Publish the first pose immediately.
                    publish = True
                elif solve_rate < 80.0:
                    # The firmware timestamp is a wrapping uint32 counter.
                    elapsed_us = (timestamp_us - previous_us) & 0xFFFFFFFF
                    if elapsed_us > 250_000:
                        # Reconnect/timestamp reset: do not accumulate a burst
                        # of historical solve credit.
                        solve_credit = 0.0
                        publish = True
                    else:
                        solve_credit = min(
                            solve_credit + elapsed_us * solve_rate / 1_000_000.0,
                            2.0,
                        )
                        publish = solve_credit >= 1.0
                        if publish:
                            solve_credit -= 1.0
                else:
                    publish = True
                if not publish:
                    continue
                sample_s = timestamp_us / 1_000_000.0
                joints = np.asarray(frame.joints_raw_m, np.float32).copy()
                # The local smoother, on the raw joints and at its own
                # parameters: the SDK's own smoothing is not what this view
                # draws, and swapping it would change how the hand moves.
                smoothed = np.asarray(
                    runtime.smoother.update(frame.joints_raw_m, sample_s),
                    np.float32).copy()
                publish_times.append(now)
                publish_times = [
                    stamp for stamp in publish_times if now - stamp <= 1.0]
                if len(publish_times) >= 2:
                    window_s = publish_times[-1] - publish_times[0]
                    measured_fps = (
                        (len(publish_times) - 1) / window_s
                        if window_s > 0.0 else 0.0)
                else:
                    measured_fps = 0.0
                status = frame.status
                details = getattr(status, "details", None) or {}
                _publish_latest(state_queue, {
                    "type": "solve",
                    "side": hand_side,
                    "joints": joints,
                    "mesh_vertices": glove.mesh_vertices_m,
                    "smoothed": smoothed,
                    "safety": bool(status.safety_intervened),
                    "contact": status.active_contact,
                    "warming_up": bool(details.get("warming_up", False)),
                    "warmup_remaining_s": float(
                        details.get("warmup_remaining_s") or 0.0),
                    "warmup_completed": bool(
                        details.get("warmup_completed", False)),
                    "fps": measured_fps,
                    # The device's own arrival rate, counted by the SDK: the two
                    # diverge whenever the publish budget above holds a frame
                    # back, and showing only one makes a busy host look like a
                    # slow glove.
                    "device_fps": float(frame.device_fps or 0.0),
                    # The pressure stream's own arrival rate, from the same
                    # SDK-side window.  Independent of the attitude rate above:
                    # the scan this frame carries may be the newest one standing
                    # over from several frames back.
                    "tactile_fps": float(frame.tactile_fps or 0.0),
                    "host_s": now,
                    "firmware_version": frame.firmware_version,
                    # Host<->device round trip in milliseconds, or ``None`` on a
                    # link that cannot measure it: a Bluetooth link is half
                    # duplex, and firmware older than v1.2.11 does not answer.
                    "latency": frame.latency_ms,
                    "link_kind": frame.link_kind,
                    # Plain dict of numbers: this crosses the process queue.
                    # ``link_errors`` counts bytes the parser rejected,
                    # ``link_drops`` frames the link never delivered; neither
                    # rises for frames this loop deliberately discards, so
                    # together they say where the frames went.
                    "link_health": link_health.snapshot(now, {
                        **(frame.link_errors or {}),
                        **(frame.link_drops or {}),
                    }),
                })
            link_health.observe(now, drained)
    except BaseException as exc:  # shipped to the parent, not swallowed
        _publish_latest(state_queue, {
            "type": "fatal",
            "side": side,
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        try:
            glove.close()
        finally:
            _cancel_stopped_worker_queue_joins(
                stop_event, state_queue)


def _drain_hand_queue(state_queue, runtimes) -> dict | None:
    """Apply pending child messages to the parent-side mirror state.

    Called by the main loop with ``state_lock`` held, once per hand's queue --
    the hands used to share one, which let either hand's publisher evict the
    other's unread messages.  Messages are applied keep-newest per hand and
    kind, so a slow GUI frame only ever sees the latest published values.
    Returns the fatal-error message when the reader process crashed, else
    ``None``.
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
        elif kind in ("force", "solve"):
            latest_by_kind[(message.get("side"), kind)] = message
    for (side, kind), message in latest_by_kind.items():
        state = runtimes[side]
        if kind == "force":
            state.latest_force = message["force"]
            state.latest_adc = message.get("adc")
            state.force_version = message["force_version"]
            continue
        state.latest_joints = message["joints"]
        state.latest_mesh_vertices = message.get("mesh_vertices")
        state.latest_smoothed = message["smoothed"]
        state.safety = message["safety"]
        state.contact = message["contact"]
        state.warming_up = message["warming_up"]
        state.warmup_remaining_s = message["warmup_remaining_s"]
        state.warmup_completed = message["warmup_completed"]
        state.fps = message["fps"]
        state.device_fps = float(message.get("device_fps") or 0.0)
        state.tactile_fps = float(message.get("tactile_fps") or 0.0)
        state.latest_link_health = message.get("link_health")
        state.latest_host_s = message["host_s"]
        state.firmware_version = message["firmware_version"]
        state.latency = message.get("latency")
        state.link_kind = message.get("link_kind") or "usb"
    return fatal


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
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--fps", type=float, default=60.0,
        help="Preview refresh rate (Hz), default 60")
    parser.add_argument(
        "--solve-fps", type=float, default=0.0,
        help="Per-hand publish decimation target in Hz; 0 = auto (60). "
             "The SDK pipeline solves every frame it receives, so this caps "
             "the pose/HUD update rate rather than the solve cost, and it is "
             "what keeps a Bluetooth batch from being drawn as a backlog.")
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--max-frame-age", type=float, default=0.25)
    parser.add_argument(
        "--link-diag", action="store_true",
        help="Print one link-diagnostics line per hand per second: how many "
             "frames each read drained, the longest arrival gap, and the "
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
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)
    return args


@dataclass
class _DetectedHands:
    """Which gloves are on the bus right now, and how each one is reached.

    Built from scratch by :func:`detect_gloves` on every detection -- once at
    startup and again every time the calibration selector's Refresh button is
    pressed -- so those two callers cannot drift apart: a glove the Refresh
    finds is resolved exactly like one that was there when the app started.
    """

    sides: list[str] = field(default_factory=list)
    ports: dict[str, str] = field(default_factory=dict)
    serials: dict[str, str] = field(default_factory=dict)
    links: dict[str, str] = field(default_factory=dict)
    # Hands found behind a link no registry entry claims.  A Bluetooth dongle
    # hides the glove's STM32 serial, so only the glove's own firmware can say
    # which hand it is; the caller remembers these to skip the probe next time.
    auto_detected: dict[str, str] = field(default_factory=dict)

    def adopt_into(self, present_sides: list[str], ports: dict[str, str],
                   serials: dict[str, str], links: dict[str, str]) -> None:
        """Make this detection the live session's view of the bench.

        In place, and that is the point: the session holds these four
        containers for its whole life, and its readers -- the recorder thread,
        the viewer's status builder, each solver process -- captured the
        objects rather than the names.  Replacing them would leave every one
        of those readers on the startup snapshot; the Refresh button writing
        nothing here at all is what made the hand it found show up as
        connected and then never open.
        """

        present_sides[:] = self.sides
        ports.clear()
        ports.update(self.ports)
        serials.clear()
        serials.update(self.serials)
        links.clear()
        links.update(self.links)


def detect_gloves(device_manager, registry: Path,
                  announce: bool = False) -> _DetectedHands:
    """Resolve whichever hands are plugged in right now.

    The registry is consulted first, then every unclaimed STM32 link is probed
    once for the hand side its firmware reports.  That probe is what makes a
    freshly plugged Bluetooth dongle usable on the first launch, and it is just
    as necessary on a later Refresh: a dongle that appeared after startup is
    not in the registry either, so ``resolve_port`` alone would never find it.

    ``announce`` prints what was found, for a console the user is watching; the
    startup summary lines are the caller's, since only the startup knows
    whether an empty bench is worth a warning.
    """

    try:
        unbound_links = detect_unbound_glove_links(registry)
    except Exception as exc:  # probing is best-effort, never fatal
        print(f"[WARN] unbound glove probe failed: {exc}", file=sys.stderr)
        unbound_links = {}
    found = _DetectedHands()
    used_ports: set[str] = set()
    for side in ("left", "right"):
        try:
            port, serial = device_manager.resolve_port(side)
        except GloveHandMismatchError as exc:
            # The link is here, and says which hand it is -- it is just filed on
            # the other side.  Reported rather than swallowed into "NOT
            # CONNECTED", because the two failures send the user to different
            # places: one to the registry entry, the other to a cable.
            if announce:
                print(f"{side} device: {exc}", file=sys.stderr)
            continue
        except DeviceNotFoundError:
            discovered = unbound_links.get(side)
            if discovered is None:
                if announce:
                    print(f"{side} device: NOT CONNECTED (skipped)")
                continue
            port, serial = discovered
            found.auto_detected[side] = serial
            if announce:
                print(
                    f"{side} device: firmware reports this hand; unbound "
                    f"link={link_kind_for_device(port) or 'usb'}  port={port}")
        # A bimanual runtime must never open one physical CDC port twice.  This
        # is a defensive guard for stale registries, unusual USB drivers, or a
        # future resolver fallback: the second logical side is absent rather
        # than being fed the first hand's frames.
        if port in used_ports:
            if announce:
                print(
                    f"{side} device: DUPLICATE PORT {port} (skipped; "
                    "already assigned to another hand)")
            continue
        used_ports.add(port)
        found.sides.append(side)
        found.ports[side] = port
        found.serials[side] = serial
        link = link_kind_for_device(port) or "usb"
        found.links[side] = link
        # A "bluetooth" link means the serial below is the dongle's, not the
        # glove's: the glove behind it is not a USB device, so the dongle is
        # what the registry and the calibration bindings key on.
        if announce:
            print(f"{side} device: link={link}  serial={serial}  port={port}")
    return found


def refresh_detected_hands(device_manager, registry: Path,
                           present_sides: list[str], ports: dict[str, str],
                           serials: dict[str, str],
                           links: dict[str, str]) -> dict[str, bool]:
    """Re-detect the bench, adopt the result, and report it to the selector.

    What the calibration selector's Refresh button runs.  The detection is
    *adopted*, not merely reported: the dialog can only draw its ✓/✗ from the
    mapping returned here, while every open path in the live session --
    ``PublishedHandState`` versus ``_AbsentHandRuntime``, the registry serial,
    the calibration binding, the solver process's port -- reads those four
    containers.  Updating the labels alone is why a glove powered on after
    startup (or a Bluetooth dongle plugged in later) was shown as connected
    and then opened no hand at all.

    A hand that vanished is dropped just as it would be at startup: the four
    containers always describe one detection, never a merge of two.
    """

    fresh = detect_gloves(device_manager, registry, announce=True)
    fresh.adopt_into(present_sides, ports, serials, links)
    remember_auto_links(fresh.auto_detected, registry)
    return {side: side in present_sides for side in ("left", "right")}


def remember_auto_links(auto_detected: dict[str, str], registry: Path) -> None:
    """Persist hands found behind a link the registry does not know yet.

    What the probe just worked out, so the next launch resolves the link by
    serial instead of probing every unclaimed port again (up to two seconds
    each).
    """

    for side, serial in auto_detected.items():
        try:
            set_glove_serial(side, serial, registry)
        except Exception as exc:  # a read-only registry must not stop the run
            print(f"[WARN] could not remember {side} link: {exc}",
                  file=sys.stderr)
        else:
            print(f"{side} device: remembered {side} -> serial {serial}")


def _run_live_session(args) -> int | str:
    """Run one live-viewer session.

    Returns an int exit code, ``"selector"`` when the in-app "back to the
    calibration selector" button was pressed, or ``"calibrate"`` when the
    selector's Calibrate button was (the launcher then runs the calibration
    program, and afterwards re-opens the picker).
    """
    # Resolve each bound hand independently: a missing glove is skipped, not
    # fatal.  Only when no glove at all is detected do we abort.
    device_manager = DeviceManager(args.registry)
    detected = detect_gloves(device_manager, args.registry, announce=True)
    present_sides: list[str] = []
    ports: dict[str, str] = {}
    serials: dict[str, str] = {}
    links: dict[str, str] = {}
    detected.adopt_into(present_sides, ports, serials, links)
    if not present_sides:
        print("No STM32 glove connected; opening the calibration selector.", file=sys.stderr)
    else:
        print(f"Connected gloves: {', '.join(present_sides)}")
    remember_auto_links(detected.auto_detected, args.registry)

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
    # picker for every connected side.  The picker's Refresh button re-detects
    # USB presence, so a glove plugged in after startup can be bound here.
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
                return refresh_detected_hands(
                    device_manager, args.registry,
                    present_sides, ports, serials, links)

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
                # The selector's Calibrate button: this session is over, and
                # ``main()`` runs the calibration program before re-opening the
                # selector.  Without this the button's answer fell through as
                # whatever came next and the app just quit.
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

    # The calibration files are checked for presence here and for usability by
    # the hand process itself: ``open_glove`` validates the file and reports an
    # unusable one back as a fatal message, which the loop below turns into the
    # same RuntimeError a parent-side probe used to raise.
    runtimes: dict[str, object] = {}
    pose_postprocess = {}
    for side in ("left", "right"):
        if side not in present_sides:
            runtimes[side] = _AbsentHandRuntime(side)
            continue
        if not calib_paths[side].is_file():
            raise RuntimeError(
                f"Missing {side} calibration: {calib_paths[side]}")
        from sdk.calibration import prepare_viewer_calibration
        calib_paths[side] = prepare_viewer_calibration(calib_paths[side], side)
        # Profile resolution already validates the file and hand side. Only
        # metadata is needed here; the worker owns the solver configuration.
        shape = json.loads(
            calib_paths[side].read_text(encoding="utf-8-sig")
        ).get("param_shape_calib") or {}
        pose_postprocess[side] = (
            "sensor-motion-refit-v1" if "sensor_motion_refit" in shape
            else "gesture-retarget-v1" if "gesture_retarget" in shape
            else "legacy-contact")
        runtimes[side] = PublishedHandState(side)

    print(f"Left IMU calibration: {calib_paths['left'].resolve()}")
    print(f"Right IMU calibration: {calib_paths['right'].resolve()}")
    print("Pipeline: sdk.open_glove -> 21 solved keypoints + finger force in "
          "newtons")

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

    # perf_counter (QPC) everywhere: time.monotonic ticks at ~15.6 ms on
    # Windows here, which quantized the preview gate to ~32 Hz.  1 ms timer
    # resolution so time.sleep() actually sleeps what it is asked (the Windows
    # default quantum is ~15.6 ms, which capped the main loop at ~30-60 Hz).
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
    stop_event = multiprocessing.Event()
    contact_control = multiprocessing.Value("d", 1.0)
    root_limits_control = multiprocessing.Value("b", True)
    # One reader process per hand, and one queue back each.  A hand that is not
    # connected has no runtime to feed and is never opened.
    readers: dict[str, multiprocessing.Process] = {}
    state_queues: dict[str, multiprocessing.Queue] = {}
    # The force matrix the panel is currently showing, per hand, so a repeat of
    # the same scan is not pushed again: the SDK re-attaches the newest scan to
    # every attitude frame, and the panel would otherwise be redrawn at the
    # attitude rate instead of the scan rate.
    last_force_versions = {side: -1 for side in present_sides}
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
        # One reader process per hand: the SDK pipeline owns the link and the
        # solve, and running it off the GUI's thread is what keeps a solve
        # burst from preempting the draw.  A process per hand is also what keeps
        # the two solvers off each other's GIL, and what stops a bursty link's
        # backlog from being shared between the hands.  Published state flows
        # back through one queue per hand, drained keep-newest by the main loop.
        for side in present_sides:
            # Live state is replaceable: cap IPC backlog so a slow GUI frame
            # cannot turn into visible pose latency on the following frames.
            state_queue = multiprocessing.Queue(maxsize=8)
            reader = multiprocessing.Process(
                target=hand_solver_process,
                args=(side, calib_paths, links, serials,
                      args.registry, solve_rate, state_queue, stop_event,
                      contact_control, root_limits_control),
                name=f"sdk-{side}",
                daemon=True,
            )
            state_queues[side] = state_queue
            readers[side] = reader
            reader.start()
        next_display = started
        display_interval = 1.0 / max(float(args.fps), 1.0)
        next_link_diag = started
        while True:
            now = time.perf_counter()
            if readers:
                fatal = None
                with state_lock:
                    for side in present_sides:
                        fatal = _drain_hand_queue(
                            state_queues[side], runtimes) or fatal
                if fatal is not None:
                    raise RuntimeError(
                        f"SDK {fatal.get('side') or ''} pipeline process "
                        f"failed: {fatal['error']}")
                for side in present_sides:
                    reader = readers[side]
                    if not reader.is_alive():
                        raise RuntimeError(
                            f"SDK {side} pipeline process exited unexpectedly "
                            f"(exitcode={reader.exitcode})")
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
                    # ``None`` on a Bluetooth link and on firmware older than
                    # v1.2.11, neither of which answers the probe; a dash says so
                    # rather than 0 ms.
                    latency_text = ("--" if latency is None
                                    else f"{float(latency):.2f}ms")
                    # English, like every other terminal line in this file; the
                    # on-screen HUD is what carries the bilingual labels.
                    # ``loss+`` is frames the device sent that never arrived, in
                    # the same whole-device unit as ``fram+``; ``dropped+`` is
                    # bytes the parser binned, and the two are unrelated.
                    print(
                        f"[{side}] FPS {snap[side]['fps']:5.1f}  "
                        f"IMU {snap[side]['device_fps']:5.1f} Hz  "
                        f"TACT {snap[side]['tactile_fps']:5.1f} Hz  "
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
                # The force panel follows the SDK's own scan counter, not the
                # attitude frames the scan rides on.
                for side in present_sides:
                    if snap[side]["force_version"] != last_force_versions[side]:
                        last_force_versions[side] = snap[side]["force_version"]
                        viewer.set_tactile_side(
                            side, _snap_force_or_none(snap, side),
                            _snap_adc_or_none(snap, side))
            ready = not present_sides or any(
                snap[side]["joints"] is not None
                for side in present_sides)
            # Freshness is judged per hand.  An ``all()`` over both hands meant
            # one hand stalling past ``--max_frame_age`` froze the other with
            # it, which is what made a single bad link look like a hung
            # application.  On a Bluetooth link that is the common case rather
            # than the exception: the two dongles share the 2.4 GHz band, so
            # one hand dropping past the age limit while the other runs on is
            # ordinary. A detected hand with no first frame must not block the
            # other hand's viewport. It stays NaN until it actually publishes;
            # recorder_loop independently requires every selected hand.
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
                    both_present = (len(present_sides) == 2 and all(
                        snap[side]["joints"] is not None for side in present_sides))
                    if both_present:
                        # Use the first frame on explicit relearn or a missing
                        # default, and persist it immediately.
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
                            "[WARN] Only one hand has data and no saved display "
                            "config; bimanual anchors derived from the connected "
                            "hand and not persisted.",
                            file=sys.stderr)
                fixed_rotations = fixed_bimanual_display_rotations(
                    initial, palm_target_bases)
                if not args.no_display:
                    viewer = BimanualViewer(
                        view_state, initial,
                        "Stouch Glove",
                        separation_m=separation_m,
                        palm_target_bases=palm_target_bases,
                        left_root_rotvec_sign=left_root_rotvec_sign,
                        right_root_rotvec_sign=right_root_rotvec_sign,
                        display_config_path=None,
                        present_sides=present_sides,
                        display_mode=args.display_mode,
                        stop_event=stop_event,
                        redraw_interval_s=1.0 / max(float(args.fps), 1.0))
                    def set_contacts(enabled, blend):
                        contact_control.value = blend if enabled else -1.0
                    def set_root_limits(enabled):
                        root_limits_control.value = bool(enabled)
                    viewer.bind_root_limits(set_root_limits)
                    viewer.bind_contact_controls(
                        set_contacts,
                        experimental='gesture-retarget-v1' in pose_postprocess.values(),
                        source_refit='sensor-motion-refit-v1' in pose_postprocess.values())
                startup_initialized = True
            if any_fresh and now >= next_display:
                if viewer:
                    statuses = {}
                    for side in ("left", "right"):
                        if side in present_sides:
                            statuses[side] = {
                                "fps": snap[side]["fps"],
                                "device_fps": snap[side]["device_fps"],
                                "tactile_fps": snap[side]["tactile_fps"],
                                "link_health": snap[side]["link_health"],
                                "safety": snap[side]["safety"],
                                "contact": snap[side]["contact"],
                                "warming_up": snap[side]["warming_up"],
                                "warmup_remaining_s": snap[side]["warmup_remaining_s"],
                                "warmup_completed": snap[side]["warmup_completed"],
                                "firmware_version": snap[side]["firmware_version"],
                                "latency": snap[side]["latency"],
                                "link": snap[side]["link_kind"],
                                "connected": True,
                                "waiting_for_data": snap[side]["joints"] is None,
                                # Abnormal, so the HUD shows it: this hand's
                                # pose above is the last good one, not this
                                # frame's.
                                "stale": side not in fresh_sides,
                            }
                        else:
                            statuses[side] = {
                                "fps": 0.0, "device_fps": 0.0,
                                "tactile_fps": 0.0,
                                "link_health": None,
                                "safety": False, "contact": None,
                                "warming_up": False,
                                "warmup_remaining_s": 0.0,
                                "warmup_completed": False,
                                "firmware_version": None,
                                "latency": None,
                                "link": "usb",
                                "connected": False,
                                "stale": False,
                            }
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
                        processed_frame_index, statuses,
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
                if viewer.consume_selector_request():
                    session_result = "selector"
                    break
                if not keep_running:
                    break
            # Deadline-aware pacing: sleep only until the next display slot
            # (clamped 1-10 ms) instead of a fixed 10 ms.  When the loop runs
            # behind the gate the fixed sleep was pure overhead; the floor
            # avoids busy-waiting when there is nothing due.
            time.sleep(max(
                0.001, min(0.010, next_display - time.perf_counter())))
    finally:
        stop_event.set()
        # Worker joins can take seconds while serial readers/queue feeders
        # stop. Close the window first: leaving it visible without pumping Qt
        # makes an orderly session exit look like a frozen live viewport.
        if viewer:
            viewer.close()
        for reader in readers.values():
            reader.join(timeout=5.0)
            if reader.is_alive():
                reader.terminate()
                reader.join(timeout=1.0)
        for session_queue in state_queues.values():
            session_queue.cancel_join_thread()
            session_queue.close()
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
        # Calibrate -> run the calibration program -> loop back to the
        # selector; the in-app "back to selector" button loops back into a
        # fresh live session (which re-opens the selector).
        result = _run_live_session(args)
        if result == "calibrate":
            result = _run_calibration_session(args)
            if result == "done":
                continue
            return result
        if result == "selector":
            args.select_calibration = True
            args.no_select_calibration = False
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
