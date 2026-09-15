#!/usr/bin/env python3
"""Qt (PySide6) GUI for HAND2mm left/right 16-IMU calibration, zh/en switchable.

Replaces the former Dear PyGui window.  PySide6 classes are imported at module
level (they are inert until a QApplication exists), but no QApplication is
created at import time, so headless imports stay safe.  The public entry point
remains ``main(argv) -> int``, so no launcher change is needed.

The Qt window reuses the existing HAND calibration engine through
``CalibrationController`` (a plain ``threading.Thread`` per background job, no
QThread), and keeps producing the same calibration JSON payload via
``ImuCalibrationCLI.build_payload()``, so ``select_calibration_files`` in the
live viewer still reads it unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QVBoxLayout,
    QWidget,
)


def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")


if getattr(sys, "frozen", False):
    BUNDLE_ROOT = Path(sys._MEIPASS)
    # Device bindings and generated calibration JSON must survive one-file
    # extraction cleanup, so all mutable files live beside the exe.
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    BUNDLE_ROOT = PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from common.enums import (  # noqa: E402
    MoCapCalibrateInstallationType,
    MoCapCalibrateRootType,
    MoCapCalibrateShapeType,
)
from glove_io.device_registry import (  # noqa: E402
    GloveDeviceError,
    clear_glove_bindings as _clear_registry_bindings,
    list_matching_ports,
    load_device_registry,
    resolve_glove,
    set_glove_serial,
    swap_glove_serials,
)
from common.usb_cdc import (  # noqa: E402
    firmware_hand_side,
)
from common.i18n import (  # noqa: E402
    L, get_lang, is_en, set_lang,
)
from algorithm.imu_calibrate_cli import (  # noqa: E402
    ImuCalibrationCLI,
    compute_motion_rms,
    default_out_path,
)
from gui import APP_VERSION  # noqa: E402
from gui.calibration_pose_preview import (  # noqa: E402
    CalibrationPosePreview,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)
from runtime import get_version  # noqa: E402


@dataclass(frozen=True)
class CalibrationStep:
    title: str
    instruction: str
    hold: str
    action: str
    flow: str
    trigger: object


def _yaw_step(side: str) -> CalibrationStep:
    """Root yaw step: the turn direction depends on the hand side.

    LEFT 手"水平右转"即指尖朝身体中线；RIGHT 手需"水平左转"才同样指尖朝内，
    两手指尖朝内、掌心相对。ROLL 翻掌方向由校准器按侧自动镜像，故文案不分侧。
    """
    if str(side).lower() == "right":
        title = L("指根 - 水平左转", "Root - Yaw Left")
        instruction = L(
            "回到水平手姿，整只手向左转 90°，使指尖指向身体中线。",
            "Return to a level hand, then turn the whole hand 90 degrees "
            "to the left so the fingertips point toward your body's midline.")
    else:
        title = L("指根 - 水平右转", "Root - Yaw Right")
        instruction = L(
            "回到水平手姿，整只手向右转 90°，使指尖指向身体中线。",
            "Return to a level hand, then turn the whole hand 90 degrees "
            "to the right so the fingertips point toward your body's midline.")
    return CalibrationStep(
        title, instruction,
        L("保持掌心水平。俯视时转向身体中线。",
          "Keep the palm horizontal. Viewed from above, the turn is toward "
          "the body's midline."),
        L("采集姿势", "Capture Pose"), "root", MoCapCalibrateRootType.YAW)


def calibration_steps(side: str) -> tuple[CalibrationStep, ...]:
    """Guided calibration steps; the root yaw step is per-side (palm-facing grip)."""
    yaw = _yaw_step(side)
    return (
        CalibrationStep(
            L("指根 - 水平", "Root - Level"),
            L("手背朝上平放，与地面平行。",
              "Lay the hand flat with the back of the hand facing up, "
              "parallel to the ground."),
            L("保持手完全静止，程序将平均 30 帧 IMU 数据。",
              "Keep the hand completely still while 30 IMU frames are averaged."),
            L("采集姿势", "Capture Pose"), "root", MoCapCalibrateRootType.HORIZONTAL),
        CalibrationStep(
            L("指根 - 指尖朝上", "Root - Fingertips Up"),
            L("保持手掌平放，整只手旋转直到指尖竖直朝上。",
              "Keep the hand flat and rotate the whole hand until the "
              "fingertips point straight up."),
            L("保持相同手型，只改变整只手的朝向。",
              "Keep the same hand shape. Change only the whole-hand orientation."),
            L("采集姿势", "Capture Pose"), "root", MoCapCalibrateRootType.VERTICAL_UP),
        CalibrationStep(
            L("指根 - 指尖朝下", "Root - Fingertips Down"),
            L("保持手掌平放，整只手旋转直到指尖竖直朝下。",
              "Keep the hand flat and rotate the whole hand until the "
              "fingertips point straight down."),
            L("保持相同手型并完全静止。",
              "Keep the same hand shape and hold completely still."),
            L("采集姿势", "Capture Pose"), "root", MoCapCalibrateRootType.VERTICAL_DOWN),
        CalibrationStep(
            L("指根 - 侧翻（拇指朝上）", "Root - Roll (Thumb Up)"),
            L("手平放，绕手腕到指尖的轴旋转 90°，使拇指朝上。",
              "Lay the hand flat, then turn the whole hand 90 degrees about "
              "the wrist-to-fingertip axis so the thumb points up."),
            L("左右手物理翻转方向相反，模型会自动镜像。",
              "Left and right hands roll in opposite physical directions; "
              "the model mirrors automatically."),
            L("采集姿势", "Capture Pose"), "root", MoCapCalibrateRootType.ROLL),
        yaw,
        CalibrationStep(
            L("计算指根标定", "Compute Root Calibration"),
            L("此计算无需任何姿势。", "No pose is required for this calculation."),
            L("已采集的五个朝向用于定义腕部参考坐标系。",
              "The five captured orientations define the wrist reference frame."),
            L("计算", "Compute"), "root", MoCapCalibrateRootType.CALC),
        CalibrationStep(
            L("安装 - 张开手掌", "Installation - Open Hand"),
            L("掌心向下，五指自然张开。",
              "Lay the palm down and spread all five fingers naturally."),
            L("保持每根手指静止，以采集固定的 IMU 安装偏移。",
              "Keep every finger still while the fixed IMU mounting offsets "
              "are captured."),
            L("采集姿势", "Capture Pose"), "installation",
            MoCapCalibrateInstallationType.POSE_0),
        CalibrationStep(
            L("计算安装标定", "Compute Installation Calibration"),
            L("此计算无需任何姿势。", "No pose is required for this calculation."),
            L("开掌采集用于确定每个 IMU 的固定安装偏移。",
              "The open-hand capture determines the fixed mounting offset of "
              "each IMU."),
            L("计算", "Compute"), "installation", MoCapCalibrateInstallationType.CALC),
        CalibrationStep(
            L("接触 - 拇指对食指", "Contact - Thumb to Index"),
            L("拇指指尖轻触食指指尖。",
              "Touch the thumb tip lightly to the index fingertip."),
            L("采集过程中不要按压、交叉或移动手指。",
              "Do not press, cross, or move the fingers during the capture."),
            L("采集接触", "Capture Contact"), "shape", MoCapCalibrateShapeType.INDEX),
        CalibrationStep(
            L("接触 - 拇指对中指", "Contact - Thumb to Middle"),
            L("拇指指尖轻触中指指尖。",
              "Touch the thumb tip lightly to the middle fingertip."),
            L("放松其他手指并保持接触完全静止。",
              "Relax the other fingers and hold the contact completely still."),
            L("采集接触", "Capture Contact"), "shape", MoCapCalibrateShapeType.MIDDLE),
        CalibrationStep(
            L("接触 - 拇指对无名指", "Contact - Thumb to Ring"),
            L("拇指指尖轻触无名指指尖。",
              "Touch the thumb tip lightly to the ring fingertip."),
            L("不要交叉手指，保持接触完全静止。",
              "Do not cross the fingers. Hold the contact completely still."),
            L("采集接触", "Capture Contact"), "shape", MoCapCalibrateShapeType.RING),
        CalibrationStep(
            L("接触 - 拇指对小指", "Contact - Thumb to Little"),
            L("拇指指尖轻触小指指尖。",
              "Touch the thumb tip lightly to the little fingertip."),
            L("保持轻触并让手保持静止。",
              "Maintain light fingertip contact and keep the hand still."),
            L("采集接触", "Capture Contact"), "shape", MoCapCalibrateShapeType.LITTLE),
        CalibrationStep(
            L("计算精细标定", "Compute Fine Calibration"),
            L("此计算无需任何姿势。", "No pose is required for this calculation."),
            L("四次指尖接触被拟合为 IMU 驱动的位置约束。",
              "The four fingertip contacts are fitted as IMU-driven position "
              "constraints."),
            L("计算", "Compute"), "shape", MoCapCalibrateShapeType.CALC),
    )


# Root turns in guided order.  Each entry maps a capture trigger to the hand
# state machine's stored pose attribute and ready-flag key, plus the short
# label used by the live angle readout.
_ROOT_POSE_READOUT = (
    (MoCapCalibrateRootType.VERTICAL_UP, "vertical_up", "Up"),
    (MoCapCalibrateRootType.VERTICAL_DOWN, "vertical_down", "Down"),
    (MoCapCalibrateRootType.ROLL, "roll", "Roll"),
    (MoCapCalibrateRootType.YAW, "yaw", "Yaw"),
)

# Solver bands, mirrored in the readout colors.  Raw2MANOCalibrator skips a
# root pose rotated less than 20 deg or more than 160 deg from the Level
# capture, warns beyond 30 deg from the 90 deg target, and requires at least
# three valid poses at CALC time.
_ROOT_ANGLE_SKIP_LO_DEG = 20.0
_ROOT_ANGLE_SKIP_HI_DEG = 160.0
_ROOT_ANGLE_TARGET_DEG = 90.0
_ROOT_ANGLE_WARN_OFF_DEG = 30.0
_ROOT_ANGLE_GOOD_BAND_DEG = 15.0

# Auto-capture: a step must hold its "ready" conditions (motion RMS below the
# threshold and, for root poses, the live angle inside the good band) for this
# long before the capture fires, so a brief accidental pass cannot trigger it.
_AUTO_STABLE_DURATION_S = 1.0
# A failed compute step is retried on this cooldown (not every tick) so a
# rejected solver run cannot spin into a tight auto-capture loop.
_AUTO_COMPUTE_RETRY_S = 2.0
# A capture step needs at least this many buffered frames before the motion
# RMS reading is trustworthy (compute_motion_rms returns 0.0 while the buffer
# warms up, which would otherwise look like a perfectly still hand).
_AUTO_MIN_FRAMES = 30

_ANGLE_COLORS = {
    "ok": (105, 220, 120),
    "weak": (235, 200, 90),
    "off": (240, 150, 80),
    "skip": (255, 95, 95),
    "muted": (145, 170, 195),
}


def _root_angle_labels() -> dict[str, str]:
    return {
        "vertical_up": L("上", "Up"),
        "vertical_down": L("下", "Down"),
        "roll": L("侧翻", "Roll"),
        "yaw": L("偏航", "Yaw"),
    }


def format_root_angle(snapshot: dict) -> tuple[str, tuple[int, int, int], str]:
    """Turn a root-angle snapshot into (readout, color, history) for the panel."""
    muted = _ANGLE_COLORS["muted"]
    labels = _root_angle_labels()
    captured_parts = [
        f"{labels[name]} {snapshot['captured'][name]:.0f}"
        if snapshot["captured"][name] is not None
        else f"{labels[name]} --"
        for name in ("vertical_up", "vertical_down", "roll", "yaw")
    ]
    history = L("已采集：", "Captured: ") + "  |  ".join(captured_parts)

    if not snapshot["base_ready"]:
        return (
            L("请先采集水平姿势；它将作为实时角度读数的 0° 基准。",
              "Capture the Level pose first; it becomes the 0 deg reference "
              "for the live angle readout."),
            muted, history)

    if snapshot["is_calc"]:
        values = [v for v in snapshot["captured"].values() if v is not None]
        valid = [
            v for v in values
            if _ROOT_ANGLE_SKIP_LO_DEG < v < _ROOT_ANGLE_SKIP_HI_DEG]
        off_target = [
            v for v in valid
            if abs(v - _ROOT_ANGLE_TARGET_DEG) > _ROOT_ANGLE_WARN_OFF_DEG]
        if len(valid) < 3:
            color, verdict = _ANGLE_COLORS["skip"], (
                L("仅 {n}/{m} 个姿势位于 {lo:.0f}-{hi:.0f}° 窗口内；至少需要 3 个。",
                  "only {n} of {m} poses are inside the {lo:.0f}-{hi:.0f} deg "
                  "window; at least 3 are required").format(
                      n=len(valid), m=len(values),
                      lo=_ROOT_ANGLE_SKIP_LO_DEG, hi=_ROOT_ANGLE_SKIP_HI_DEG))
        elif off_target:
            color, verdict = _ANGLE_COLORS["off"], (
                L("{n} 个姿势偏离 {target:.0f}° 超过 {off:.0f}°；结果可通过但精度下降（含警告）。",
                  "{n} pose(s) deviate more than {off:.0f} deg from "
                  "{target:.0f} deg; the result passes with a warning but "
                  "accuracy is reduced").format(
                      n=len(off_target), off=_ROOT_ANGLE_WARN_OFF_DEG,
                      target=_ROOT_ANGLE_TARGET_DEG))
        else:
            color, verdict = _ANGLE_COLORS["ok"], (
                L("所有已采集姿势都接近 {target:.0f}°。",
                  "all captured poses are near {target:.0f} deg").format(
                      target=_ROOT_ANGLE_TARGET_DEG))
        return (L("计算前请复核：{verdict}",
                  "Review before computing: {verdict}.").format(verdict=verdict),
                color, history)

    live = snapshot["live_angle_deg"]
    if live is None:
        return (
            L("腕部 IMU 信号丢失。请检查手套连接并保持静止。",
              "Wrist IMU signal lost. Check the glove connection and hold still."),
            _ANGLE_COLORS["skip"], history)

    if snapshot["target_deg"] == 0.0:
        # Re-capturing Level: the stored base is the original capture.
        if live < 10.0:
            color, verdict = _ANGLE_COLORS["ok"], (
                L("接近原始水平采集", "close to the original Level capture"))
        elif live < 30.0:
            color, verdict = _ANGLE_COLORS["weak"], (
                L("与原始水平采集略有偏差；仅在有意修正时重新采集",
                  "somewhat off the original Level capture; re-capture only "
                  "when correcting it intentionally"))
        else:
            color, verdict = _ANGLE_COLORS["skip"], (
                L("与原始水平采集偏差较大；重新采集将替换 0° 基准",
                  "far from the original Level capture; re-capturing replaces "
                  "the 0 deg reference"))
        return (L("相对水平的实时旋转：{live:.0f}° - {verdict}。",
                  "Live rotation from Level: {live:.0f} deg - {verdict}.").format(
                      live=live, verdict=verdict), color, history)

    if live < _ROOT_ANGLE_SKIP_LO_DEG or live > _ROOT_ANGLE_SKIP_HI_DEG:
        color, verdict = _ANGLE_COLORS["skip"], (
            L("超出 {lo:.0f}-{hi:.0f}°；求解器将跳过此姿势",
              "outside {lo:.0f}-{hi:.0f} deg; the solver would SKIP this pose")
            .format(lo=_ROOT_ANGLE_SKIP_LO_DEG, hi=_ROOT_ANGLE_SKIP_HI_DEG))
    elif abs(live - _ROOT_ANGLE_TARGET_DEG) > _ROOT_ANGLE_WARN_OFF_DEG:
        color, verdict = _ANGLE_COLORS["off"], (
            L("偏离目标超过 {off:.0f}°；精度警告",
              "more than {off:.0f} deg from the target; accuracy warning")
            .format(off=_ROOT_ANGLE_WARN_OFF_DEG))
    elif abs(live - _ROOT_ANGLE_TARGET_DEG) > _ROOT_ANGLE_GOOD_BAND_DEG:
        color, verdict = _ANGLE_COLORS["weak"], (
            L("略微偏离目标；请靠近 {target:.0f}°",
              "slightly off target; move closer to {target:.0f} deg")
            .format(target=_ROOT_ANGLE_TARGET_DEG))
    else:
        color, verdict = _ANGLE_COLORS["ok"], L("处于良好区间", "in the good band")
    return (
        L("相对水平的实时旋转：{live:.0f}°（目标 {target:.0f}°）- {verdict}。",
          "Live rotation from Level: {live:.0f} deg "
          "(target {target:.0f} deg) - {verdict}.").format(
              live=live, target=_ROOT_ANGLE_TARGET_DEG, verdict=verdict),
        color, history)


def contains_han(text: str) -> bool:
    return any("一" <= character <= "鿿" for character in str(text))


def validate_hand2mm_runtime(path: Path | None = None) -> dict:
    # The HAND2mm local runtime is hardcoded to hand_pinky_plus_2mm
    # (see algorithm/protected/pc/runtime_backend.py); external JSON is no longer read.
    return {
        "schema": "stm32-imu-usb-local-runtime-override",
        "schema_version": 1,
        "local_only": True,
        "backend": "hand_pinky_plus_2mm",
        "pinky_extra_length_mm": 2.0,
    }


def detected_glove_options() -> list[tuple[str, str]]:
    """Return display labels and USB serials for connected STM32 gloves."""
    options = []
    for port in list_matching_ports():
        serial_number = str(port.serial_number or "").strip().upper()
        if not serial_number:
            continue
        label = f"{serial_number} | {port.device} | {port.location or 'unknown location'}"
        options.append((label, serial_number))
    return options


def detected_glove_port(serial_number: str) -> str | None:
    """Return the current COM port for a detected USB serial number.

    Calibration used to resolve the port only through ``glove_devices.json``.
    That is intentionally persistent, but it can be stale after replacing or
    reflashing a board.  The device combo already shows the live serial list,
    so the Connect action can safely use the user's selected device without
    silently rewriting the left/right binding.
    """

    wanted = str(serial_number or "").strip().upper()
    if not wanted:
        return None
    for port in list_matching_ports():
        if str(port.serial_number or "").strip().upper() == wanted:
            return str(port.device)
    return None


def detect_hand_pair(registry: Path) -> dict[str, str] | None:
    """Return ``{"left": port, "right": port}`` when a full left+right pair is
    bound and connected, else ``None``.

    Each side resolves through ``resolve_glove`` (registry serial -> COM port);
    a missing, unbound, or ambiguous side raises ``GloveDeviceError``, which
    means the pair is incomplete and dual-hand calibration must stay disabled.
    """
    ports: dict[str, str] = {}
    for side in ("left", "right"):
        try:
            port, _serial = resolve_glove(side, registry)
        except GloveDeviceError:
            return None
        ports[side] = port
    return ports


def bind_glove_serial(side: str, serial_number: str, registry: Path) -> Path:
    return set_glove_serial(side, serial_number, registry)


def glove_serial_bindings(registry: Path) -> dict[str, list[str]]:
    payload = load_device_registry(registry)
    return {side: payload[side]["usb_serials"] for side in ("left", "right")}


def swap_glove_bindings(registry: Path) -> Path:
    return swap_glove_serials(registry)


def clear_glove_bindings(registry: Path) -> Path:
    return _clear_registry_bindings(registry)


def format_glove_bindings(bindings: dict[str, list[str]]) -> str:
    return ", ".join(
        f"{side}={','.join(serials) if serials else '-'}"
        for side, serials in bindings.items())


def write_calibration_output(path: Path, payload: dict, force: bool = False) -> Path:
    output = Path(path).resolve()
    if output.exists() and not force:
        raise FileExistsError(
            "The selected output file already exists. Choose another path or enable replacement.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return output


def calibration_output_with_name(current_output: Path, filename: str) -> Path:
    value = str(filename or "").strip()
    if not value:
        raise ValueError(L("JSON 文件名不能为空。", "JSON file name must not be empty."))
    if "/" in value or "\\" in value or Path(value).name != value:
        raise ValueError(L("JSON 文件名不能包含目录路径。",
                           "JSON file name must not contain a directory path."))
    if not value.lower().endswith(".json"):
        value += ".json"
    if value in {".json", "..json"}:
        raise ValueError(L("JSON 文件名无效。", "JSON file name is invalid."))
    return Path(current_output).parent / value


class CalibrationController:
    """Own the existing HAND calibration engine while the Qt window stays responsive.

    User-facing strings are stored as ``(zh, en)`` pairs and resolved via
    ``L(*pair)`` in ``snapshot()`` so an in-flight status follows a language
    switch immediately instead of leaving stale text behind.
    """

    def __init__(self, args):
        self.args = args
        self.side = args.side
        # 固件版本号识别出的固定手型；None 表示未锁定（用户可自由选）。
        self.locked_side: str | None = None
        # 单手套连接时若版本号手型与所选不一致，自动纠正；双手套模式关闭。
        self.auto_correct_side = True
        self._steps = calibration_steps(self.side)
        self.backend: ImuCalibrationCLI | None = None
        self.output = self._default_output(args.side)
        self.connected = False
        self.session_started = False
        self.current_step = 0
        self.completed_steps: set[int] = set()
        self.busy = False
        self.saved_path: Path | None = None
        self._status = (
            "选择手型，然后连接对应的 STM32 手套。",
            "Select the hand, then connect the corresponding STM32 glove.")
        self._error = ("", "")
        self._device_text = ("未连接", "Not connected")
        self._imu_text = ("IMU 就绪：0 / 16", "IMUs ready: 0 / 16")
        self._motion_text = ("运动 RMS：--", "Motion RMS: --")
        self.stream_fps = 0.0
        self.auto_capture = True
        self.rms_threshold = 0.5
        self._total_rms = 0.0
        self._worst_rms = 0.0
        self._auto_stable_since: float | None = None
        self._auto_last_step = -1
        self._auto_status = ("", "")
        self.replace_output = bool(args.force)
        self._worker: threading.Thread | None = None
        self._lock = threading.RLock()
        self._health_time = time.monotonic()
        self._health_count = 0

    @property
    def step(self) -> CalibrationStep:
        return self._steps[self.current_step]

    @property
    def progress(self) -> float:
        return len(self.completed_steps) / len(self._steps)

    @property
    def complete(self) -> bool:
        return len(self.completed_steps) == len(self._steps)

    def _default_output(self, side: str) -> Path:
        if self.args.out:
            return Path(self.args.out)
        default = default_out_path(side)
        output_dir = getattr(self.args, "output_dir", None)
        if output_dir:
            return Path(output_dir) / default.name
        return default

    def _backend_args(self, serial_port: str | None = None,
                      side: str | None = None):
        side = side or self.side
        config = self.args.config
        if config is None:
            config = PROJECT_ROOT / "config" / (
                "config_left.json" if side == "left" else "config.json")
        return argparse.Namespace(
            side=side,
            config=Path(config),
            registry=Path(self.args.registry),
            # A live device selected in the GUI takes precedence over the
            # persisted registry.  ``None`` retains the normal registry
            # lookup for command-line and non-GUI callers.
            serial_port=serial_port or self.args.serial_port,
            fps=self.args.fps,
            startup_timeout=self.args.startup_timeout,
            out=self.output,
            force=self.replace_output,
            check_stream=False,
            flow="all",
            # 连接阶段跳过 BNO055 冷启动静止稳定：GUI 的自动采集在每次采集
            # 前已有更严的 RMS 静止门控，连接只需串口 + 16 路 IMU 就绪。
            skip_settle=True,
        )

    def _boot_backend(self, serial_port: str | None = None) -> ImuCalibrationCLI:
        """Boot a backend, auto-correcting the hand side from the firmware version.

        The firmware encodes the hand side in a two-digit patch (1.2.91 = left,
        1.2.92 = right).  When the detected side disagrees with the selected
        one, single-hand mode reboots with the correct side; dual-hand mode
        raises so a swapped pair is surfaced rather than silently corrected.
        """
        side = self.side
        while True:
            backend = ImuCalibrationCLI(self._backend_args(serial_port, side))
            try:
                backend.boot()
            except Exception as exc:
                backend.shutdown()
                detail = str(exc).strip() or exc.__class__.__name__
                raise RuntimeError(L(
                    "无法启动健康的 16-IMU 数据流。请检查 USB 连接、设备手型分配与串口可用性。详情：{detail}",
                    "Could not start a healthy 16-IMU stream. Check USB connection, "
                    "device side assignment, and serial-port availability. "
                    "Details: {detail}").format(detail=detail)) from exc
            detected = firmware_hand_side(
                getattr(getattr(backend, "hand", None), "firmware_version", None)
                or "")
            with self._lock:
                if detected is not None and detected != side:
                    backend.shutdown()
                    if not self.auto_correct_side:
                        side_zh = "左" if detected == "left" else "右"
                        raise RuntimeError(L(
                            "该位置的手套固件识别为{side}手，与已分配的左右位置不符。请检查两只手套是否插反。",
                            "This glove's firmware identifies as the {side} hand, "
                            "which does not match its assigned side. Check whether "
                            "the two gloves are swapped.").format(side=side_zh))
                    # 用户选错手型：换成固件手型后重连一次。
                    side = detected
                    self.side = detected
                    self._steps = calibration_steps(detected)
                    self.output = self._default_output(detected)
                    self.locked_side = detected
                    continue
                self.locked_side = detected  # None 或识别出的手型
                return backend

    def snapshot(self) -> dict:
        with self._lock:
            backend = self.backend
            firmware_version = (
                getattr(getattr(backend, "hand", None), "firmware_version", None)
                if backend is not None else None)
            return {
                "side": self.side,
                "side_locked": self.locked_side is not None,
                "locked_side": self.locked_side,
                "output": str(self.output.resolve()),
                "connected": self.connected,
                "session_started": self.session_started,
                "current_step": self.current_step,
                "completed_steps": set(self.completed_steps),
                "busy": self.busy,
                "complete": self.complete,
                "progress": self.progress,
                "status": L(*self._status),
                "error": L(*self._error),
                "device_text": L(*self._device_text),
                "imu_text": L(*self._imu_text),
                "motion_text": L(*self._motion_text),
                "stream_fps": self.stream_fps,
                "firmware_version": firmware_version,
                "saved_path": str(self.saved_path) if self.saved_path else "",
                "auto_capture": self.auto_capture,
                "rms_threshold": self.rms_threshold,
                "auto_status": L(*self._auto_status),
            }

    def retranslate(self) -> None:
        """Rebuild the step list in the newly selected language."""
        with self._lock:
            self._steps = calibration_steps(self.side)

    def _start_worker(self, target, name: str) -> bool:
        with self._lock:
            if self.busy:
                return False
            self.busy = True
            self._error = ("", "")

        def run():
            try:
                target()
            except BaseException as exc:  # Keep background errors inside the GUI.
                message = str(exc).strip() or exc.__class__.__name__
                if contains_han(message):
                    # Keep the real (possibly Chinese) detail visible instead of
                    # hiding it behind the generic text, so failures are debuggable.
                    message = L(
                        "操作失败。请检查手套连接与姿势后重试。\n详情：{detail}",
                        "The operation failed. Check the glove connection and "
                        "pose, then retry.\nDetail: {detail}").format(detail=message)
                with self._lock:
                    self._error = (message, message)
                    self._status = ("操作失败。", "Operation failed.")
            finally:
                with self._lock:
                    self.busy = False

        self._worker = threading.Thread(target=run, name=name, daemon=True)
        self._worker.start()
        return True

    def select_side(self, side: str) -> None:
        normalized = str(side).strip().lower()
        if normalized not in ("left", "right"):
            raise ValueError("Hand side must be left or right.")
        with self._lock:
            if self.busy:
                return
            if normalized == self.side:
                return
        # Switching hands while a glove is connected tears the old backend
        # down and resets the session first; otherwise the combo shows the
        # new side while the stream still belongs to the old hand.
        switched = self.reset_session(
            ("已切换手型。请连接所选手套以开始。",
             "Switched hand. Connect the selected glove to begin."))
        with self._lock:
            self.side = normalized
            self._steps = calibration_steps(normalized)
            self.output = self._default_output(normalized)
            self._status = (
                ("已切换手型。请连接所选手套以开始。",
                 "Switched hand. Connect the selected glove to begin.")
                if switched else
                ("请连接所选手套以开始。", "Connect the selected glove to begin."))
            self._error = ("", "")

    def reset_session(self, status: tuple[str, str] | None = None) -> bool:
        """Tear down a connected glove and reset the calibration session.

        Returns True when a connected backend was shut down. Does not touch
        the status text if nothing was connected. Safe to call from the GUI
        thread at any time; shutdown runs after the lock is released.
        """
        with self._lock:
            if self.busy:
                return False
            backend = self.backend
            was_connected = backend is not None or self.connected
            self.backend = None
            self.connected = False
            self.session_started = False
            self.completed_steps.clear()
            self.current_step = 0
            self.saved_path = None
            self.locked_side = None
            self._device_text = ("未连接", "Not connected")
            self._imu_text = ("IMU 就绪：0 / 16", "IMUs ready: 0 / 16")
            self._motion_text = ("运动 RMS：--", "Motion RMS: --")
            self.stream_fps = 0.0
            if was_connected:
                self._status = status or (
                    "请连接所选手套以开始。", "Connect the selected glove to begin.")
            self._error = ("", "")
        if backend is not None:
            backend.shutdown()
        return was_connected

    def set_output_filename(self, filename: str) -> Path:
        with self._lock:
            if self.busy:
                raise RuntimeError(L("请等待当前操作完成。",
                                     "Wait for the current operation to finish."))
            self.output = calibration_output_with_name(self.output, filename)
            self._error = ("", "")
            return self.output

    def report_error(self, message: str) -> None:
        with self._lock:
            self._error = (str(message), str(message))
            self._status = ("操作失败。", "Operation failed.")

    def connect(self, serial_port: str | None = None) -> bool:
        def work():
            with self._lock:
                self._status = (
                    "正在连接并等待全部 16 个 IMU…",
                    "Connecting and waiting for all 16 IMUs...")
            backend = self._boot_backend(serial_port)
            with self._lock:
                self.backend = backend
                self.connected = True
                self.session_started = False
                self.completed_steps.clear()
                self.current_step = 0
                port = backend.hand_config.serial_port or "automatic USB discovery"
                side_zh = "左" if self.side == "left" else "右"
                side_en = "Left" if self.side == "left" else "Right"
                self._device_text = (
                    f"{side_zh}手套 | {port}", f"{side_en} glove | {port}")
                self._status = (
                    "全部 16 个 IMU 已就绪。开始新的标定会话。",
                    "All 16 IMUs are ready. Start a new calibration session.")
                self._health_time = time.monotonic()
                self._health_count = backend.hand.buffer.get_counter()
        return self._start_worker(work, "hand2mm-calibration-connect")

    def start_new_session(self) -> bool:
        def work():
            backend = self.backend
            if backend is None:
                raise RuntimeError(L("开始标定前请先连接手套。",
                                     "Connect a glove before starting calibration."))
            # Root reset already invalidates installation and contact data.
            # Calling the downstream reset commands after this would be
            # rejected because the root calibrator intentionally no longer
            # exists.
            backend.hand.update_root_calibrator(MoCapCalibrateRootType.RESET)
            with self._lock:
                self.session_started = True
                self.completed_steps.clear()
                self.current_step = 0
                self.saved_path = None
                self._status = (
                    "新标定会话已开始。请采集水平姿势。",
                    "New calibration session started. Capture the level pose.")
        return self._start_worker(work, "hand2mm-calibration-reset")

    def _execute_step(self, step_index: int) -> None:
        """Execute one step synchronously.

        Single-hand mode wraps this method in its normal worker.  The dual
        coordinator can call it directly from one coordinator worker so the
        Torch/MANO calculation steps never run concurrently in two threads.
        """
        step_index = int(step_index)
        step = self._steps[step_index]
        backend = self.backend
        if backend is None or not self.connected:
            raise RuntimeError(L("采集标定数据前请先连接手套。",
                                 "Connect a glove before capturing calibration data."))
        if not self.session_started:
            raise RuntimeError(L("请先开始新的标定会话。",
                                 "Start a new calibration session first."))
        if backend.hand.missing_imu_ids:
            raise RuntimeError(L(
                "标定继续前必须全部 16 个 IMU 就绪。",
                "All 16 IMUs must be ready before calibration can continue."))
        with self._lock:
            self._status = (
                "正在执行当前步骤，请按说明操作…",
                "Running the current step. Keep following the instruction...")
        before_revision = backend.hand.mano_state.calibration_result_revision
        if step.flow == "root":
            backend.hand.update_root_calibrator(step.trigger)
        elif step.flow == "installation":
            backend.hand.update_installation_calibrator(step.trigger)
        else:
            backend.hand.update_shape_calibrator(step.trigger)
        state = backend.hand.mano_state
        if state.calibration_result_revision <= before_revision:
            raise RuntimeError(L("标定命令未完成。",
                                 "The calibration command did not complete."))
        if not state.last_calibration_ok:
            message = state.last_calibration_message
            if contains_han(message):
                message = L("标定被拒绝。请保持所需姿势静止并重试。",
                            "Calibration was rejected. Hold the required pose "
                            "still and retry.")
            raise RuntimeError(message)
        with self._lock:
            self.completed_steps.add(step_index)
            if step_index + 1 < len(self._steps):
                self.current_step = step_index + 1
                self._status = (
                    "本步骤已完成。请继续下一步。",
                    "Step completed. Continue to the next step.")
            else:
                self._status = (
                    "IMU 标定已完成。请保存标定文件。",
                    "IMU calibration completed. Save the calibration file.")

    def capture_current_step(self) -> bool:
        step_index = self.current_step
        return self._start_worker(
            lambda: self._execute_step(step_index),
            f"hand2mm-calibration-step-{step_index}")

    def go_to_step(self, index: int) -> None:
        with self._lock:
            if self.busy or not self.session_started:
                return
            target = max(0, min(int(index), len(self._steps) - 1))
            # Recapturing an upstream pose invalidates every downstream GUI
            # completion marker. The calibration engine will rebuild those
            # dependent values as the user advances through the steps again.
            self.completed_steps = {
                completed for completed in self.completed_steps
                if completed < target
            }
            self.current_step = target
            self._error = ("", "")
            self._status = (
                "请查看姿势说明，然后执行此步骤。",
                "Review the pose instruction, then run this step.")

    def save(self) -> bool:
        def work():
            backend = self.backend
            if backend is None or not self.complete:
                raise RuntimeError(L("保存前请完成所有标定步骤。",
                                     "Complete every calibration step before saving."))
            payload = backend.build_payload()
            payload["tool"] = "imu_calibration_gui"
            path = write_calibration_output(
                self.output, payload, force=self.replace_output)
            with self._lock:
                self.saved_path = path
                self._status = (
                    f"标定保存成功：{path}", f"Calibration saved successfully: {path}")
        return self._start_worker(work, "hand2mm-calibration-save")

    def refresh_health(self) -> None:
        backend = self.backend
        if backend is None or not self.connected:
            return
        now = time.monotonic()
        if now - self._health_time < 0.5:
            return
        counter = backend.hand.buffer.get_counter()
        fps = (counter - self._health_count) / max(now - self._health_time, 1e-9)
        self._health_time = now
        self._health_count = counter
        missing = backend.hand.missing_imu_ids
        total_rms, worst_rms = compute_motion_rms(backend.hand.buffer, 30)
        with self._lock:
            self.stream_fps = fps
            self._total_rms = total_rms
            self._worst_rms = worst_rms
            ready = 16 - len(missing)
            if missing:
                self._imu_text = (
                    f"IMU 就绪：{ready} / 16 | 缺失：{', '.join(missing)}",
                    f"IMUs ready: {ready} / 16 | Missing: {', '.join(missing)}")
            else:
                self._imu_text = (
                    f"IMU 就绪：{ready} / 16", f"IMUs ready: {ready} / 16")
            self._motion_text = (
                f"运动 RMS：平均 {total_rms:.1f}° | 最差 {worst_rms:.1f}°",
                f"Motion RMS: {total_rms:.1f} deg average | "
                f"{worst_rms:.1f} deg worst IMU")

    def _is_compute_step(self, step: CalibrationStep) -> bool:
        return step.trigger in (
            MoCapCalibrateRootType.CALC,
            MoCapCalibrateInstallationType.CALC,
            MoCapCalibrateShapeType.CALC,
        )

    def _angle_ready(self, step: CalibrationStep) -> bool:
        """True when the live angle meets this step's pose requirement.

        Installation/shape steps have no angle target (stillness alone gates
        them).  For root steps the live angle is measured relative to the
        captured Level pose; on the very first (Level) capture there is no
        reference yet, so stillness alone gates it there too.
        """
        if step.flow != "root":
            return True
        snap = self.root_angle_snapshot()
        if snap is None:
            return False
        if not snap["base_ready"]:
            # First Level capture: the 0-degree reference is being captured
            # right now, so there is no live angle to compare against.
            return True
        live = snap["live_angle_deg"]
        if live is None:
            return False
        target = snap["target_deg"]
        if target == 0.0:
            return live < 10.0
        return abs(live - target) <= _ROOT_ANGLE_GOOD_BAND_DEG

    def auto_capture_readiness(self) -> tuple[bool, bool, bool]:
        """Return ``(warm, rms_ok, angle_ok)`` for the current step.

        A read-only readiness probe used both by this controller's own
        auto-capture and by the bimanual coordinator's joint gate.  ``warm``
        guards the buffer warm-up window (``compute_motion_rms`` reports 0.0
        before enough frames arrive, which would look like perfect stillness).
        """
        backend = self.backend
        warm = (backend is not None
                and backend.hand.buffer.get_counter() >= _AUTO_MIN_FRAMES)
        return (
            warm,
            self._total_rms < self.rms_threshold,
            self._angle_ready(self.step),
        )

    def auto_capture_tick(self) -> None:
        """Drive the hands-free capture flow; called from the GUI tick.

        Compute steps fire on arrival (with a retry cooldown).  Capture steps
        fire once the motion RMS stays below ``rms_threshold`` and, for root
        poses, the live angle stays in the good band for the stable duration.
        """
        if not self.auto_capture:
            self._auto_stable_since = None
            return
        if self.busy:
            self._auto_stable_since = None
            self._auto_status = (
                "自动采集：正在采集…", "Auto-capture: capturing...")
            return
        if not self.connected or not self.session_started or self.complete:
            self._auto_stable_since = None
            return
        if self.current_step != self._auto_last_step:
            self._auto_stable_since = None
            self._auto_last_step = self.current_step
        step = self.step

        if self._is_compute_step(step):
            now = time.monotonic()
            if (self._auto_stable_since is None
                    or now - self._auto_stable_since >= _AUTO_COMPUTE_RETRY_S):
                self._auto_stable_since = now
                self.capture_current_step()
                self._auto_status = (
                    "自动采集：正在执行计算步骤…",
                    "Auto-capture: running the compute step...")
            else:
                self._auto_status = (
                    "自动采集：计算步骤未完成，稍后自动重试…",
                    "Auto-capture: compute step pending; will retry shortly...")
            return

        backend = self.backend
        if backend is not None and backend.hand.buffer.get_counter() < _AUTO_MIN_FRAMES:
            self._auto_stable_since = None
            self._auto_status = (
                "自动采集：等待 IMU 数据…",
                "Auto-capture: waiting for IMU data...")
            return

        rms_ok = self._total_rms < self.rms_threshold
        angle_ok = self._angle_ready(step)
        if rms_ok and angle_ok:
            now = time.monotonic()
            if self._auto_stable_since is None:
                self._auto_stable_since = now
            if now - self._auto_stable_since >= _AUTO_STABLE_DURATION_S:
                self._auto_stable_since = None
                self.capture_current_step()
                return
            self._auto_status = (
                "自动采集：保持当前姿势…",
                "Auto-capture: hold the current pose...")
        else:
            self._auto_stable_since = None
            if not rms_ok:
                self._auto_status = (
                    f"自动采集：等待手部稳定（RMS 平均 {self._total_rms:.1f}°"
                    f" ≥ {self.rms_threshold:.1f}°）…",
                    f"Auto-capture: waiting for the hand to settle "
                    f"(RMS {self._total_rms:.1f} deg >= "
                    f"{self.rms_threshold:.1f} deg)...")
            else:
                self._auto_status = (
                    "自动采集：请靠近目标角度…",
                    "Auto-capture: move closer to the target angle...")

    def root_angle_snapshot(self) -> dict | None:
        """Live rotation data for the guided root steps.

        The angle shown is the exact quantity the root solver measures: the
        rotation magnitude between the current wrist IMU orientation and the
        captured Level (base) pose, ``|R_current * R_base^-1|``.  The solver
        fits the same relative rotations, so the readout reuses its numbers
        and its 20-160 deg skip window.

        Returns None when the readout should be hidden (no connected root
        step).  Safe to call from the GUI thread; the backend only swaps
        whole rotation references during captures.
        """
        backend = self.backend
        if backend is None or not self.connected or not self.session_started:
            return None
        step = self.step
        if step.flow != "root":
            return None
        state = backend.hand.mano_state
        base_ready = bool(state.calib_root_pose_ready.get("horizontal"))
        base = state.calib_root_horizontal_pose if base_ready else None

        live_angle = None
        if base is not None:
            buffer = backend.hand.buffer
            counter = buffer.get_counter()
            if counter > 0:
                frame, _, err = buffer.peek(counter - 1)
                if (err is None and frame is not None
                        and bool(frame.valid_mask[0])):
                    current = frame.imu_rotation[0]
                    live_angle = float(np.degrees(
                        (current * base[0].inv()).magnitude()))

        captured = {}
        for _trigger, name, _label in _ROOT_POSE_READOUT:
            stored = (
                getattr(state, f"calib_root_{name}_pose", None)
                if state.calib_root_pose_ready.get(name) else None)
            captured[name] = (
                None if stored is None or base is None
                else float(np.degrees((stored[0] * base[0].inv()).magnitude())))

        return {
            "base_ready": base_ready,
            "live_angle_deg": live_angle,
            "target_deg": (
                0.0 if step.trigger == MoCapCalibrateRootType.HORIZONTAL
                else _ROOT_ANGLE_TARGET_DEG),
            "is_calc": step.trigger == MoCapCalibrateRootType.CALC,
            "captured": captured,
        }

    def shutdown(self) -> None:
        backend = self.backend
        if backend is not None:
            backend.shutdown()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)


def _side_args(args, side: str):
    """Clone the CLI namespace for one side of a bimanual calibration.

    ``side`` drives config selection and payload side tagging through the same
    code path a single-hand controller uses.  ``out`` is cleared so the dual
    coordinator owns the (paired) output naming instead of a per-side ``--out``.
    """
    clone = argparse.Namespace(**vars(args))
    clone.side = side
    clone.out = None
    return clone


class BimanualCalibrationController:
    """Coordinate a matched left+right pair through the same calibration flow.

    Two independent ``CalibrationController`` instances each own one glove's
    backend (two ``ImuCalibrationCLI`` objects run safely side by side on two
    COM ports).  This coordinator keeps their step indices in lockstep and only
    fires a capture once *both* hands satisfy the motion-RMS and pose-angle
    requirements at the same time.
    """

    def __init__(self, args):
        self.args = args
        self.hands: dict[str, CalibrationController] = {
            side: CalibrationController(_side_args(args, side))
            for side in ("left", "right")
        }
        for hand in self.hands.values():
            hand.auto_capture = False  # the coordinator drives capture itself
            hand.auto_correct_side = False  # 位置插错应报错，而非自动纠正
        self.mode = "dual"
        self.side = "dual"
        self._steps = self.hands["left"]._steps
        self.auto_capture = True
        self.rms_threshold = 0.5
        self.replace_output = bool(args.force)
        self._output_dir = getattr(args, "output_dir", None) or (
            PROJECT_ROOT / "calibration")
        self._pair_base = self._default_pair_base()
        self.saved_paths: dict[str, Path] = {}
        self._auto_stable_since: float | None = None
        self._auto_last_compute_attempt_s: float | None = None
        self._auto_last_step = -1
        self._auto_status = ("", "")
        self._status = (
            "请先绑定成对的左右手，然后连接并开始双手标定。",
            "Bind a left+right pair first, then connect and start dual-hand "
            "calibration.")
        self._error = ("", "")
        self._lock = threading.RLock()
        self._busy = False
        self._worker: threading.Thread | None = None

    # -- helpers ------------------------------------------------------------
    def _start_worker(self, target, name: str) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._error = ("", "")

        def run():
            try:
                target()
            except BaseException as exc:  # Keep background errors inside the GUI.
                message = str(exc).strip() or exc.__class__.__name__
                if contains_han(message):
                    message = L(
                        "操作失败。请检查手套连接与姿势后重试。\n详情：{detail}",
                        "The operation failed. Check the glove connection and "
                        "pose, then retry.\nDetail: {detail}").format(detail=message)
                with self._lock:
                    self._error = (message, message)
                    self._status = ("操作失败。", "Operation failed.")
            finally:
                with self._lock:
                    self._busy = False

        self._worker = threading.Thread(target=run, name=name, daemon=True)
        self._worker.start()
        return True

    def _default_pair_base(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"imu_calibration_pair_{timestamp}"

    def _output_paths(self) -> dict[str, Path]:
        return {
            side: self._output_dir / f"{self._pair_base}_{side}.json"
            for side in ("left", "right")
        }

    def _output_text(self) -> str:
        return "\n".join(
            str(path.resolve()) for path in self._output_paths().values())

    def _hand_readiness(self, side: str) -> dict:
        hand = self.hands[side]
        warm, rms_ok, angle_ok = hand.auto_capture_readiness()
        return {
            "connected": hand.connected,
            "warm": warm,
            "rms_ok": rms_ok,
            "angle_ok": angle_ok,
            "rms": hand._total_rms,
            "worst_rms": hand._worst_rms,
        }

    def set_output_filename(self, filename: str) -> None:
        name = str(filename).strip()
        if name.lower().endswith(".json"):
            name = name[:-5]
        if not name or name in {".", ".."}:
            raise ValueError(L("JSON 文件名无效。", "JSON file name is invalid."))
        self._pair_base = name
        self._error = ("", "")

    def report_error(self, message: str) -> None:
        with self._lock:
            self._error = (str(message), str(message))
            self._status = ("操作失败。", "Operation failed.")

    # -- lifecycle ------------------------------------------------------------
    def connect(self) -> bool:
        def work():
            # Serial enumeration can block briefly on some Windows USB stacks;
            # keep it off the GUI thread (single-hand connect already boots in
            # a worker for the same reason).
            ports = detect_hand_pair(Path(self.args.registry))
            if ports is None:
                self.report_error(L(
                    "未检测到成对的左右手。请先绑定左右手套，再选择双手标定。",
                    "No matched left+right pair detected. Bind the left and "
                    "right gloves first, then choose dual-hand calibration."))
                return
            with self._lock:
                self._status = (
                    "正在连接双手并等待全部 16 个 IMU…",
                    "Connecting both hands and waiting for all 16 IMUs...")
            # 一次只启动一只手。两个后端同时扫 USB 注册表 / 打开串口在部分
            # Windows USB 栈上会死锁，所以串行启动（SDK 的双手 viewer 也是给
            # 每只手单独开进程，同理规避并发）。
            for side, hand in self.hands.items():
                hand.connect(serial_port=ports[side])
                while hand.busy:
                    time.sleep(0.1)
        return self._start_worker(work, "hand2mm-bimanual-connect")

    def start_new_session(self) -> bool:
        for hand in self.hands.values():
            hand.start_new_session()
        with self._lock:
            self._status = (
                "双手新标定会话已开始。请双手同时采集水平姿势。",
                "Dual-hand session started. Capture the level pose with both "
                "hands at once.")
        return True

    def capture_current_step(self) -> bool:
        left = self.hands["left"]
        right = self.hands["right"]
        if left.current_step != right.current_step:
            self.report_error(L(
                "左右手步骤不同步，请返回当前步骤后重试。",
                "Left and right steps are out of sync. Return to the current "
                "step and retry."))
            return False
        step_index = left.current_step
        step = self._steps[step_index]

        if left._is_compute_step(step):
            def compute_sequentially():
                # Torch/MANO construction and calibration state replacement
                # are not safe to run concurrently in two Python threads.
                # Pose acquisition remains simultaneous; only compute-only
                # steps are deliberately serialized here.
                for side, hand in self.hands.items():
                    side_zh = "左手" if side == "left" else "右手"
                    side_en = "left hand" if side == "left" else "right hand"
                    with self._lock:
                        self._status = (
                            f"正在计算{side_zh}标定，请稍候…",
                            f"Computing {side_en} calibration; please wait...")
                    try:
                        hand._execute_step(step_index)
                    except Exception as exc:
                        detail = str(exc).strip() or exc.__class__.__name__
                        raise RuntimeError(L(
                            "{side}计算失败：{detail}",
                            "{side} computation failed: {detail}").format(
                                side=side_zh if not is_en() else side_en,
                                detail=detail)) from exc
                with self._lock:
                    self._status = (
                        "双手计算完成，请继续下一步。",
                        "Both-hand computation completed. Continue to the next step.")

            return self._start_worker(
                compute_sequentially,
                f"hand2mm-bimanual-compute-step-{step_index}")

        # Capture steps must sample the two live streams at the same time, so
        # retain their existing parallel acquisition behavior.
        started = [hand.capture_current_step() for hand in self.hands.values()]
        if not all(started):
            self.report_error(L(
                "双手采集未能同时启动，请等待当前操作完成后重试。",
                "Both-hand capture could not start together. Wait for the "
                "current operation to finish and retry."))
            return False
        return True

    def go_to_step(self, index: int) -> None:
        for hand in self.hands.values():
            hand.go_to_step(index)

    def refresh_health(self) -> None:
        for hand in self.hands.values():
            hand.refresh_health()

    def reset_session(self) -> bool:
        any_connected = False
        for hand in self.hands.values():
            any_connected = hand.reset_session() or any_connected
        self.saved_paths = {}
        self._auto_stable_since = None
        self._auto_last_compute_attempt_s = None
        self._auto_last_step = -1
        self._auto_status = ("", "")
        return any_connected

    def retranslate(self) -> None:
        for hand in self.hands.values():
            hand.retranslate()
        self._steps = self.hands["left"]._steps

    def shutdown(self) -> None:
        for hand in self.hands.values():
            hand.shutdown()

    def save(self) -> bool:
        left = self.hands["left"]
        right = self.hands["right"]
        if not (left.complete and right.complete):
            raise RuntimeError(L(
                "保存前请完成所有标定步骤。",
                "Complete every calibration step before saving."))
        try:
            paths = self._output_paths()
            saved: dict[str, Path] = {}
            for side, hand in self.hands.items():
                payload = hand.backend.build_payload()
                payload["tool"] = "imu_calibration_gui"
                saved[side] = write_calibration_output(
                    paths[side], payload, force=self.replace_output)
        except Exception as exc:
            self.report_error(str(exc))
            return False
        with self._lock:
            self.saved_paths = saved
            self._status = (
                f"双手标定保存成功：\n{saved['left']}\n{saved['right']}",
                f"Dual calibration saved:\n{saved['left']}\n{saved['right']}")
        return True

    # -- snapshot ------------------------------------------------------------
    def snapshot(self) -> dict:
        left = self.hands["left"].snapshot()
        right = self.hands["right"].snapshot()
        connected = left["connected"] and right["connected"]
        session = left["session_started"] and right["session_started"]
        busy = self._busy or left["busy"] or right["busy"]
        complete = left["complete"] and right["complete"]
        progress = (left["progress"] + right["progress"]) / 2.0
        errors = [error for error in (left["error"], right["error"]) if error]
        with self._lock:
            status_pair = self._status
            error_pair = self._error
        if errors:
            status = L("操作失败。", "Operation failed.")
        elif busy:
            status = L(*status_pair)
        elif connected and not session:
            status = L(
                "双手 16 个 IMU 已就绪。开始新的标定会话。",
                "Both hands have 16 IMUs ready. Start a new calibration session.")
        else:
            status = L(*status_pair)
        error = " | ".join(errors) if errors else L(*error_pair)
        return {
            "mode": "dual",
            "side": "dual",
            "output": self._output_text(),
            "connected": connected,
            "session_started": session,
            "current_step": left["current_step"],
            "completed_steps": left["completed_steps"],
            "busy": busy,
            "complete": complete,
            "progress": progress,
            "status": status,
            "error": error,
            "device_text": left["device_text"] + "\n" + right["device_text"],
            "imu_text": left["imu_text"] + "\n" + right["imu_text"],
            "motion_text": left["motion_text"] + "\n" + right["motion_text"],
            "stream_fps": (
                min(left["stream_fps"], right["stream_fps"]) if connected
                else 0.0),
            "firmware_version": left["firmware_version"],
            "saved_path": (
                "\n".join(str(path) for path in self.saved_paths.values())
                if self.saved_paths else ""),
            "auto_capture": self.auto_capture,
            "rms_threshold": self.rms_threshold,
            "auto_status": L(*self._auto_status),
            "left": self._hand_readiness("left"),
            "right": self._hand_readiness("right"),
        }

    # -- auto-capture gate ----------------------------------------------------
    def auto_capture_tick(self) -> None:
        """Joint capture gate: fire only when both hands are ready together."""
        if not self.auto_capture:
            self._auto_stable_since = None
            return
        left = self.hands["left"]
        right = self.hands["right"]
        if self._busy or left.busy or right.busy:
            self._auto_stable_since = None
            self._auto_status = (
                "自动采集：正在采集或计算…",
                "Auto-capture: capturing or computing...")
            return
        if not (left.connected and right.connected
                and left.session_started and right.session_started):
            self._auto_stable_since = None
            return
        if left.complete and right.complete:
            self._auto_stable_since = None
            return

        # Heal a rare asymmetric failure (one hand advanced, the other was
        # rejected) by rolling both back to the lagging hand's step.
        if left.current_step != right.current_step:
            target = min(left.current_step, right.current_step)
            left.go_to_step(target)
            right.go_to_step(target)

        if left.current_step != self._auto_last_step:
            self._auto_stable_since = None
            self._auto_last_compute_attempt_s = None
            self._auto_last_step = left.current_step

        step = self._steps[left.current_step]

        if left._is_compute_step(step):
            now = time.monotonic()
            if (self._auto_last_compute_attempt_s is None
                    or now - self._auto_last_compute_attempt_s
                    >= _AUTO_COMPUTE_RETRY_S):
                self._auto_last_compute_attempt_s = now
                self.capture_current_step()
                self._auto_status = (
                    "自动采集：正在依次计算左手和右手…",
                    "Auto-capture: computing left then right...")
            else:
                self._auto_status = (
                    "自动采集：计算步骤未完成，稍后自动重试…",
                    "Auto-capture: compute step pending; will retry shortly...")
            return

        l_warm, l_rms, l_angle = left.auto_capture_readiness()
        r_warm, r_rms, r_angle = right.auto_capture_readiness()
        if l_warm and r_warm and l_rms and r_rms and l_angle and r_angle:
            now = time.monotonic()
            if self._auto_stable_since is None:
                self._auto_stable_since = now
            if now - self._auto_stable_since >= _AUTO_STABLE_DURATION_S:
                self._auto_stable_since = None
                left.capture_current_step()
                right.capture_current_step()
                return
            self._auto_status = (
                "自动采集：双手保持当前姿势…",
                "Auto-capture: hold the pose with both hands...")
        else:
            self._auto_stable_since = None
            l_ok = l_warm and l_rms and l_angle
            r_ok = r_warm and r_rms and r_angle
            if not l_ok and not r_ok:
                self._auto_status = (
                    "自动采集：双手未就绪（RMS 或角度）…",
                    "Auto-capture: both hands not ready (RMS or angle)...")
            elif not l_ok:
                self._auto_status = (
                    "自动采集：左手未就绪…", "Auto-capture: left hand not ready...")
            else:
                self._auto_status = (
                    "自动采集：右手未就绪…", "Auto-capture: right hand not ready...")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Qt GUI (zh/en) for HAND2mm STM32 16-IMU calibration.")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh",
                        help="Initial interface language (default: zh).")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--registry", type=Path,
                        default=PROJECT_ROOT / "config" / "glove_devices.json")
    parser.add_argument("--serial-port")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "calibration",
                        help="Directory used for GUI-selected JSON file names.")
    parser.add_argument("--force", action="store_true",
                        help="Allow replacement when --out already exists.")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    try:
        validate_hand2mm_runtime()
    except RuntimeError as exc:
        parser.error(str(exc))
    return args


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class CalibrationWindow(QWidget):
    """Qt window: left control panel + right guided status/preview panel."""

    def __init__(self, controller: CalibrationController, args):
        super().__init__()
        self._controller = controller
        self._dual_controller: BimanualCalibrationController | None = None
        self._dual_mode = False
        self._args = args
        self._preview = CalibrationPosePreview()
        self._detected_serials: dict[str, str] = {}
        self._shown_preview_key = None
        self._preview_bgr = None
        self._finished = False

        self._build_ui()
        self._retranslate_ui()
        self.refresh_devices()
        self._on_tick()

        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # -- construction helpers ----------------------------------------------
    def _heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #BECCEB; font-weight: 600;")
        return label

    def _muted(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #91AABF;")
        return label

    @staticmethod
    def _wrap(label: QLabel) -> None:
        label.setWordWrap(True)

    def _build_ui(self) -> None:
        self.resize(1240, 900)
        self.setMinimumSize(1100, 760)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        # Top bar: title + subtitle + language toggle.
        top = QHBoxLayout()
        self._app_title = QLabel()
        self._app_title.setStyleSheet(
            "font-size: 17px; color: #69BEFF; font-weight: 600;")
        self._app_subtitle = self._muted("")
        top.addWidget(self._app_title)
        top.addWidget(self._app_subtitle)
        top.addStretch(1)
        self._lang_btn = QPushButton()
        self._lang_btn.clicked.connect(self._toggle_language)
        top.addWidget(self._lang_btn)
        outer.addLayout(top)

        main = QHBoxLayout()
        main.setSpacing(14)
        outer.addLayout(main, 1)

        # ---- left column ----
        left = QWidget()
        left.setFixedWidth(470)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)

        self._device_heading = self._heading("")
        left_layout.addWidget(self._device_heading)

        self._side_label = QLabel()
        left_layout.addWidget(self._side_label)
        self._side_combo = QComboBox()
        self._side_combo.addItem("", "left")
        self._side_combo.addItem("", "right")
        self._side_combo.setCurrentIndex(
            0 if self._controller.side == "left" else 1)
        self._side_combo.currentIndexChanged.connect(self._choose_side)
        left_layout.addWidget(self._side_combo)

        self._detected_label = QLabel()
        left_layout.addWidget(self._detected_label)
        self._detected_combo = QComboBox()
        left_layout.addWidget(self._detected_combo)

        row = QHBoxLayout()
        self._refresh_btn = QPushButton()
        self._refresh_btn.clicked.connect(self.refresh_devices)
        self._bind_left_btn = QPushButton()
        self._bind_left_btn.clicked.connect(lambda: self._bind_selected("left"))
        self._bind_right_btn = QPushButton()
        self._bind_right_btn.clicked.connect(lambda: self._bind_selected("right"))
        row.addWidget(self._refresh_btn)
        row.addWidget(self._bind_left_btn)
        row.addWidget(self._bind_right_btn)
        left_layout.addLayout(row)

        self._swap_btn = QPushButton()
        self._swap_btn.clicked.connect(self._swap_bindings)
        left_layout.addWidget(self._swap_btn)

        self._clear_btn = QPushButton()
        self._clear_btn.clicked.connect(self._clear_bindings)
        left_layout.addWidget(self._clear_btn)

        self._mode_label = QLabel()
        left_layout.addWidget(self._mode_label)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("", "single")
        self._mode_combo.addItem("", "dual")
        self._mode_combo.setCurrentIndex(0)
        self._mode_combo.currentIndexChanged.connect(self._choose_mode)
        left_layout.addWidget(self._mode_combo)
        self._pair_hint = self._muted("")
        self._wrap(self._pair_hint)
        left_layout.addWidget(self._pair_hint)

        self._binding_status = QLabel()
        self._wrap(self._binding_status)
        left_layout.addWidget(self._binding_status)

        self._connect_btn = QPushButton()
        self._connect_btn.clicked.connect(self._connect_selected)
        left_layout.addWidget(self._connect_btn)
        self._start_btn = QPushButton()
        self._start_btn.clicked.connect(self._start_calibration)
        left_layout.addWidget(self._start_btn)

        self._device_text = QLabel()
        left_layout.addWidget(self._device_text)
        self._firmware_text = QLabel()
        left_layout.addWidget(self._firmware_text)
        self._imu_text = QLabel()
        left_layout.addWidget(self._imu_text)
        self._fps_text = QLabel()
        left_layout.addWidget(self._fps_text)
        self._motion_text = QLabel()
        self._wrap(self._motion_text)
        left_layout.addWidget(self._motion_text)

        left_layout.addWidget(_separator())

        self._steps_heading = self._heading("")
        left_layout.addWidget(self._steps_heading)
        self._step_buttons: list[QPushButton] = []
        for index in range(13):
            btn = QPushButton()
            btn.clicked.connect(
                lambda _checked=False, i=index: self._choose_step(i))
            self._step_buttons.append(btn)
            left_layout.addWidget(btn)

        left_layout.addWidget(_separator())

        self._output_heading = self._heading("")
        left_layout.addWidget(self._output_heading)
        self._output_name_label = QLabel()
        left_layout.addWidget(self._output_name_label)
        self._output_name = QLineEdit()
        self._output_name.setText(self._controller.output.name)
        left_layout.addWidget(self._output_name)
        self._output_path = QLabel()
        self._wrap(self._output_path)
        left_layout.addWidget(self._output_path)

        left_layout.addStretch(1)
        main.addWidget(left)

        # ---- right column ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        self._guided_heading = self._heading("")
        right_layout.addWidget(self._guided_heading)

        self._preview_caption = self._muted("")
        right_layout.addWidget(self._preview_caption)
        self._preview_label = QLabel()
        self._preview_label.setMinimumSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self._preview_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self._preview_label)
        self._compute_hint = QLabel()
        self._compute_hint.setStyleSheet("color: #5ADCFF; font-weight: 600;")
        self._compute_hint.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self._compute_hint)

        self._angle_heading = self._heading("")
        right_layout.addWidget(self._angle_heading)
        self._angle_readout = QLabel()
        self._wrap(self._angle_readout)
        right_layout.addWidget(self._angle_readout)
        self._angle_history = QLabel()
        self._wrap(self._angle_history)
        right_layout.addWidget(self._angle_history)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        right_layout.addWidget(self._progress)

        self._step_counter = QLabel()
        right_layout.addWidget(self._step_counter)
        self._step_title = QLabel()
        self._step_title.setStyleSheet("color: #69BEFF; font-weight: 600;")
        right_layout.addWidget(self._step_title)

        self._instruction_heading = self._heading("")
        right_layout.addWidget(self._instruction_heading)
        self._instruction = QLabel()
        self._wrap(self._instruction)
        right_layout.addWidget(self._instruction)

        self._requirement_heading = self._heading("")
        right_layout.addWidget(self._requirement_heading)
        self._hold = QLabel()
        self._wrap(self._hold)
        right_layout.addWidget(self._hold)

        self._auto_row = QHBoxLayout()
        self._auto_capture_check = QCheckBox()
        self._auto_capture_check.toggled.connect(self._auto_capture_changed)
        self._auto_capture_check.setChecked(self._controller.auto_capture)
        self._auto_row.addWidget(self._auto_capture_check)
        self._rms_threshold_label = QLabel()
        self._rms_threshold_label.setStyleSheet("color: #BECCEB;")
        self._auto_row.addWidget(self._rms_threshold_label)
        self._rms_threshold_spin = QDoubleSpinBox()
        self._rms_threshold_spin.setRange(0.1, 5.0)
        self._rms_threshold_spin.setSingleStep(0.1)
        self._rms_threshold_spin.setDecimals(1)
        self._rms_threshold_spin.setValue(self._controller.rms_threshold)
        self._rms_threshold_spin.setSuffix("°")
        self._rms_threshold_spin.valueChanged.connect(self._rms_threshold_changed)
        self._auto_row.addWidget(self._rms_threshold_spin)
        self._auto_row.addStretch(1)
        right_layout.addLayout(self._auto_row)

        self._action_btn = QPushButton()
        self._action_btn.clicked.connect(self._capture_step)
        right_layout.addWidget(self._action_btn)

        self._auto_status_label = self._muted("")
        self._wrap(self._auto_status_label)
        right_layout.addWidget(self._auto_status_label)

        self._dual_readiness = self._muted("")
        self._wrap(self._dual_readiness)
        right_layout.addWidget(self._dual_readiness)

        self._status_heading = self._heading("")
        right_layout.addWidget(self._status_heading)
        self._status_text = QLabel()
        self._wrap(self._status_text)
        right_layout.addWidget(self._status_text)
        self._error_text = QLabel()
        self._error_text.setStyleSheet("color: #FF6969;")
        self._wrap(self._error_text)
        right_layout.addWidget(self._error_text)

        self._replace_check = QCheckBox()
        self._replace_check.setChecked(self._controller.replace_output)
        self._replace_check.toggled.connect(self._replace_changed)
        right_layout.addWidget(self._replace_check)

        self._save_btn = QPushButton()
        self._save_btn.clicked.connect(self._save)
        right_layout.addWidget(self._save_btn)

        self._done_btn = QPushButton()
        self._done_btn.clicked.connect(self._return_to_main)
        right_layout.addWidget(self._done_btn)

        self._footer = self._muted("")
        self._wrap(self._footer)
        right_layout.addWidget(self._footer)

        right_layout.addStretch(1)
        main.addWidget(right, 1)

    def _retranslate_ui(self) -> None:
        self.setWindowTitle(
            L("Stouch Glove标定程序", "Stouch Glove Calibration")
            + f" V{APP_VERSION} (SDK v{get_version()})")
        self._app_title.setText(
            L("Stouch Glove标定程序", "Stouch Glove Calibration"))
        self._app_subtitle.setText(
            L("指根朝向、IMU 安装与指尖接触标定。",
              "Root orientation, IMU installation, and fingertip-contact "
              "calibration."))
        self._lang_btn.setText(L("English", "中文"))
        self._device_heading.setText(L("设备", "Device"))
        self._side_label.setText(L("手型", "Hand Side"))
        self._side_combo.setItemText(0, L("左手", "Left"))
        self._side_combo.setItemText(1, L("右手", "Right"))
        self._detected_label.setText(L("检测到的设备", "Detected Device"))
        self._refresh_btn.setText(L("刷新设备", "Refresh Devices"))
        self._bind_left_btn.setText(L("绑定为左手", "Bind as Left"))
        self._bind_right_btn.setText(L("绑定为右手", "Bind as Right"))
        self._swap_btn.setText(L("交换左右绑定", "Swap Left / Right Bindings"))
        self._clear_btn.setText(L("清除所有绑定", "Clear All Bindings"))
        self._mode_label.setText(L("标定模式", "Calibration Mode"))
        self._mode_combo.setItemText(0, L("单手标定", "Single Hand"))
        self._mode_combo.setItemText(1, L("双手标定", "Dual Hand"))
        self._connect_btn.setText(L("连接手套", "Connect Glove"))
        self._start_btn.setText(L("开始新标定", "Start New Calibration"))
        self._steps_heading.setText(L("标定步骤", "Calibration Steps"))
        self._output_heading.setText(L("输出文件", "Output File"))
        self._output_name_label.setText(L("JSON 文件名", "JSON File Name"))
        self._guided_heading.setText(L("引导式标定", "Guided Calibration"))
        self._angle_heading.setText(L("实时指根角度", "Live Root Angle"))
        self._instruction_heading.setText(L("姿势说明", "Pose Instruction"))
        self._requirement_heading.setText(L("采集要求", "Capture Requirement"))
        self._status_heading.setText(L("状态", "Status"))
        self._auto_capture_check.setText(
            L("自动采集", "Auto-capture"))
        self._rms_threshold_label.setText(
            L("RMS 阈值", "RMS threshold"))
        self._replace_check.setText(
            L("若所选输出文件已存在则替换",
              "Replace the selected output file if it already exists"))
        self._save_btn.setText(L("保存标定", "Save Calibration"))
        self._done_btn.setText(L("返回", "Back"))
        self._footer.setText(
            L("本界面会生成独立的标定 JSON 文件，绝不修改 config.json。",
              "The GUI writes an independent calibration JSON file and never "
              "modifies config.json."))

    # -- language switching -------------------------------------------------
    def _toggle_language(self) -> None:
        set_lang("zh" if is_en() else "en")
        self._controller.retranslate()
        if self._dual_controller is not None:
            self._dual_controller.retranslate()
        # Force the preview caption + "compute step" hint to retranslate too;
        # otherwise _apply_preview skips them until the current step changes.
        self._shown_preview_key = None
        self._retranslate_ui()
        self._on_tick()

    # -- handlers ------------------------------------------------------------
    def _replace_changed(self, checked: bool) -> None:
        self._controller.replace_output = bool(checked)
        if self._dual_controller is not None:
            self._dual_controller.replace_output = bool(checked)

    def _auto_capture_changed(self, checked: bool) -> None:
        value = bool(checked)
        self._controller.auto_capture = value
        self._controller._auto_stable_since = None
        if self._dual_controller is not None:
            self._dual_controller.auto_capture = value
            self._dual_controller._auto_stable_since = None

    def _rms_threshold_changed(self, value: float) -> None:
        self._controller.rms_threshold = float(value)
        if self._dual_controller is not None:
            self._dual_controller.rms_threshold = float(value)

    def _active_controller(self):
        return self._dual_controller if self._dual_mode else self._controller

    def _choose_mode(self, index: int) -> None:
        mode = self._mode_combo.itemData(index)
        if mode == "dual":
            if self._dual_controller is None:
                self._dual_controller = BimanualCalibrationController(self._args)
            self._dual_controller.auto_capture = self._controller.auto_capture
            self._dual_controller.rms_threshold = self._controller.rms_threshold
            self._dual_controller.replace_output = self._controller.replace_output
            self._dual_mode = True
        else:
            self._dual_mode = False
        self._output_name.setText(self._active_controller_output_name())
        self._on_tick()

    def _active_controller_output_name(self) -> str:
        controller = self._active_controller()
        if self._dual_mode:
            return controller._pair_base + ".json"
        return controller.output.name

    def _choose_side(self, index: int) -> None:
        if self._dual_mode:
            return
        side = self._side_combo.itemData(index)
        if side:
            self._controller.select_side(side)
            self._output_name.setText(self._controller.output.name)

    def _start_calibration(self) -> None:
        self._active_controller().start_new_session()

    def _capture_step(self) -> None:
        self._active_controller().capture_current_step()

    def _choose_step(self, index: int) -> None:
        self._active_controller().go_to_step(index)

    def _save(self) -> None:
        controller = self._active_controller()
        try:
            controller.set_output_filename(self._output_name.text())
        except (RuntimeError, ValueError) as exc:
            controller.report_error(str(exc))
            return
        controller.save()

    def _return_to_main(self) -> None:
        """Close the calibration window and hand control back to the main
        program (the launcher reopens the calibration-file selector)."""
        self._finished = True
        self.close()

    def refresh_devices(self) -> None:
        # While a glove is connected, Refresh Devices also resets the
        # calibration session (tears the backend down and clears state) so
        # the bind/connect flow can be redone from a clean slate.
        self._controller.reset_session()
        if self._dual_controller is not None:
            self._dual_controller.reset_session()
        try:
            options = detected_glove_options()
            self._detected_serials = dict(options)
            labels = list(self._detected_serials) or [
                L("未检测到 STM32 手套", "No STM32 gloves detected")]
            self._detected_combo.clear()
            for label in labels:
                self._detected_combo.addItem(label)
            bindings = glove_serial_bindings(Path(self._args.registry))
            intro = (
                L("检测到 {n} 个 STM32 手套。", "Detected {n} STM32 glove(s).")
                .format(n=len(options))
                if options else
                L("未检测到 STM32 手套。", "No STM32 gloves detected."))
            self._binding_status.setText(
                intro + " " + L("绑定：", "Bindings: ")
                + format_glove_bindings(bindings))
        except Exception as exc:
            self._detected_serials = {}
            self._detected_combo.clear()
            self._detected_combo.addItem(
                L("未检测到 STM32 手套", "No STM32 gloves detected"))
            self._binding_status.setText(
                L("设备刷新失败：{exc}", "Device refresh failed: {exc}")
                .format(exc=exc))
        self._update_pair_availability()

    def _update_pair_availability(self) -> None:
        """Enable the dual-hand mode option only when a full pair is bound."""
        pair = detect_hand_pair(Path(self._args.registry))
        enabled = pair is not None
        dual_index = self._mode_combo.findData("dual")
        if dual_index >= 0:
            self._mode_combo.model().item(dual_index).setEnabled(enabled)
        if not enabled and self._mode_combo.currentData() == "dual":
            self._mode_combo.setCurrentIndex(0)
        self._pair_hint.setText(
            L("检测到成对左右手，可进行双手标定。",
              "A matched left+right pair is detected; dual-hand calibration "
              "is available.")
            if enabled else
            L("未检测到成对左右手；仅可单手标定。",
              "No matched left+right pair detected; single-hand calibration "
              "only."))

    def _bind_selected(self, side: str) -> None:
        label = self._detected_combo.currentText()
        serial_number = self._detected_serials.get(label)
        if serial_number is None:
            self._binding_status.setText(
                L("请先选择一个检测到的 STM32 手套。",
                  "Select a detected STM32 glove first."))
            return
        locked_side = getattr(self._controller, "locked_side", None)
        if locked_side is not None and side != locked_side:
            locked_zh = "左" if locked_side == "left" else "右"
            side_zh = "左" if side == "left" else "右"
            self._binding_status.setText(L(
                "该手套固件已识别为{locked}手，不能绑定为{side}手。",
                "This glove's firmware is identified as the {locked} hand; "
                "it cannot be bound as the {side} hand.").format(
                    locked=locked_zh, side=side_zh))
            return
        try:
            path = bind_glove_serial(side, serial_number, Path(self._args.registry))
            bindings = glove_serial_bindings(Path(self._args.registry))
            side_label = L("左", "Left") if side == "left" else L("右", "Right")
            self._binding_status.setText(
                L("已将序列号 {serial} 绑定到{side}手，保存在 {path}。",
                  "Bound serial {serial} to the {side} hand in {path}.").format(
                      serial=serial_number, side=side_label, path=path)
                + " " + format_glove_bindings(bindings))
            self._side_combo.setCurrentIndex(0 if side == "left" else 1)
            if not self._dual_mode:
                self._controller.select_side(side)
            self._output_name.setText(self._active_controller_output_name())
        except Exception as exc:
            self._binding_status.setText(
                L("序列号绑定失败：{exc}", "Serial binding failed: {exc}")
                .format(exc=exc))
        self._update_pair_availability()

    def _swap_bindings(self) -> None:
        try:
            path = swap_glove_bindings(Path(self._args.registry))
            bindings = glove_serial_bindings(Path(self._args.registry))
            self._binding_status.setText(
                L("已在 {path} 中交换左右绑定。",
                  "Swapped left/right bindings in {path}.").format(path=path)
                + " " + format_glove_bindings(bindings))
        except Exception as exc:
            self._binding_status.setText(
                L("绑定交换失败：{exc}", "Binding swap failed: {exc}")
                .format(exc=exc))
        self._update_pair_availability()

    def _clear_bindings(self) -> None:
        answer = QMessageBox.question(
            self,
            L("清除所有绑定", "Clear all bindings"),
            L("确定要清除所有已记住的手套序列号吗？标定文件不会被删除。",
              "Clear every remembered glove serial? Calibration files are kept."),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            path = clear_glove_bindings(Path(self._args.registry))
            bindings = glove_serial_bindings(Path(self._args.registry))
            self._binding_status.setText(
                L("已清除所有绑定记录（{path}）。",
                  "Cleared all binding records ({path}).").format(path=path)
                + " " + format_glove_bindings(bindings))
        except Exception as exc:
            self._binding_status.setText(
                L("清除绑定失败：{exc}", "Clearing bindings failed: {exc}")
                .format(exc=exc))
        self._update_pair_availability()

    def _connect_selected(self) -> None:
        """Connect the device currently shown in the live detection combo.

        This avoids a confusing failure when a replacement board has a new
        USB serial number that is not yet in ``glove_devices.json``.  The
        explicit Bind buttons remain available for making that assignment
        persistent; connecting does not change the registry.

        In dual mode both gloves are resolved by their bound left/right serials
        and connected together.
        """
        if self._dual_mode:
            self._active_controller().connect()
            return
        label = self._detected_combo.currentText()
        serial_number = self._detected_serials.get(label)
        selected_port = detected_glove_port(serial_number) \
            if serial_number else None
        if serial_number and selected_port:
            self._controller.connect(serial_port=selected_port)
        else:
            # Preserve the normal registry-based path when no live selection
            # is available so the resulting error still contains diagnostics.
            self._controller.connect()

    # -- periodic refresh ----------------------------------------------------
    def _on_tick(self) -> None:
        controller = self._active_controller()
        controller.refresh_health()
        controller.auto_capture_tick()
        state = controller.snapshot()
        self._apply_snapshot(state)
        self._apply_preview(state)

    def _apply_snapshot(self, state: dict) -> None:
        controller = self._active_controller()
        dual = state.get("mode") == "dual"
        self._device_text.setText(state["device_text"])
        firmware_version = state["firmware_version"]
        self._firmware_text.setText(
            L("固件：V{0}", "Firmware: V{0}").format(firmware_version)
            if firmware_version is not None else L("固件：--", "Firmware: --"))
        self._imu_text.setText(state["imu_text"])
        self._fps_text.setText(
            L("流速率：{0:.1f} FPS", "Stream rate: {0:.1f} FPS").format(
                state["stream_fps"])
            if state["connected"] else L("流速率：--", "Stream rate: --"))
        self._motion_text.setText(state["motion_text"])

        steps = controller._steps
        step = steps[state["current_step"]]
        self._step_counter.setText(
            L("步骤 {cur} / {total}", "Step {cur} of {total}").format(
                cur=state["current_step"] + 1, total=len(steps)))
        if dual:
            index = state["current_step"]
            left_step = controller.hands["left"]._steps[index]
            right_step = controller.hands["right"]._steps[index]
            if (left_step.title != right_step.title
                    or left_step.instruction != right_step.instruction):
                self._step_title.setText(
                    L("指根 - 双手向内水平转", "Root - Turn Both Hands Inward"))
                self._instruction.setText(
                    L("左手：", "Left: ") + left_step.instruction + "\n"
                    + L("右手：", "Right: ") + right_step.instruction)
            else:
                self._step_title.setText(step.title)
                self._instruction.setText(step.instruction)
            self._hold.setText(step.hold)
        else:
            self._step_title.setText(step.title)
            self._instruction.setText(step.instruction)
            self._hold.setText(step.hold)
        self._action_btn.setText(step.action)

        self._status_text.setText(state["status"])
        self._error_text.setText(state["error"])

        self._progress.setValue(round(state["progress"] * 100))
        self._output_path.setText(state["output"])

        busy = state["busy"]
        connected = state["connected"]
        session = state["session_started"]
        side_locked = bool(state.get("side_locked", False))
        self._side_combo.setEnabled(not busy and not dual and not side_locked)
        self._side_label.setText(
            L("手型（已自动识别）", "Hand Side (auto-detected)")
            if side_locked else L("手型", "Hand Side"))
        self._mode_combo.setEnabled(not busy and not connected)
        self._refresh_btn.setEnabled(not busy)
        self._bind_left_btn.setEnabled(not busy and not connected)
        self._bind_right_btn.setEnabled(not busy and not connected)
        self._swap_btn.setEnabled(not busy and not connected)
        self._connect_btn.setEnabled(not busy and not connected)
        self._start_btn.setEnabled(not busy and connected)
        self._output_name.setEnabled(not busy)
        self._action_btn.setEnabled(
            not busy and connected and session and not state["auto_capture"])
        self._save_btn.setEnabled(not busy and state["complete"])
        self._done_btn.setEnabled(not busy)

        self._auto_status_label.setText(state["auto_status"])
        self._auto_status_label.setVisible(state["auto_capture"])

        for index, btn in enumerate(self._step_buttons):
            title = steps[index].title
            if dual and index == 4:
                title = L("指根 - 双手向内水平转",
                          "Root - Turn Both Hands Inward")
            if index in state["completed_steps"]:
                marker = L("完成", "DONE")
            elif index == state["current_step"]:
                marker = L("当前", "CURRENT")
            else:
                marker = L("待定", "PENDING")
            btn.setText(f"{index + 1}. [{marker}] {title}")
            btn.setEnabled(not busy and session)

        if dual:
            self._angle_readout.setTextFormat(Qt.RichText)
            self._angle_readout.setText(self._dual_angle_html(controller))
            self._angle_readout.setStyleSheet("")
            self._angle_history.setText("")
            self._dual_readiness.setText(self._format_dual_readiness(state))
            self._dual_readiness.show()
        else:
            self._dual_readiness.hide()
            self._angle_readout.setTextFormat(Qt.PlainText)
            angle_snapshot = controller.root_angle_snapshot()
            if angle_snapshot is None:
                self._angle_readout.setText("")
                self._angle_history.setText("")
            else:
                readout, color, history = format_root_angle(angle_snapshot)
                self._angle_readout.setText(readout)
                self._angle_readout.setStyleSheet(
                    f"color: rgb({color[0]},{color[1]},{color[2]});")
                self._angle_history.setText(history)

    def _format_dual_readiness(self, state: dict) -> str:
        """Two-line readiness readout: one RMS + angle line per hand."""
        lines = []
        for label, key in (
                (L("左手", "Left hand"), "left"),
                (L("右手", "Right hand"), "right")):
            info = state[key]
            if not info["connected"]:
                lines.append(f"{label}: " + L("未连接", "not connected"))
                continue
            rms_mark = "✓" if info["rms_ok"] else "✗"
            angle_mark = "✓" if info["angle_ok"] else "✗"
            lines.append(
                f"{label}: RMS {info['rms']:.1f}°{rms_mark}  "
                + L("角度", "angle") + angle_mark)
        return "\n".join(lines)

    def _dual_angle_html(self, controller) -> str:
        """Two-line live root-angle readout (rich text): left hand on the
        first line, right hand on the second, each colored by its own verdict
        just like the single-hand readout."""
        muted = _ANGLE_COLORS["muted"]
        lines = []
        for label, side in (
                (L("左手", "Left hand"), "left"),
                (L("右手", "Right hand"), "right")):
            snap = controller.hands[side].root_angle_snapshot()
            if snap is None:
                lines.append(
                    f'<span style="color: rgb({muted[0]},{muted[1]},{muted[2]});">'
                    f"{label} --</span>")
                continue
            readout, color, _history = format_root_angle(snap)
            lines.append(
                f'<span style="color: rgb({color[0]},{color[1]},{color[2]});">'
                f"{label} {readout}</span>")
        return "<br/>".join(lines)

    def _apply_preview(self, state: dict) -> None:
        if state.get("mode") == "dual":
            key = ("dual", state["current_step"])
            if key != self._shown_preview_key:
                self._shown_preview_key = key
                bgr = self._preview.render_bimanual_bgr(
                    state["current_step"])
                self._preview_bgr = bgr
                height, width = bgr.shape[:2]
                image = QImage(
                    bgr.data, width, height, bgr.strides[0],
                    QImage.Format.Format_BGR888)
                self._preview_label.setPixmap(QPixmap.fromImage(image))
                self._preview_caption.setText(
                    L("双手完整手姿势参考 | 左右手分别按 L / R 演示动作",
                      "Dual Full-Hand Reference | Follow the separate L / R poses"))
            if state["current_step"] in (5, 7, 12):
                self._compute_hint.setText(
                    L("无需姿势 - 计算步骤", "NO POSE REQUIRED - COMPUTE STEP"))
                self._compute_hint.show()
            else:
                self._compute_hint.hide()
            return
        key = (state["side"], state["current_step"])
        if key != self._shown_preview_key:
            self._shown_preview_key = key
            bgr = self._preview.render_bgr(*key)
            self._preview_bgr = bgr  # keep alive for the zero-copy QImage
            height, width = bgr.shape[:2]
            image = QImage(
                bgr.data, width, height, bgr.strides[0],
                QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(image)
            self._preview_label.setPixmap(pixmap)
            side_label = L("左", "LEFT") if state["side"] == "left" else L("右", "RIGHT")
            self._preview_caption.setText(
                L("轻量 3D 姿势参考 | {side} | 步骤 {n}",
                  "Full 3D Hand Reference | {side} | STEP {n}").format(
                      side=side_label, n=state["current_step"] + 1))
            if state["current_step"] in (5, 7, 12):
                self._compute_hint.setText(
                    L("无需姿势 - 计算步骤", "NO POSE REQUIRED - COMPUTE STEP"))
                self._compute_hint.show()
            else:
                self._compute_hint.hide()

    def closeEvent(self, ev) -> None:  # noqa: N802
        self._controller.shutdown()
        if self._dual_controller is not None:
            self._dual_controller.shutdown()
        ev.accept()


def run_calibration_session(args) -> int | str:
    """Run one calibration session.

    Returns ``"done"`` when the user pressed Calibration Complete (the launcher
    then switches back to the live viewer), otherwise the Qt event-loop exit
    code (an int).
    """
    set_lang(args.lang)
    controller = CalibrationController(args)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = CalibrationWindow(controller, args)
    window.show()
    exit_code = app.exec()
    if window._finished:
        return "done"
    return exit_code


def run_gui(args) -> int:
    """Standalone entry point: run a session and map ``"done"`` to exit code 0."""
    result = run_calibration_session(args)
    return 0 if result == "done" else result


def main(argv=None) -> int:
    print(f"Host software v{APP_VERSION}, SDK v{get_version()}: "
          f"HAND2mm 16-IMU calibration")
    return run_gui(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
