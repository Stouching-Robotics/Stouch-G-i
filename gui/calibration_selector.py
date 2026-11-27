"""Qt startup selector for left and right Friston wearing calibration files."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

from gui import APP_VERSION
from common.calibration_root import ROOT_METHOD_SIMPLIFIED
from common.calibration_validation import is_return_open_skipped
from common.i18n import L

def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "calibration"


def discover_calibration_files(directory: Path, side: str) -> list[Path]:
    """List solver-compatible Friston calibrations, newest first."""
    from common.errors import CalibrationError
    from sdk import open_pose_processor

    folder = Path(directory).expanduser().resolve()
    matches = []
    for path in [*folder.glob("*.json"), *(folder / "experiments").glob("*.json")]:
        try:
            payload = _read_profile(path)
            if payload.get("side") != side or not isinstance(payload.get("friston"), dict):
                continue
            open_pose_processor(side, path)
            matches.append((path.parent != folder, -path.stat().st_mtime_ns, path.name, path))
        except (OSError, ValueError, CalibrationError):
            continue
    return [path for _, _, _, path in sorted(matches)]


def _read_profile(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a calibration object in {path}")
    return payload


def _label(path: Path) -> str:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        payload = _read_profile(path)
    except (OSError, ValueError, OverflowError):
        return f"{L('[文件无效]', '[Invalid file]')} {path.name}"

    tags = ["Friston"]
    installation = payload.get("param_inst_calib") or {}
    metadata = installation.get("metadata") or {}
    if metadata.get("root_method") == ROOT_METHOD_SIMPLIFIED:
        tags.append(L("简化标定", "Simplified"))
    shape = payload.get("param_shape_calib") or {}
    profile = shape.get("sensor_motion_refit") or {}
    quality = profile.get("quality") or {}
    check = quality.get("return_open_validation")
    if isinstance(check, dict) and check.get("passed") is True:
        tags.append(L("平展已验收", "Open pose verified"))
    elif is_return_open_skipped(check):
        tags.append(L("平展已跳过", "Open verification skipped"))
    elif profile:
        tags.append(L("平展未验收", "Open pose unverified"))
    return f"[{' / '.join(tags)}] {path.name}    [{modified}]"


def select_calibration_files(
        directory: Path = DEFAULT_CALIBRATION_DIR,
        sides: tuple[str, ...] = ("right",),
        title: str = "Select IMU Calibration",
        connected: dict[str, bool] | None = None,
        refresh_status=None,
        calibrate_button: bool = False,
        connection_notice: str | None = None) -> dict[str, Path] | str | None:
    """Open a blocking startup selector; return None when the user cancels.

    ``sides`` lists the hands whose calibration JSON must be picked (the
    connected hands); ``connected`` optionally maps every hand to its USB
    connection state.  When ``connected`` is given, both the left and the
    right rows are always rendered — a missing glove's row stays visible with
    a red cross + ``NOT CONNECTED`` at the end, and connected hands get a
    green check.  Only currently-connected sides are collected on confirm.

    ``refresh_status`` (used together with ``connected``) is a callable that
    re-detects USB connection state on demand; when given, a ``Refresh``
    button appears and re-runs it, updating the ✓/✗ indicators, the combo
    enable states, and the Open button live.
    """
    normalized = tuple(str(side).lower() for side in sides)
    if not normalized or any(side not in ("left", "right") for side in normalized):
        raise ValueError("sides must contain left and/or right")
    if connected is not None:
        connection_status = {
            str(side).lower(): bool(status) for side, status in connected.items()
        }
        render_sides = ("left", "right")
    else:
        connection_status = {side: True for side in normalized}
        render_sides = normalized

    options = {
        side: discover_calibration_files(directory, side)
        if connection_status.get(side) else []
        for side in render_sides
    }
    missing = [
        side for side in normalized
        if side in options and not options[side]
    ]
    # With a Calibrate button the dialog is also the entry point into the
    # calibration program, so an empty list is allowed (the user calibrates
    # instead of picking an existing file).  Without it, keep the original
    # "no JSON found" error for the plain file-picker callers.
    if missing and not calibrate_button:
        raise RuntimeError(
            f"No Friston calibration JSON found for: {', '.join(missing)} "
            f"{Path(directory).resolve()}")

    # Lazy PySide6 import keeps module import time headless-safe.
    # Note: QFrame lives in QtWidgets (not QtGui) - a common PySide6 gotcha.
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
        QPushButton, QVBoxLayout)

    app = QApplication.instance() or QApplication([])

    dialog = QDialog()
    dialog.setWindowTitle(f"{title} V{APP_VERSION}")
    # The fields can shrink and their labels wrap, so avoid forcing a wide
    # desktop-only dialog on 1366px laptops or split-screen use.
    dialog.setMinimumWidth(520)

    root = QVBoxLayout(dialog)
    hint = L("请选择当前手套的 Friston 佩戴标定。",
             "Select the Friston calibration for the connected glove.")
    if calibrate_button:
        hint += L(" 若没有可选文件，请点击“标定 Calibrate”完成标定并保存。",
                  " If no file is listed, choose Calibrate, complete the calibration and save it.")
    intro = QLabel(hint)
    intro.setWordWrap(True)
    root.addWidget(intro)
    if connection_notice:
        warning = QLabel(connection_notice)
        warning.setWordWrap(True)
        warning.setStyleSheet("color: rgb(235,165,75)")
        root.addWidget(warning)
    folder_label = QLabel(f"Folder: {Path(directory).resolve()}")
    folder_label.setWordWrap(True)
    folder_label.setStyleSheet("color: rgb(120,120,120)")
    root.addWidget(folder_label)
    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)
    root.addWidget(separator)

    combos: dict[str, QComboBox] = {}
    status_labels: dict[str, QLabel] = {}
    for side in render_sides:
        is_connected = bool(connection_status.get(side, False))
        side_name = "Left" if side == "left" else "Right"

        row = QHBoxLayout()
        name_label = QLabel(f"{side_name} JSON")
        name_label.setMinimumWidth(120)
        combo = QComboBox()
        if is_connected and not options[side]:
            combo.setPlaceholderText(L("缺少标定文件", "No calibration file"))
        for path in options[side]:
            combo.addItem(_label(path), path)
        if options[side]:
            combo.setCurrentIndex(0)
        combo.setEnabled(is_connected and bool(options[side]))
        combos[side] = combo
        row.addWidget(name_label)
        row.addWidget(combo, 1)
        if connected is not None:
            status = QLabel("✓ connected" if is_connected else "✗ NOT CONNECTED")
            status.setStyleSheet(
                "color: rgb(40,210,80);" if is_connected else "color: rgb(235,70,70);")
            status_labels[side] = status
            row.addWidget(status)
        root.addLayout(row)

    root.addSpacing(12)
    buttons = QHBoxLayout()
    refresh_button: QPushButton | None = None
    if connected is not None and callable(refresh_status):
        refresh_button = QPushButton("↻ Refresh")
        buttons.addWidget(refresh_button)
    buttons.addStretch(1)
    calibrate_btn: QPushButton | None = None
    if calibrate_button:
        calibrate_btn = QPushButton("标定 Calibrate")
        buttons.addWidget(calibrate_btn)
    open_button = QPushButton("Open Live Viewer")
    open_button.setDefault(True)
    open_button.setEnabled(bool(normalized) and all(options[side] for side in normalized))
    cancel_button = QPushButton("Cancel")
    buttons.addWidget(open_button)
    buttons.addWidget(cancel_button)
    root.addLayout(buttons)

    result: dict[str, Path] | str | None = None
    collect_sides: list[str] = list(normalized)

    def do_refresh() -> None:
        nonlocal connection_status, collect_sides
        if not callable(refresh_status):
            return
        fresh = refresh_status()
        connection_status = {
            str(side).lower(): bool(status) for side, status in fresh.items()
        }
        for side in render_sides:
            is_connected = bool(connection_status.get(side, False))
            paths = discover_calibration_files(directory, side) if is_connected else []
            combo = combos[side]
            combo.blockSignals(True)
            combo.clear()
            combo.setPlaceholderText(
                L("缺少标定文件", "No calibration file") if is_connected and not paths else "")
            for path in paths:
                combo.addItem(_label(path), path)
            if paths:
                combo.setCurrentIndex(0)
            combo.setEnabled(is_connected and bool(paths))
            combo.blockSignals(False)
            if connected is not None:
                status = status_labels[side]
                if is_connected and paths:
                    status.setText("✓ connected")
                    status.setStyleSheet("color: rgb(40,210,80);")
                elif is_connected:
                    status.setText("✓ no JSON")
                    status.setStyleSheet("color: rgb(240,150,40);")
                else:
                    status.setText("✗ NOT CONNECTED")
                    status.setStyleSheet("color: rgb(235,70,70);")
        collect_sides = [side for side in render_sides if connection_status.get(side)]
        open_button.setEnabled(bool(collect_sides) and all(
            combos[side].count() for side in collect_sides))

    def do_confirm() -> None:
        nonlocal result
        collected: dict[str, Path] = {}
        for side in collect_sides:
            pick = combos[side].currentData()
            if pick is not None:
                collected[side] = Path(pick)
        if collected and len(collected) == len(collect_sides):
            result = collected
            dialog.accept()

    def do_cancel() -> None:
        dialog.reject()

    def do_calibrate() -> None:
        nonlocal result
        result = "calibrate"
        dialog.accept()

    if refresh_button is not None:
        refresh_button.clicked.connect(do_refresh)
    if calibrate_btn is not None:
        calibrate_btn.clicked.connect(do_calibrate)
    open_button.clicked.connect(do_confirm)
    cancel_button.clicked.connect(do_cancel)

    dialog.exec()
    return result
