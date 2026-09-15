"""Qt (PySide6) startup selector: pick the left/right HAND2mm calibration JSON files.

The function signature and return semantics match the original DearPyGui
version exactly (the three live callers need no changes), but the GUI is now a
Qt QDialog and **keeps the glove-detection/refresh capability**: ``refresh_status``
is the caller-supplied USB re-scan callback (plain device_manager.resolve_port
logic, independent of the GUI framework); the Qt dialog's Refresh button calls
it directly, with the same effect as the original.

PySide6 is imported lazily inside this function; no QApplication is created at
module import time, so headless scenarios such as replay/calibration previews
remain safe.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

from gui import APP_VERSION

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
    """Return newest-first HAND2mm calibration JSON files for one side."""
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
        if payload.get("kinematics") in {"direct_fk", "fitted_direct_fk"}:
            continue
        if str(payload.get("tool") or "") in {
                "direct_fk_calibrate_cli", "hand_fk_fit_cli"}:
            continue
        matches.append(path.resolve())
    return sorted(
        matches, key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True)


def _label(path: Path) -> str:
    modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"{path.name}    [{modified}]"


def select_calibration_files(
        directory: Path = DEFAULT_CALIBRATION_DIR,
        sides: tuple[str, ...] = ("right",),
        title: str = "Select IMU Calibration",
        connected: dict[str, bool] | None = None,
        refresh_status=None,
        calibrate_button: bool = False) -> dict[str, Path] | str | None:
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

    def discover(side: str) -> list[Path]:
        try:
            return discover_calibration_files(directory, side)
        except (OSError, RuntimeError):
            return []

    options = {side: discover(side) for side in render_sides}
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
            f"No HAND2mm calibration JSON found for: {', '.join(missing)} "
            f"{Path(directory).resolve()}")

    # Lazy PySide6 import keeps module import time headless-safe.
    # Note: QFrame lives in QtWidgets (not QtGui) - a common PySide6 gotcha.
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
        QPushButton, QVBoxLayout)

    app = QApplication.instance() or QApplication([])

    dialog = QDialog()
    dialog.setWindowTitle(f"{title} V{APP_VERSION}")
    dialog.setMinimumWidth(760)

    root = QVBoxLayout(dialog)
    intro = QLabel("Choose calibration JSON before opening the live viewer.")
    root.addWidget(intro)
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
    has_files = any(bool(options.get(side)) for side in render_sides)
    calibrate_btn: QPushButton | None = None
    if calibrate_button:
        calibrate_btn = QPushButton("标定 Calibrate")
        buttons.addWidget(calibrate_btn)
    open_button = QPushButton("Open Live Viewer")
    open_button.setDefault(True)
    open_button.setEnabled(has_files)
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
        enabled_sides: list[str] = []
        for side in render_sides:
            is_connected = bool(connection_status.get(side, False))
            paths = discover(side) if is_connected else []
            combo = combos[side]
            combo.blockSignals(True)
            combo.clear()
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
                    enabled_sides.append(side)
                elif is_connected:
                    status.setText("✓ no JSON")
                    status.setStyleSheet("color: rgb(240,150,40);")
                else:
                    status.setText("✗ NOT CONNECTED")
                    status.setStyleSheet("color: rgb(235,70,70);")
        collect_sides = enabled_sides
        if enabled_sides:
            open_button.setText("Open Live Viewer")
            open_button.setEnabled(True)
        else:
            open_button.setText("No glove connected")
            open_button.setEnabled(False)

    def do_confirm() -> None:
        nonlocal result
        collected: dict[str, Path] = {}
        for side in collect_sides:
            pick = combos[side].currentData()
            if pick is not None:
                collected[side] = Path(pick)
        if collected:
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
