"""Serial -> calibration-file bindings shared by the live viewer and calibration GUI.

Both the live viewer (``live_3d_bimanual_auto``) and the calibration GUI
(``imu_calibration_gui``) need to read/write the same persisted map of a glove's
USB serial number to the calibration JSON that was produced for it.  Keeping the
helpers here avoids a heavyweight / circular import between the two programs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")


if getattr(sys, "frozen", False):
    # Mutable files live beside the portable exe (PyInstaller _MEIPASS is temp).
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)

DEFAULT_CALIBRATION_BINDINGS = PROJECT_ROOT / "config" / "calibration_bindings.json"


def load_calibration_bindings(
        path: Path | str = DEFAULT_CALIBRATION_BINDINGS) -> dict[str, str]:
    """Return the persisted serial -> calibration-file map (empty if absent/invalid)."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("bindings")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for serial, calib in raw.items():
        if isinstance(serial, str) and isinstance(calib, str) and serial and calib:
            result[serial.strip().upper()] = calib
    return result


def save_calibration_binding(
        serial: str, calib_path: str | Path,
        path: Path | str = DEFAULT_CALIBRATION_BINDINGS) -> None:
    """Atomically record one glove's calibration file, keyed by its USB serial.

    The path is stored repo-relative (e.g. ``calibration/imu_calibration.json``)
    whenever the file lives under the project root, so the same binding works on
    any machine; a file outside the project keeps its absolute path.
    """
    path = Path(path)
    bindings = load_calibration_bindings(path)
    calib = Path(calib_path).expanduser()
    if not calib.is_absolute():
        calib = PROJECT_ROOT / calib
    try:
        stored = calib.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        stored = str(calib)  # outside the project: fall back to absolute
    bindings[str(serial).strip().upper()] = stored
    payload = {"schema_version": 1, "bindings": bindings}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def remembered_calibration(
        bindings: dict[str, str], serial: str | None) -> Path | None:
    """Resolve a remembered calibration file for a serial, or None if absent/stale."""
    if not serial:
        return None
    stored = bindings.get(str(serial).strip().upper())
    if not stored:
        return None
    calib = Path(stored).expanduser()
    if not calib.is_absolute():
        calib = PROJECT_ROOT / calib
    return calib if calib.is_file() else None
