"""Public HAND2mm IMU calibration interfaces, and calibration-file discovery.

Two calibration engines live behind this module, and the difference is not
cosmetic:

* :class:`ImuCalibrator` -- the step-driven session engine
  (``algorithm.calibration_session``).  Fully scripted: a caller asks for a
  step, the engine tells it what to do and validates the result.
* :class:`ImuCalibrationCLI` -- the engine the calibration **GUI** drives
  (``algorithm.imu_calibrate_cli``).  It exposes raw per-step primitives that
  the GUI sequences itself, which is why the GUI's magnetometer gate behaves
  differently from the scripted flow and why it is re-exported rather than
  replaced.

Both are SDK surface: a caller outside this repo should be able to run either
without importing ``algorithm`` by path.
"""

from __future__ import annotations

import json
from pathlib import Path

from algorithm.calibration_session import ImuCalibratorEngine as _Engine
from algorithm.calibration_profiles import (
    CalibrationProfileResolution, prepare_viewer_calibration,
    resolve_calibration_profile)
from algorithm.imu_calibrate_cli import (
    ImuCalibrationCLI, compute_motion_rms, default_out_path)
from algorithm.utils import get_average_rotation
from common.errors import CalibrationError
from common.types import (
    CalibrationProgress, CalibrationStep, CalibrationStepResult, DeviceHealth)

__all__ = [
    "ImuCalibrator", "ImuCalibrationCLI", "compute_motion_rms",
    "default_out_path", "find_latest_hand2mm_calibration",
    "get_average_rotation",
    "CalibrationProfileResolution", "resolve_calibration_profile",
    "prepare_viewer_calibration",
]


def find_latest_hand2mm_calibration(directory: Path | str, side: str) -> Path:
    """Find the newest HAND2mm-compatible calibration for one hand.

    "Compatible" is the point: the folder holds calibrations from several
    tools, and only some produce what the HAND2mm path consumes.  A direct-FK
    calibration (``kinematics`` of ``direct_fk``/``fitted_direct_fk``, or a
    ``tool`` of ``direct_fk_calibrate_cli``/``hand_fk_fit_cli``) is a different
    solver's output and is skipped rather than picked by recency.
    """

    folder = Path(directory).expanduser().resolve()
    if not folder.is_dir():
        raise CalibrationError(f"Calibration directory does not exist: {folder}")
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
        # contact payload.  Direct-FK factory files use a different solver.
        if payload.get("kinematics") in {"direct_fk", "fitted_direct_fk"}:
            continue
        if str(payload.get("tool") or "") in {
                "direct_fk_calibrate_cli", "hand_fk_fit_cli"}:
            continue
        matches.append(path)
    if not matches:
        raise CalibrationError(
            f"No usable HAND2mm {side} IMU calibration JSON in {folder}")
    canonical = folder / f"imu_calibration_{side}.json"
    if canonical in matches:
        legacy_names = {"imu_2d_calibration.json", "imu_calibration.json",
                        "imu_default_left.json", "imu_default_right.json",
                        "imu_calibration_left_default.json", "imu_calibration_right_default.json"}
        matches = [path for path in matches if path.name not in legacy_names]
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, path.name))


class ImuCalibrator:
    __slots__ = ("_impl",)

    def __init__(self, side: str, serial_port: str | None = None,
                 registry: Path | str | None = None,
                 config: Path | str | None = None,
                 startup_timeout: float = 30.0):
        # Calibration reads IMU, tactile and status frames only; the type-0x05
        # magnetometer telemetry is not consumed here and :meth:`connect` never
        # waits for it.  The live viewer's own receiver is what caches 0x05.
        self._impl = _Engine(
            side, serial_port, registry, config, startup_timeout)

    @property
    def steps(self) -> tuple[CalibrationStep, ...]:
        return self._impl.steps

    def connect(self) -> DeviceHealth:
        return self._impl.connect()

    def start_new(self) -> CalibrationProgress:
        return self._impl.start_new()

    def run_step(self, step_id: str) -> CalibrationStepResult:
        return self._impl.run_step(step_id)

    def retry_step(self, step_id: str) -> CalibrationStepResult:
        return self._impl.retry_step(step_id)

    def progress(self) -> CalibrationProgress:
        return self._impl.progress()

    def health(self) -> DeviceHealth:
        return self._impl.health()

    def motion_rms(self) -> tuple[float, float]:
        return self._impl.motion_rms()

    def save(self, filename: str, overwrite: bool = False,
             directory: Path | str | None = None) -> Path:
        return self._impl.save(filename, overwrite, directory)

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "ImuCalibrator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
