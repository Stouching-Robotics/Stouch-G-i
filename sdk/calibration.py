"""Public Friston IMU calibration interfaces, and calibration-file discovery.

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

from pathlib import Path

from algorithm.calibration_session import ImuCalibratorEngine as _Engine
from algorithm.calibration_profiles import (
    CalibrationProfileCatalog, CalibrationProfileResolution,
    prepare_viewer_calibration, resolve_calibration_profile)
from algorithm.imu_calibrate_cli import (
    ImuCalibrationCLI, compute_motion_rms, default_out_path)
from algorithm.utils import get_average_rotation
from common.errors import CalibrationError
from common.types import (
    CalibrationProgress, CalibrationStep, CalibrationStepResult, DeviceHealth)

__all__ = [
    "ImuCalibrator", "ImuCalibrationCLI", "compute_motion_rms",
    "default_out_path", "find_latest_friston_calibration", "find_latest_hand2mm_calibration",
    "get_average_rotation",
    "CalibrationProfileCatalog", "CalibrationProfileResolution",
    "resolve_calibration_profile", "prepare_viewer_calibration",
]


def find_latest_friston_calibration(folder: Path | str, side: str) -> Path:
    """Find the most recent readable, solver-compatible Friston file for this hand."""
    from algorithm.solver import HandSolverEngine, read_friston_calibration
    from common.errors import CalibrationCompatibilityError

    side = str(side).strip().lower()
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    directory = Path(folder).expanduser().resolve()
    if not directory.is_dir():
        raise CalibrationError(f"Calibration directory does not exist: {directory}")
    candidates = []
    for path in directory.glob("*.json"):
        try:
            candidates.append((path.stat().st_mtime_ns, path.name, path))
        except OSError:
            continue
    for _, _, path in sorted(candidates, reverse=True):
        try:
            payload = read_friston_calibration(path)
            HandSolverEngine.from_payload(side, payload)
        except CalibrationCompatibilityError:
            continue
        return path
    raise CalibrationError(f"No usable Friston {side} IMU calibration JSON in {directory}. Recalibrate this glove.")


def find_latest_hand2mm_calibration(directory: Path | str, side: str) -> Path:
    """Compatibility name for callers that previously selected HAND2mm files."""
    return find_latest_friston_calibration(directory, side)


class ImuCalibrator:
    __slots__ = ("_impl",)

    def __init__(self, side: str, serial_port: str | None = None,
                 registry: Path | str | None = None,
                 config: Path | str | None = None,
                 startup_timeout: float = 30.0, *,
                 workflow: str = "simplified", include_return_open: bool = False):
        # Calibration reads IMU, tactile and status frames only; the type-0x05
        # magnetometer telemetry is not consumed here and :meth:`connect` never
        # waits for it.  The live viewer's own receiver is what caches 0x05.
        self._impl = _Engine(
            side, serial_port, registry, config, startup_timeout,
            workflow=workflow, include_return_open=include_return_open)

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

    def set_workflow(self, workflow: str, *, include_return_open: bool = False) -> CalibrationProgress:
        return self._impl.set_workflow(workflow, include_return_open=include_return_open)

    def recompute(self) -> CalibrationStepResult:
        """Rebuild derived results from retained captures without acquiring data."""
        return self._impl.recompute()

    def skip_return_open(self) -> CalibrationStepResult:
        """Skip the optional final check; saved quality remains unverified."""
        return self._impl.skip_return_open()

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
