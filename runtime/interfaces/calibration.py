"""Public step-driven HAND2mm IMU calibration interface."""

from __future__ import annotations

from pathlib import Path

from algorithm.calibration_session import ImuCalibratorEngine as _Engine
from common.types import (
    CalibrationProgress, CalibrationStep, CalibrationStepResult, DeviceHealth)


class ImuCalibrator:
    __slots__ = ("_impl",)

    def __init__(self, side: str, serial_port: str | None = None,
                 registry: Path | str | None = None,
                 config: Path | str | None = None,
                 startup_timeout: float = 30.0):
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
