"""Public offline 16-IMU to HAND2mm 21-joint solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithm.solver import HandSolverEngine as _Engine
from common.types import KeypointFrame, RawImuFrame, SolvedHandFrame


class HandSolver:
    __slots__ = ("_impl",)

    def __init__(self, side: str, calibration: Path | str,
                 geometry: Path | str | None = None):
        self._impl = (_Engine(side, calibration, geometry)
                      if geometry is not None else _Engine(side, calibration))

    @property
    def side(self) -> str:
        return self._impl.side

    def neutral_joints(self) -> np.ndarray:
        return self._impl.neutral_joints()

    @property
    def neutral_imu_xyzw(self) -> np.ndarray:
        """Return the calibrated neutral 16-IMU quaternion array."""

        return self._impl.neutral_imu_xyzw

    @property
    def backend(self) -> dict:
        """Return non-secret runtime identification for UI/session metadata."""

        return dict(self._impl.backend)

    def solve(self, imu_xyzw: np.ndarray,
              valid_mask: np.ndarray | None = None) -> SolvedHandFrame:
        return self._impl.solve(imu_xyzw, valid_mask)

    def process(self, frame: RawImuFrame) -> KeypointFrame:
        """Convert one public raw physical IMU frame into 21 keypoints."""

        return self._impl.process(frame)

    solve_raw = process

    def reset_stream_state(self) -> None:
        self._impl.reset_stream_state()
