"""Public offline 16-IMU to Friston 21-joint solver."""

from __future__ import annotations

from pathlib import Path
from threading import RLock

import numpy as np

from algorithm.solver import HandSolverEngine as _Engine
from common.types import KeypointFrame, RawImuFrame, SolvedHandFrame


class HandSolver:
    __slots__ = ("_impl", "_stream_lock", "_connection")

    def __init__(self, side: str, calibration: Path | str,
                 geometry: Path | str | None = None):
        self._impl = (_Engine(side, calibration, geometry)
                      if geometry is not None else _Engine(side, calibration))
        self._stream_lock = RLock()
        self._connection = (None, None)

    @property
    def side(self) -> str:
        return self._impl.side

    def neutral_joints(self) -> np.ndarray:
        return self._impl.neutral_joints()

    def set_root_limits_enabled(self, enabled: bool) -> None:
        """Enable four-finger twist/spread and flexion limits (default True)."""
        self._impl.set_root_limits_enabled(enabled)

    def set_root_limits_blending_enabled(self, enabled: bool) -> None:
        """Relax joint limits linearly toward opposition (default True)."""
        self._impl.set_root_limits_blending_enabled(enabled)

    def set_fist_correction_enabled(self, enabled: bool) -> None:
        """Correct the four fingers near the sampled fist; preserve independent thumb motion."""
        self._impl.set_fist_correction_enabled(enabled)

    def set_contact_controls(self, enabled: bool, blend: float = 1.0) -> None:
        """Control original runtime attraction (including solved recordings)."""
        self._impl.set_contact_controls(enabled, blend)

    @property
    def pose_recording_metadata(self) -> dict:
        """Snapshot the selected solver mode/profile for replay provenance."""
        return self._impl.pose_recording_metadata

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
        """Solve logical, axis-corrected IMUs in reference sensor order."""
        return self._impl.solve(imu_xyzw, valid_mask)

    def process(self, frame: RawImuFrame) -> KeypointFrame:
        """Solve one physical IMU frame into 16 joint rotations and 21 keypoints."""

        with self._stream_lock:
            connection = (getattr(frame, "_stream_id", None),
                          getattr(frame, "connection_generation", None))
            if connection != self._connection:
                self._impl.reset_stream_state()
                self._connection = connection
            return self._impl.process(frame)

    solve_raw = process

    def reset_stream_state(self) -> None:
        """Clear temporal state before changing to another acquisition session."""
        with self._stream_lock:
            self._impl.reset_stream_state()
            self._connection = (None, None)

    @property
    def mesh_vertices_m(self) -> np.ndarray | None:
        """The newest solved Friston mesh in metres, or None before a solve."""
        return self._impl.mesh_vertices_m


def open_pose_processor(side: str, calibration: Path | str) -> HandSolver:
    """Load a Friston calibration for offline or live IMU pose processing."""
    return HandSolver(side, Path(calibration).expanduser())
