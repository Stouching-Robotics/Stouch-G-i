"""Public offline 16-IMU to HAND2mm 21-joint solver."""

from __future__ import annotations

from pathlib import Path
import copy

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

    def set_root_limits_enabled(self, enabled: bool) -> None:
        """Enable four-finger twist/spread and flexion limits (default True)."""
        self._impl.set_root_limits_enabled(enabled)

    def set_contact_controls(self, enabled: bool, blend: float = 1.0) -> None:
        """Control original runtime attraction (including solved recordings)."""
        self._impl.set_contact_controls(enabled, blend)

    @property
    def pose_recording_metadata(self) -> dict:
        """Snapshot the selected solver mode/profile for replay provenance."""
        shape = self._impl.hand_config.param_shape_calib or {}
        return {
            "pose_postprocess": self.backend.get('pose_postprocess', 'legacy-contact'),
            "gesture_profile": copy.deepcopy(shape.get('gesture_retarget')),
            "motion_refit_profile": copy.deepcopy(shape.get('sensor_motion_refit')),
        }

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

    @property
    def mesh_vertices_m(self) -> np.ndarray | None:
        """The newest solved MANO mesh, wrist-centred, ``(778, 3)`` float32.

        ``None`` until the first solve.  This is the mesh the renderer draws; it
        is not keyed to any one :class:`~common.types.HandFrame`, so a caller
        painting at 30 Hz reads the current one rather than paying for a copy of
        every frame the solver produced.
        """

        for candidate in (self._impl, getattr(self._impl, "_solver", None)):
            vertices = getattr(candidate, "last_mesh_vertices_m", None)
            if vertices is None:
                continue
            value = np.asarray(vertices, dtype=np.float32)
            if value.shape == (778, 3) and np.isfinite(value).all():
                return value.copy()
        return None
