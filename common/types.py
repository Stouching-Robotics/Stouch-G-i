"""Versioned public data contracts used by every SDK interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


Side = Literal["left", "right"]


def _side(value: str) -> Side:
    normalized = str(value).strip().lower()
    if normalized not in ("left", "right"):
        raise ValueError(f"side must be left or right, got {value!r}")
    return normalized  # type: ignore[return-value]


def _array(value, shape, dtype, name):
    result = np.asarray(value, dtype=dtype).reshape(shape).copy()
    if not np.all(np.isfinite(result)) and result.dtype.kind == "f":
        # NaN is meaningful for sensor age/tactile absence, so callers decide
        # whether it is acceptable. Shape and ownership are enforced here.
        pass
    return result


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    serial_number: str
    vid: int
    pid: int
    location: str = ""
    bound_side: Side | None = None
    # "usb" for a directly wired glove, "bluetooth" for a glove reached through
    # the BP101Y dongle.  The wire protocol is identical; only the USB PID and
    # the meaning of ``serial_number`` (glove vs dongle) differ.
    link_kind: str = "usb"


@dataclass(frozen=True)
class DeviceBindings:
    left_serial: str
    right_serial: str
    left_port: str | None = None
    right_port: str | None = None
    # Full remembered lists (schema v2); the *_serial fields remain the primary
    # (most recently bound) entry for backwards compatibility.
    left_serials: tuple[str, ...] = ()
    right_serials: tuple[str, ...] = ()


@dataclass(frozen=True)
class GloveConfig:
    side: Side
    calibration: Path | str
    serial_number: str | None = None
    registry: Path | str | None = None
    sample_rate_hz: int = 80
    include_tactile: bool = True
    smoothing_ms: float = 35.0
    recording_root: Path | str | None = None

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "calibration", Path(self.calibration))
        if self.registry is not None:
            object.__setattr__(self, "registry", Path(self.registry))
        if self.recording_root is not None:
            object.__setattr__(self, "recording_root", Path(self.recording_root))
        if int(self.sample_rate_hz) <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if float(self.smoothing_ms) < 0.0:
            raise ValueError("smoothing_ms must not be negative")


@dataclass(frozen=True)
class BimanualConfig:
    left: GloveConfig
    right: GloveConfig
    output_root: Path | str | None = None
    display_separation_m: float = 0.30

    def __post_init__(self):
        if self.left.side != "left" or self.right.side != "right":
            raise ValueError("BimanualConfig requires left and right GloveConfig objects")
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root))
        if float(self.display_separation_m) <= 0.0:
            raise ValueError("display_separation_m must be positive")


@dataclass(frozen=True)
class DeviceHealth:
    connected: bool
    ready_imu_count: int
    missing_imus: tuple[str, ...] = ()
    stream_fps: float = 0.0
    message: str = ""
    last_error: str | None = None


@dataclass(frozen=True)
class BimanualHealth:
    left: DeviceHealth
    right: DeviceHealth
    synchronization_error_ms: float | None = None

    @property
    def connected(self) -> bool:
        return self.left.connected and self.right.connected


@dataclass(frozen=True)
class HandStatus:
    safety_intervened: bool = False
    active_contact: str | None = None
    kinematics: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawImuFrame:
    """One decoded STM32 IMU packet in physical channel order.

    The quaternion order is XYZW.  No physical-to-hand remapping, axis
    conversion, filtering, calibration, or hand-pose solving has been applied.
    """

    sequence: int
    device_timestamp_us: int
    host_timestamp_us: int
    quaternions_xyzw: np.ndarray
    present_mask: np.ndarray
    valid_mask: np.ndarray
    mag_ready: bool = True

    def __post_init__(self):
        object.__setattr__(self, "quaternions_xyzw", _array(
            self.quaternions_xyzw, (16, 4), np.float64,
            "quaternions_xyzw"))
        object.__setattr__(self, "present_mask", _array(
            self.present_mask, (16,), bool, "present_mask"))
        object.__setattr__(self, "valid_mask", _array(
            self.valid_mask, (16,), bool, "valid_mask"))


@dataclass(frozen=True)
class ImuFrame:
    sequence: int
    timestamp_us: int
    quaternions_xyzw: np.ndarray
    valid_mask: np.ndarray
    sensor_age_s: np.ndarray
    rejected_mask: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "quaternions_xyzw", _array(
            self.quaternions_xyzw, (16, 4), np.float64, "quaternions_xyzw"))
        object.__setattr__(self, "valid_mask", _array(
            self.valid_mask, (16,), bool, "valid_mask"))
        object.__setattr__(self, "sensor_age_s", _array(
            self.sensor_age_s, (16,), np.float32, "sensor_age_s"))
        object.__setattr__(self, "rejected_mask", _array(
            self.rejected_mask, (16,), bool, "rejected_mask"))


@dataclass(frozen=True)
class TactileFrame:
    sequence: int
    timestamp_us: int
    samples: np.ndarray
    processed: bool = False

    def __post_init__(self):
        object.__setattr__(self, "samples", _array(
            self.samples, (16, 16), np.float32, "samples"))

    @property
    def scan_time_us(self) -> int:
        """Firmware scan-time field retained as a descriptive alias."""

        return self.timestamp_us


@dataclass(frozen=True)
class SolvedHandFrame:
    side: Side
    joints_m: np.ndarray
    status: HandStatus

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "joints_m", _array(
            self.joints_m, (21, 3), np.float32, "joints_m"))


@dataclass(frozen=True)
class KeypointFrame:
    """Versioned output of the protected raw-IMU to 21-keypoint pipeline."""

    side: Side
    sequence: int
    timestamp_us: int
    joints_m: np.ndarray
    imu_xyzw: np.ndarray
    valid_mask: np.ndarray
    sensor_age_s: np.ndarray
    rejected_mask: np.ndarray
    status: HandStatus

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "joints_m", _array(
            self.joints_m, (21, 3), np.float32, "joints_m"))
        object.__setattr__(self, "imu_xyzw", _array(
            self.imu_xyzw, (16, 4), np.float64, "imu_xyzw"))
        object.__setattr__(self, "valid_mask", _array(
            self.valid_mask, (16,), bool, "valid_mask"))
        object.__setattr__(self, "sensor_age_s", _array(
            self.sensor_age_s, (16,), np.float32, "sensor_age_s"))
        object.__setattr__(self, "rejected_mask", _array(
            self.rejected_mask, (16,), bool, "rejected_mask"))

    @property
    def keypoints_21_m(self) -> np.ndarray:
        """Descriptive alias for :attr:`joints_m`."""

        return self.joints_m.copy()


@dataclass(frozen=True)
class HandFrame:
    side: Side
    sequence: int
    timestamp_us: int
    imu_xyzw: np.ndarray
    imu_valid: np.ndarray
    sensor_age_s: np.ndarray
    joints_raw_m: np.ndarray
    joints_smoothed_m: np.ndarray
    tactile: np.ndarray | None
    status: HandStatus
    raw_imu_xyzw: np.ndarray | None = None
    raw_imu_present: np.ndarray | None = None
    raw_imu_valid: np.ndarray | None = None
    raw_device_timestamp_us: int | None = None
    tactile_raw: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "imu_xyzw", _array(
            self.imu_xyzw, (16, 4), np.float64, "imu_xyzw"))
        object.__setattr__(self, "imu_valid", _array(
            self.imu_valid, (16,), bool, "imu_valid"))
        object.__setattr__(self, "sensor_age_s", _array(
            self.sensor_age_s, (16,), np.float32, "sensor_age_s"))
        object.__setattr__(self, "joints_raw_m", _array(
            self.joints_raw_m, (21, 3), np.float32, "joints_raw_m"))
        object.__setattr__(self, "joints_smoothed_m", _array(
            self.joints_smoothed_m, (21, 3), np.float32, "joints_smoothed_m"))
        if self.tactile is not None:
            object.__setattr__(self, "tactile", _array(
                self.tactile, (16, 16), np.float32, "tactile"))
        if self.raw_imu_xyzw is not None:
            object.__setattr__(self, "raw_imu_xyzw", _array(
                self.raw_imu_xyzw, (16, 4), np.float64, "raw_imu_xyzw"))
        if self.raw_imu_present is not None:
            object.__setattr__(self, "raw_imu_present", _array(
                self.raw_imu_present, (16,), bool, "raw_imu_present"))
        if self.raw_imu_valid is not None:
            object.__setattr__(self, "raw_imu_valid", _array(
                self.raw_imu_valid, (16,), bool, "raw_imu_valid"))
        if self.tactile_raw is not None:
            object.__setattr__(self, "tactile_raw", _array(
                self.tactile_raw, (16, 16), np.float32, "tactile_raw"))


@dataclass(frozen=True)
class BimanualFrame:
    sequence: int
    timestamp_us: int
    left: HandFrame
    right: HandFrame
    synchronization_error_ms: float


@dataclass(frozen=True)
class CalibrationStep:
    id: str
    title: str
    action: Literal["capture", "compute"]
    stage: Literal["root", "installation", "contact"]


@dataclass(frozen=True)
class CalibrationStepResult:
    step_id: str
    ok: bool
    message: str


@dataclass(frozen=True)
class CalibrationProgress:
    current_index: int
    total_steps: int
    completed_step_ids: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class RecordingStatus:
    state: Literal["disabled", "idle", "recording", "saved"]
    take_index: int
    frame_count: int
    output_path: Path | None = None


@dataclass(frozen=True)
class RecordingResult:
    parquet_path: Path | None
    session_path: Path | None
    take_index: int
    frame_count: int
    duration_s: float
    sample_fps: float


@dataclass(frozen=True)
class RecordingInfo:
    parquet_path: Path
    session_path: Path | None
    mode: str
    sides: tuple[Side, ...]
    frame_count: int
    sample_fps: float
    has_imu: bool
    has_tactile: bool
    tactile_valid_frames: int = 0
    tactile_nonzero_samples: int = 0
    has_raw_imu: bool = False
    has_raw_tactile: bool = False

    @property
    def has_tactile_signal(self) -> bool:
        return self.tactile_nonzero_samples > 0


@dataclass(frozen=True)
class ReplayResult:
    input_path: Path
    output_path: Path
    frame_count: int
    sample_fps: float
