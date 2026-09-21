"""Public modular SDK for STM32 HAND2mm gloves.

Runtime and calibration interfaces are loaded lazily so raw IMU/tactile users
do not import the HAND2mm or calibration implementation.
"""

from __future__ import annotations

from common.errors import (
    CalibrationCompatibilityError,
    CalibrationError,
    CalibrationQualityError,
    DeviceBindingError,
    DeviceBusyError,
    DeviceNotFoundError,
    GloveSdkError,
    RecordingError,
    ReplayError,
    StreamClosedError,
    StreamError,
    StreamTimeoutError,
    UnsupportedPlatformError,
)
from common.types import (
    BimanualConfig,
    BimanualFrame,
    BimanualHealth,
    CalibrationProgress,
    CalibrationStep,
    CalibrationStepResult,
    DeviceBindings,
    DeviceHealth,
    DeviceInfo,
    GloveConfig,
    HandFrame,
    HandStatus,
    ImuFrame,
    KeypointFrame,
    RawImuFrame,
    RecordingResult,
    RecordingInfo,
    RecordingStatus,
    ReplayResult,
    Side,
    SolvedHandFrame,
    TactileFrame,
)

__version__ = "2.0.7"


def get_version() -> str:
    """Return the SDK version string (e.g. ``"1.2.0"``)."""
    return __version__


_INTERFACE_NAMES = (
    "RawImuStream",
    "SensorStream",
    "TactileStream",
    "DeviceManager",
    "HandSolver",
    "Glove",
    "BimanualGlove",
    "ImuCalibrator",
    "RecordingReplay",
)
_INTERFACE_MODULES = {
    "RawImuStream": "sdk.interfaces.sensors",
    "SensorStream": "sdk.interfaces.sensors",
    "TactileStream": "sdk.interfaces.sensors",
    "DeviceManager": "sdk.interfaces.device",
    "HandSolver": "sdk.interfaces.solver",
    "Glove": "sdk.interfaces.glove",
    "BimanualGlove": "sdk.interfaces.bimanual",
    "ImuCalibrator": "sdk.interfaces.calibration",
    "RecordingReplay": "sdk.interfaces.replay",
}
_LEGACY_STREAM_NAMES = {
    "ImuStream": ("sdk.compat.imu", "ImuStream"),
    "PhysicalFrame": ("sdk.compat.imu", "PhysicalFrame"),
    "StreamImuFrame": ("sdk.compat.imu", "ImuFrame"),
    "StreamTactileFrame": ("sdk.compat.tactile", "TactileFrame"),
}
_DEPRECATED_KEYPOINT_NAMES = {
    "KeypointsFrame",
    "KeypointSolver",
    "create_solver",
    "uncalibrated_solver",
    "backend_metadata",
}

__all__ = [
    "__version__",
    "get_version",
    *_INTERFACE_NAMES,
    "Side",
    "DeviceInfo", "DeviceBindings", "DeviceHealth",
    "GloveConfig", "BimanualConfig",
    "RawImuFrame", "ImuFrame", "TactileFrame", "KeypointFrame",
    "RawImuStream", "SensorStream", "TactileStream",
    "PhysicalFrame", "ImuStream", "StreamImuFrame",
    "TactileStream", "StreamTactileFrame",
    "HandStatus", "SolvedHandFrame", "HandFrame", "BimanualFrame",
    "BimanualHealth",
    "CalibrationStep", "CalibrationStepResult", "CalibrationProgress",
    "RecordingStatus", "RecordingResult", "RecordingInfo", "ReplayResult",
    "GloveSdkError", "DeviceNotFoundError", "DeviceBindingError",
    "DeviceBusyError", "StreamError", "StreamTimeoutError",
    "StreamClosedError", "CalibrationError",
    "CalibrationQualityError", "CalibrationCompatibilityError",
    "RecordingError", "ReplayError", "UnsupportedPlatformError",
]


def __getattr__(name: str):
    if name in _INTERFACE_NAMES:
        import importlib
        return getattr(importlib.import_module(_INTERFACE_MODULES[name]), name)
    if name in _LEGACY_STREAM_NAMES:
        import importlib
        module_name, attribute = _LEGACY_STREAM_NAMES[name]
        return getattr(importlib.import_module(module_name), attribute)
    if name in _DEPRECATED_KEYPOINT_NAMES:
        import warnings
        warnings.warn(
            f"runtime.{name} is deprecated; use "
            f"sdk.compat.keypoints.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        from sdk.compat import keypoints
        return getattr(keypoints, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
