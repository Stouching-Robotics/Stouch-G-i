"""Stouch glove SDK for raw IMU acquisition, hand poses and tactile sensing."""

from __future__ import annotations

from common.errors import (
    CalibrationCompatibilityError,
    CalibrationError,
    CalibrationQualityError,
    DeviceBindingError,
    DeviceBusyError,
    DeviceNotFoundError,
    GloveSdkError,
    StreamClosedError,
    StreamError,
    StreamTimeoutError,
    UnsupportedPlatformError,
)
from common.usb_cdc import display_firmware_version, firmware_real_version
from common.types import (
    BimanualConfig,
    BimanualHealth,
    DeviceHealth,
    GloveConfig,
    GloveInfo,
    HandStatus,
    KeypointFrame,
    LINK_AUTO,
    LINK_BLUETOOTH,
    LINK_USB,
    RawImuFrame,
    Side,
)
from sdk.user_facade import (
    UserBimanualFrame as BimanualFrame,
    UserBimanualGlove as BimanualGlove,
    UserGlove as Glove,
    UserHandFrame as HandFrame,
)

__version__ = "2.4.1"


def get_version() -> str:
    """Return the Python SDK version."""
    return __version__


def __getattr__(name: str):
    """Load the public factories on demand with their original signatures."""
    if name in _INTERFACE_MODULES:
        import importlib
        return getattr(importlib.import_module(_INTERFACE_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_INTERFACE_MODULES = {
    "load_skeleton": "sdk.friston",
    "open_glove": "sdk.api",
    "open_bimanual": "sdk.api",
    "open_raw_glove": "sdk.transport",
    "open_pose_processor": "sdk.solver",
}

__all__ = [
    "__version__",
    "get_version",
    "open_glove", "open_bimanual", "load_skeleton",
    "open_raw_glove", "open_pose_processor", "RawImuFrame", "KeypointFrame",
    "Glove", "BimanualGlove",
    "GloveConfig", "BimanualConfig", "GloveInfo",
    "HandFrame", "BimanualFrame", "HandStatus",
    "DeviceHealth", "BimanualHealth",
    "Side",
    "display_firmware_version", "firmware_real_version",
    "GloveSdkError", "DeviceNotFoundError", "DeviceBindingError",
    "DeviceBusyError", "StreamError", "StreamTimeoutError",
    "StreamClosedError", "CalibrationError", "CalibrationQualityError",
    "CalibrationCompatibilityError", "UnsupportedPlatformError",
]
