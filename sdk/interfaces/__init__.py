"""Lazy public-interface namespace."""

from __future__ import annotations

import importlib


_MODULES = {
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

__all__ = [
    "RawImuStream",
    "SensorStream",
    "TactileStream",
    "DeviceManager",
    "HandSolver",
    "Glove",
    "BimanualGlove",
    "ImuCalibrator",
    "RecordingReplay",
]


def __getattr__(name: str):
    if name in _MODULES:
        return getattr(importlib.import_module(_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
