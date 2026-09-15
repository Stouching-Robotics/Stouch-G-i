"""Lazy public-interface namespace."""

from __future__ import annotations

import importlib


_MODULES = {
    "RawImuStream": "runtime.interfaces.sensors",
    "SensorStream": "runtime.interfaces.sensors",
    "TactileStream": "runtime.interfaces.sensors",
    "DeviceManager": "runtime.interfaces.device",
    "HandSolver": "runtime.interfaces.solver",
    "Glove": "runtime.interfaces.glove",
    "BimanualGlove": "runtime.interfaces.bimanual",
    "ImuCalibrator": "runtime.interfaces.calibration",
    "RecordingReplay": "runtime.interfaces.replay",
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
