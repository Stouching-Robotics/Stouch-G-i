"""Public single-glove runtime interface."""

from __future__ import annotations

from runtime.glove import GloveEngine as _Engine
from common.types import (
    DeviceHealth, GloveConfig, HandFrame,
    RecordingResult, RecordingStatus)


class Glove:
    __slots__ = ("_impl",)

    def __init__(self, config: GloveConfig):
        self._impl = _Engine(config)

    def start(self) -> "Glove":
        self._impl.start()
        return self

    def read(self, timeout: float | None = None) -> HandFrame:
        return self._impl.read(timeout)

    def latest(self) -> HandFrame | None:
        return self._impl.latest()

    def health(self) -> DeviceHealth:
        return self._impl.health()

    @property
    def backend(self) -> dict:
        return self._impl.backend

    def neutral_joints(self):
        return self._impl.neutral_joints()

    def start_recording(self, name: str | None = None) -> RecordingStatus:
        return self._impl.start_recording(name)

    def stop_recording(self) -> RecordingResult:
        return self._impl.stop_recording()

    @property
    def recording_status(self) -> RecordingStatus:
        return self._impl.recording_status()

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "Glove":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
