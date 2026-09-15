"""Public synchronized bimanual runtime and repeated recording interface."""

from __future__ import annotations

from runtime.bimanual import BimanualEngine as _Engine
from common.types import (
    BimanualConfig, BimanualFrame, BimanualHealth,
    RecordingResult, RecordingStatus)


class BimanualGlove:
    __slots__ = ("_impl",)

    def __init__(self, config: BimanualConfig):
        self._impl = _Engine(config)

    def start(self) -> "BimanualGlove":
        self._impl.start()
        return self

    def read(self, timeout: float | None = None) -> BimanualFrame:
        return self._impl.read(timeout)

    def health(self) -> BimanualHealth:
        return self._impl.health()

    def start_recording(self, name: str | None = None) -> RecordingStatus:
        return self._impl.start_recording(name)

    def stop_recording(self) -> RecordingResult:
        return self._impl.stop_recording()

    @property
    def recording_status(self) -> RecordingStatus:
        return self._impl.recording_status()

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "BimanualGlove":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
