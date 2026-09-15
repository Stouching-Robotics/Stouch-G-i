"""Public device discovery and persistent left/right binding interface."""

from __future__ import annotations

from pathlib import Path

from glove_io.devices import DeviceManagerEngine as _Engine
from common.types import DeviceBindings, DeviceInfo


class DeviceManager:
    __slots__ = ("_impl",)

    def __init__(self, registry: Path | str | None = None):
        self._impl = _Engine() if registry is None else _Engine(registry)

    def list_devices(self) -> list[DeviceInfo]:
        return self._impl.list_devices()

    def get_bindings(self, resolve_ports: bool = False) -> DeviceBindings:
        return self._impl.get_bindings(resolve_ports)

    def bind(self, side: str, serial_number: str) -> DeviceBindings:
        return self._impl.bind(side, serial_number)

    def swap_bindings(self) -> DeviceBindings:
        return self._impl.swap_bindings()

    def validate_bimanual(self) -> DeviceBindings:
        return self._impl.validate_bimanual()

    def resolve_port(self, side: str) -> tuple[str, str]:
        """Resolve one bound hand without requiring the other glove."""

        return self._impl.resolve_port(side)
