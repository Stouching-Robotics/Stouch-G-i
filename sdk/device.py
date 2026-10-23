"""Glove discovery, persistent left/right binding, and dongle recovery.

Two surfaces share this module:

* :class:`DeviceManager` -- the object-oriented view, for code that wants the
  current bindings as one snapshot.
* The module-level functions -- the same registry operations as free functions,
  for the command-line tools and the calibration UI, which need one binding
  changed without constructing a manager around it.

Nothing here opens a glove stream.  Enumerating and probing are safe to call
while another process holds the port: :func:`probe_glove_side` opens it only
briefly and only reads frame headers, and it fails soft when the port is busy.
"""

from __future__ import annotations

from pathlib import Path

from sdk.io.bluetooth_dongle import reboot_bluetooth_dongle
from sdk.io.device_registry import (
    DEFAULT_REGISTRY, GloveDeviceError, GloveHandMismatchError,
    clear_glove_bindings, describe_ports, detect_unbound_glove_links,
    glove_link_sides, link_kind_for_device, link_kind_of, list_matching_ports,
    load_device_registry, probe_glove_side, resolve_both_ports, resolve_glove,
    resolve_glove_port, set_glove_serial, swap_glove_serials)
from sdk.io.devices import DeviceManagerEngine as _Engine
from common.types import DeviceBindings, DeviceInfo

__all__ = [
    "DEFAULT_REGISTRY", "DeviceManager", "GloveDeviceError",
    "GloveHandMismatchError",
    "clear_glove_bindings", "describe_ports", "detect_unbound_glove_links",
    "glove_link_sides",
    "link_kind_for_device", "link_kind_of", "list_matching_ports",
    "load_device_registry", "probe_glove_side", "reboot_bluetooth_dongle",
    "resolve_both_ports", "resolve_glove", "resolve_glove_port",
    "set_glove_serial", "swap_glove_serials",
]


class DeviceManager:
    """The registry as one object: list, bind, swap, validate."""

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
