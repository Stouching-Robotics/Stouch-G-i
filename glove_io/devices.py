"""Internal adapter around the current persistent device registry."""

from __future__ import annotations

from pathlib import Path

from common.errors import DeviceBindingError, DeviceNotFoundError
from common.types import DeviceBindings, DeviceInfo
from glove_io.device_registry import (
    DEFAULT_REGISTRY,
    GloveDeviceError,
    clear_glove_bindings,
    list_matching_ports,
    load_device_registry,
    resolve_both_ports,
    resolve_glove,
    set_glove_serial,
    swap_glove_serials,
)


class DeviceManagerEngine:
    def __init__(self, registry: Path | str = DEFAULT_REGISTRY):
        self.registry = Path(registry)

    def list_devices(self) -> list[DeviceInfo]:
        try:
            registry = load_device_registry(self.registry)
        except GloveDeviceError as exc:
            raise DeviceBindingError(str(exc)) from exc
        serial_to_side = {
            serial: side
            for side in ("left", "right")
            for serial in registry[side]["usb_serials"]
        }
        return [
            DeviceInfo(
                device=str(port.device),
                serial_number=str(port.serial_number or "").upper(),
                vid=int(port.vid or 0),
                pid=int(port.pid or 0),
                location=str(port.location or ""),
                bound_side=serial_to_side.get(
                    str(port.serial_number or "").upper()),
            )
            for port in list_matching_ports()
        ]

    def get_bindings(self, resolve_ports: bool = False) -> DeviceBindings:
        try:
            registry = load_device_registry(self.registry)
            ports = resolve_both_ports(self.registry) if resolve_ports else {}
        except GloveDeviceError as exc:
            if resolve_ports:
                raise DeviceNotFoundError(str(exc)) from exc
            raise DeviceBindingError(str(exc)) from exc
        return DeviceBindings(
            left_serial=registry["left"]["usb_serials"][0]
            if registry["left"]["usb_serials"] else "",
            right_serial=registry["right"]["usb_serials"][0]
            if registry["right"]["usb_serials"] else "",
            left_port=ports.get("left"),
            right_port=ports.get("right"),
            left_serials=tuple(registry["left"]["usb_serials"]),
            right_serials=tuple(registry["right"]["usb_serials"]),
        )

    def bind(self, side: str, serial_number: str) -> DeviceBindings:
        try:
            set_glove_serial(side, serial_number, self.registry)
        except GloveDeviceError as exc:
            raise DeviceBindingError(str(exc)) from exc
        return self.get_bindings(resolve_ports=False)

    def swap_bindings(self) -> DeviceBindings:
        try:
            swap_glove_serials(self.registry)
        except GloveDeviceError as exc:
            raise DeviceBindingError(str(exc)) from exc
        return self.get_bindings(resolve_ports=False)

    def clear_bindings(self) -> DeviceBindings:
        try:
            clear_glove_bindings(self.registry)
        except GloveDeviceError as exc:
            raise DeviceBindingError(str(exc)) from exc
        return self.get_bindings(resolve_ports=False)

    def validate_bimanual(self) -> DeviceBindings:
        return self.get_bindings(resolve_ports=True)

    def resolve_port(self, side: str) -> tuple[str, str]:
        normalized = str(side).strip().lower()
        if normalized not in ("left", "right"):
            raise ValueError("side must be left or right")
        try:
            port, serial = resolve_glove(normalized, self.registry)
        except GloveDeviceError as exc:
            raise DeviceNotFoundError(str(exc)) from exc
        return str(port), str(serial)
