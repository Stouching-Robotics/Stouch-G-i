"""Resolve left/right STM32 gloves by their stable USB serial numbers.

Schema v2 stores a list of remembered serials per side (``usb_serials``), so
multiple left and multiple right gloves can be remembered.  The first entry is
treated as the primary serial and is mirrored back into ``usb_serial`` on load
so legacy readers (``live_3d.py``) keep working against a single value.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import sys
import time
from typing import Any

from common.usb_cdc import glove_link_kind, validate_channel_to_hand


def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

PROJECT_ROOT = (Path(sys.executable).resolve().parent
                if getattr(sys, "frozen", False)
                else _sdk_root(Path(__file__).resolve().parent))
DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "glove_devices.json"
VALID_SIDES = ("left", "right")


class GloveDeviceError(RuntimeError):
    pass


def _entry_serials(entry: dict) -> list[str]:
    """Return the normalized (uppercased, deduped) serial list for one side entry."""
    raw = entry.get("usb_serials")
    if isinstance(raw, list):
        values = raw
    else:
        single = entry.get("usb_serial")
        values = [single] if single else []
    serials: list[str] = []
    for value in values:
        serial = str(value or "").strip().upper()
        if serial and serial not in serials:
            serials.append(serial)
    return serials


def _serialize_registry(data: dict) -> dict:
    """Build a clean schema-v2 payload from an in-memory (normalized) registry."""
    out: dict[str, Any] = {"schema_version": 2}
    for side in VALID_SIDES:
        entry = dict(data.get(side) or {})
        entry.pop("usb_serial", None)  # v1 single-value field is derived, not stored
        entry["usb_serials"] = [str(s).upper() for s in entry.get("usb_serials", [])]
        out[side] = entry
    return out


def _atomic_write(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def load_device_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_path = Path(path)
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GloveDeviceError(f"Could not read glove device registry {registry_path}: {exc}") from exc
    if data.get("schema_version") not in (1, 2):
        raise GloveDeviceError("glove_devices.json schema_version must be 1 or 2")
    for side in VALID_SIDES:
        entry = data.get(side)
        if not isinstance(entry, dict):
            raise GloveDeviceError(f"glove_devices.json is missing the {side} entry")
        serials = _entry_serials(entry)
        entry["usb_serials"] = serials
        entry["usb_serial"] = serials[0] if serials else ""
        entry["usb_vid"] = int(entry.get("usb_vid", 0x0483))
        entry["usb_pid"] = int(entry.get("usb_pid", 0x5740))
        entry["channel_to_hand"] = list(validate_channel_to_hand(
            entry.get("channel_to_hand", [])))
    overlap = set(data["left"]["usb_serials"]) & set(data["right"]["usb_serials"])
    if overlap:
        raise GloveDeviceError(
            "Left and right gloves must use different usb_serial values; "
            f"duplicated: {', '.join(sorted(overlap))}")
    return data


def link_kind_of(port) -> str | None:
    """Return ``"usb"``/``"bluetooth"`` for one enumerated port, else ``None``."""

    return glove_link_kind(port.vid, port.pid)


def list_matching_ports(kinds: tuple[str, ...] | None = None):
    """Return every USB CDC port that carries a glove stream.

    Both the wired glove (VID 0483 / PID 5740) and the Bluetooth dongle
    (VID 0483 / PID 2013) qualify: the dongle relays the glove's SPP stream
    verbatim, so from the registry's point of view it is an ordinary glove port
    whose serial number happens to belong to the dongle.  ``kinds`` optionally
    restricts the result to ``"usb"`` and/or ``"bluetooth"``.
    """

    import serial.tools.list_ports

    matching = []
    for port in serial.tools.list_ports.comports():
        kind = link_kind_of(port)
        if kind is None or (kinds is not None and kind not in kinds):
            continue
        matching.append(port)
    return matching


def link_kind_for_device(device: str) -> str | None:
    """Return the link kind of a live COM port by device name, else ``None``."""

    wanted = str(device or "").strip()
    if not wanted:
        return None
    for port in list_matching_ports():
        if str(port.device) == wanted:
            return link_kind_of(port)
    return None


def _port_serves_side(port, expected: dict) -> bool:
    """Return whether one enumerated port belongs to this side's glove entry.

    A wired glove must match the side's configured ``usb_vid``/``usb_pid`` — the
    same check as before, kept so a registry pointed at a non-STM32 device is
    still rejected.  A Bluetooth dongle is accepted on its own known VID/PID
    pair instead: the glove behind it is not a USB device at all, so the dongle's
    identity is what the registry binds for that side.
    """

    if link_kind_of(port) == "bluetooth":
        return True
    return port.vid == expected["usb_vid"] and port.pid == expected["usb_pid"]


def probe_glove_side(device: str, timeout_s: float = 2.0) -> str | None:
    """Return the hand side a glove reports over ``device``, or ``None``.

    The firmware encodes its hand side in the ones digit of the version patch,
    so one frame header is enough (see :func:`common.usb_cdc.firmware_hand_side`).
    Only frame headers are parsed — no payload decoding, calibration, or solver
    work happens here.

    This is the fallback for a link no registry entry claims.  A wired glove
    normally resolves by its STM32 serial, but a Bluetooth dongle hides the
    glove behind it: the only identity reaching the host is the dongle's own
    serial plus this firmware-side marker.  Returns ``None`` when the port
    cannot be opened (another process holds it) or no frame arrives in time.
    """

    from common.usb_cdc import UsbCdcFrameParser, firmware_hand_side

    try:
        import serial
    except ImportError:
        return None

    try:
        handle = serial.Serial(device, baudrate=115200, timeout=0.2)
    except (OSError, serial.SerialException):
        return None

    parser = UsbCdcFrameParser()
    try:
        deadline = time.monotonic() + max(float(timeout_s), 0.1)
        while time.monotonic() < deadline:
            try:
                data = handle.read(handle.in_waiting or 1)
            except (OSError, serial.SerialException):
                return None
            if not data:
                continue
            for frame in parser.feed(data):
                side = firmware_hand_side(frame.version)
                if side is not None:
                    return side
    finally:
        try:
            handle.close()
        except Exception:
            pass
    return None


def detect_unbound_glove_links(
        registry_path: str | Path = DEFAULT_REGISTRY,
        timeout_s: float = 2.0) -> dict[str, tuple[str, str]]:
    """Return ``{side: (port, serial)}`` for connected-but-unbound glove links.

    A link is unbound when its USB serial appears in no side's ``usb_serials``
    list, so nothing in the registry claims it — the normal case the first time
    a Bluetooth dongle is plugged in.  Each such link is probed for the hand
    side its firmware reports.  A link that reports no side is left out, as is a
    second link claiming a side an earlier one already reported: with one entry
    per side it would only ever be ambiguous.
    """

    try:
        registry = load_device_registry(registry_path)
    except GloveDeviceError:
        return {}
    bound = {serial for side in VALID_SIDES
             for serial in registry[side]["usb_serials"]}

    discovered: dict[str, tuple[str, str]] = {}
    for port in list_matching_ports():
        serial = str(port.serial_number or "").strip().upper()
        if not serial or serial in bound:
            continue
        side = probe_glove_side(str(port.device), timeout_s)
        if side is None or side in discovered:
            continue
        discovered[side] = (str(port.device), serial)
    return discovered


def _describe_ports(ports) -> str:
    return ", ".join(
        f"{port.device}:{port.serial_number or 'no-serial'}"
        f"({link_kind_of(port) or 'unknown'})"
        for port in ports) or "none"


def resolve_glove(
        side: str,
        registry_path: str | Path = DEFAULT_REGISTRY,
        *,
        allow_single_unbound: bool = False) -> tuple[str, str]:
    """Resolve one bound glove to ``(COM port, matched serial)``.

    A side now remembers a list of serials; any connected STM32 glove whose
    serial is in that list resolves to the side.  Exactly one connected match
    is required (otherwise it is ambiguous).  The ``allow_single_unbound``
    fallback keeps the calibration single-device path working and reports the
    live port's own serial as the matched serial.
    """
    side = str(side).strip().lower()
    if side not in VALID_SIDES:
        raise GloveDeviceError(f"side must be left or right, got {side!r}")
    registry = load_device_registry(registry_path)
    expected = registry[side]
    serials = expected["usb_serials"]
    matches = [
        port for port in list_matching_ports()
        if str(port.serial_number or "").upper() in serials
        and _port_serves_side(port, expected)
    ]
    if not matches:
        present_ports = list_matching_ports()
        # A replacement/reflashed board can legitimately have a different
        # USB serial while it is the only connected STM32 glove.  Calibration
        # passes ``allow_single_unbound=True`` only after the user has selected
        # the hand, so this fallback is never used by live auto-discovery.
        if allow_single_unbound and len(present_ports) == 1:
            candidate = present_ports[0]
            return str(candidate.device), str(candidate.serial_number or "").upper()
        present = _describe_ports(present_ports)
        if not serials:
            raise GloveDeviceError(
                f"No {side} glove is bound (usb_serials is empty); "
                f"detected STM32 devices: {present}")
        raise GloveDeviceError(
            f"Could not find the {side} glove with serials={serials}; "
            f"detected STM32 devices: {present}")
    if len(matches) != 1:
        raise GloveDeviceError(
            f"Multiple devices match the {side} glove serials={serials}; "
            "unplug the extra same-side glove")
    return matches[0].device, str(matches[0].serial_number or "").upper()


def resolve_glove_port(
        side: str,
        registry_path: str | Path = DEFAULT_REGISTRY,
        *,
        allow_single_unbound: bool = False) -> str:
    """Resolve one bound glove to its COM port (see :func:`resolve_glove`)."""
    return resolve_glove(
        side, registry_path, allow_single_unbound=allow_single_unbound)[0]


def resolve_both_ports(registry_path: str | Path = DEFAULT_REGISTRY) -> dict[str, str]:
    result = {side: resolve_glove_port(side, registry_path) for side in VALID_SIDES}
    if result["left"] == result["right"]:
        raise GloveDeviceError("Left and right gloves resolved to the same serial port")
    return result


def set_glove_serial(
        side: str, usb_serial: str,
        registry_path: str | Path = DEFAULT_REGISTRY) -> Path:
    side = str(side).strip().lower()
    if side not in VALID_SIDES:
        raise GloveDeviceError(f"side must be left or right, got {side!r}")
    serial_number = str(usb_serial or "").strip().upper()
    if not serial_number:
        raise GloveDeviceError("usb_serial must not be empty")
    path = Path(registry_path)
    data = load_device_registry(path)
    other = "right" if side == "left" else "left"
    # Reassigning a serial already bound to the other hand moves it over, so a
    # serial never belongs to both sides.  The freshly bound serial is prepended
    # to become the primary (most recently bound) entry.
    data[side]["usb_serials"] = (
        [serial_number]
        + [s for s in data[side]["usb_serials"] if s != serial_number])
    data[other]["usb_serials"] = [
        s for s in data[other]["usb_serials"] if s != serial_number]
    _atomic_write(path, _serialize_registry(data))
    load_device_registry(path)
    return path


def swap_glove_serials(
        registry_path: str | Path = DEFAULT_REGISTRY) -> Path:
    """Atomically swap the persisted left/right USB serial lists."""
    path = Path(registry_path)
    data = load_device_registry(path)
    data["left"]["usb_serials"], data["right"]["usb_serials"] = (
        data["right"]["usb_serials"], data["left"]["usb_serials"])
    _atomic_write(path, _serialize_registry(data))
    load_device_registry(path)
    return path


def clear_glove_bindings(
        registry_path: str | Path = DEFAULT_REGISTRY) -> Path:
    """Forget every remembered glove serial (both sides become unbound).

    Only the serial lists are cleared; the per-side profile fields
    (``usb_vid``/``usb_pid``/``hardware_id``/``channel_to_hand``) are kept.
    """
    path = Path(registry_path)
    data = load_device_registry(path)
    for side in VALID_SIDES:
        data[side]["usb_serials"] = []
    _atomic_write(path, _serialize_registry(data))
    load_device_registry(path)
    return path
