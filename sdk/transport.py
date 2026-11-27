"""链路接缝：有线 USB 与蓝牙 dongle 对调用方透明。

两条链路跑的是**同一套线协议**（14 字节瘦头
``A5 5A | M m P | type | seq | ts_us | len | payload | CRC16``、0x01–0x05 五种帧、
姿态与压力各 60 fps）。差别只在物理层：

============  =====================  ============
链路           主机侧枚举为            VID:PID
============  =====================  ============
有线 1.2.14    STM32 USB CDC          0483:5740
蓝牙 2.0.7     BP101Y dongle 中继      0483:2013
============  =====================  ============

所以“两个数据接口”不是两套解码：蓝牙 dongle 把 SPP 流逐字节中继上来，主机侧看到的
是一个协议完全一致的手套口。:func:`glove_link_kind` 是区分二者的唯一处。

**固件行为（两版接口文档一致）**：手套一旦检测到 USB 连着，姿态与压力全部走 USB，
蓝牙侧一帧都不发。两条链路不能同时工作——想走蓝牙必须拔掉 USB 线。所以
``link="auto"`` 在两者都在场时选有线，并把原因写进 :attr:`LinkSpec.note`。

挑选逻辑没有复用 :func:`glove_io.device_registry.resolve_glove`，是刻意的：那个函数
用**注册表文件里存的** ``usb_vid``/``usb_pid`` 校验端口归属，而注册表是按有线手套注册
的，于是挂在蓝牙 dongle 上的手会被判成"不匹配"。枚举（:func:`list_matching_ports`
已同时认 0483:5740 与 0483:2013）和定手型（:func:`detect_unbound_glove_links`）
仍然全部复用，只有归属判定改成按**当前真实端口**的链路种类走。
"""

from __future__ import annotations

from dataclasses import dataclass
from os import path
from pathlib import Path
from threading import RLock
from typing import Callable

from common.errors import DeviceBindingError, DeviceNotFoundError, StreamClosedError
from common.types import (
    LINK_AUTO, LINK_BLUETOOTH, LINK_USB, GloveConfig, GloveInfo, RawImuFrame, _side)
from common.usb_cdc import (
    BLUETOOTH_DONGLE_PID, BLUETOOTH_DONGLE_VID, STM32_USB_PID, STM32_USB_VID,
    firmware_hand_side)
from glove_io.device_registry import (
    DEFAULT_REGISTRY, GloveDeviceError, describe_ports,
    detect_unbound_glove_links, link_kind_of, list_matching_ports,
    load_device_registry, probe_link_sides)
from glove_io.streams import RawImuStream
from sdk import get_version
from sdk.mag import mag_available

# 每条链路的规范 VID/PID。必须用这张表而不是端口的 vid/pid 直接透传：
# ``_UsbSensorTransport`` 靠 ``usb_pid == BLUETOOTH_DONGLE_PID`` 判定
# ``_is_bluetooth``，判错的后果是半双工蓝牙链路上仍开自动延迟探测（把链路打爆），
# 且丢掉 2 秒静默自动重连。
_LINK_IDS: dict[str, tuple[int, int]] = {
    LINK_USB: (STM32_USB_VID, STM32_USB_PID),
    LINK_BLUETOOTH: (BLUETOOTH_DONGLE_VID, BLUETOOTH_DONGLE_PID),
}
_LINK_LABELS = {LINK_USB: "USB", LINK_BLUETOOTH: "Bluetooth"}

BOUND_BY_SERIAL = "serial"
BOUND_BY_BINDING = "binding"
BOUND_BY_FIRMWARE = "firmware-side"
BOUND_BY_PORT = "port"


@dataclass(frozen=True)
class LinkSpec:
    """One resolved glove link, ready to hand to the acquisition transport."""

    port: str
    serial_number: str
    vid: int
    pid: int
    kind: str
    resolved_by: str
    note: str = ""
    port_identity: tuple[int, int, str, str | None] | None = None

    @property
    def label(self) -> str:
        where = f"{_LINK_LABELS.get(self.kind, self.kind)} {self.port}"
        return f"{where} [{self.serial_number}]" if self.serial_number else where

    @property
    def is_bluetooth(self) -> bool:
        return self.kind == LINK_BLUETOOTH


@dataclass(frozen=True)
class _RawLinkConfig:
    side: str
    link: str
    serial_number: str | None
    registry: Path | str | None


def _normalise_kind(link: str | None) -> str:
    value = str(link or LINK_AUTO).strip().lower()
    if value not in (LINK_AUTO, LINK_USB, LINK_BLUETOOTH):
        raise ValueError(
            f"link must be one of {LINK_AUTO}, {LINK_USB}, {LINK_BLUETOOTH}; "
            f"got {link!r}")
    return value


def _spec_from_port(port, kind: str, resolved_by: str, note: str = "") -> LinkSpec:
    vid, pid = _LINK_IDS[kind]
    return LinkSpec(
        port=str(port.device),
        serial_number=str(port.serial_number or "").strip().upper(),
        vid=vid,
        pid=pid,
        kind=kind,
        resolved_by=resolved_by,
        note=note,
        port_identity=_port_identity(port),
    )


def _usb_priority_note(kind: str, ports) -> str:
    """Explain the wired-vs-wireless interaction the firmware imposes."""

    wired = [port for port in ports if link_kind_of(port) == LINK_USB]
    if kind == LINK_BLUETOOTH and wired:
        return ("a wired glove is also connected; the firmware stops sending over "
                "Bluetooth while USB is attached, so wireless may stay silent")
    return ""


def resolve_link(config: GloveConfig | _RawLinkConfig, link: str | None = None) -> LinkSpec:
    """Resolve one hand to a concrete link, wired or Bluetooth.

    ``link`` defaults to :attr:`GloveConfig.link`.  ``"auto"`` looks for the
    registry binding first, then asks any unbound glove link which hand its
    firmware reports (the normal path the first time a Bluetooth dongle is
    plugged in).  ``"usb"`` / ``"bluetooth"`` restrict the search to one link
    kind and fail loudly rather than silently using the other.
    """

    kind_wanted = _normalise_kind(link if link is not None else config.link)
    registry = Path(config.registry) if config.registry else Path(DEFAULT_REGISTRY)
    wanted_kinds = None if kind_wanted == LINK_AUTO else (kind_wanted,)
    ports = list_matching_ports(wanted_kinds)

    scope = "any glove link" if kind_wanted == LINK_AUTO else f"a {kind_wanted} link"
    if not ports:
        raise DeviceNotFoundError(
            f"No {scope} is connected for the {config.side} hand. "
            f"Detected glove ports: {describe_ports(list_matching_ports())}. "
            "A wireless glove needs its dongle plugged in and the USB cable "
            "unplugged (the firmware sends nothing over Bluetooth while USB is "
            "attached).")

    if config.serial_number:
        # Pin the link identity here. The pose consumer checks the hand on
        # frames from the actual serial handle, including after a reconnect.
        wanted = str(config.serial_number).strip().upper()
        matches = [p for p in ports
                   if str(p.serial_number or "").strip().upper() == wanted]
        if len(matches) != 1:
            raise DeviceNotFoundError(
                f"Expected one glove link with serial {wanted}, found "
                f"{len(matches)}. Detected: {describe_ports(ports)}")
        chosen = matches[0]
        kind = link_kind_of(chosen)
        assert kind is not None  # list_matching_ports already filtered to gloves
        return _spec_from_port(chosen, kind, BOUND_BY_SERIAL,
                               _usb_priority_note(kind, ports))

    # 1) The hand the link itself names, and the registry's binding for a link
    #    that names none.  The firmware wins over the filing on purpose: a
    #    serial can be filed under the wrong side, and driving a left glove as
    #    the right hand mirrors every pose and reads every pressure region from
    #    the wrong channels without failing once (see
    #    :func:`glove_io.device_registry.glove_link_sides`).
    bound = _bound_serials(config.side, registry)
    # Through the shared prober: it runs the ports in parallel and caches the
    # answers for the process, so a silent link costs one timeout rather than
    # one per lookup (see glove_io/device_registry.probe_link_sides).
    named = probe_link_sides(ports)

    def link_side(port) -> str | None:
        return named.get(str(port.device)) or (
            config.side
            if str(port.serial_number or "").strip().upper() in bound else None)

    claiming = [port for port in ports if link_side(port) == config.side]
    # A link the registry filed for this hand beats one that merely claims it,
    # so a second unfiled glove naming the same hand is not an ambiguity.
    filed = [port for port in claiming
             if str(port.serial_number or "").strip().upper() in bound]
    matches = filed or claiming
    if len(matches) > 1:
        raise DeviceNotFoundError(
            f"Multiple glove links match the {config.side} hand's bound serials "
            f"{sorted(bound)}: {describe_ports(matches)}. Unplug the extra device.")
    if len(matches) == 1:
        chosen = matches[0]
        kind = link_kind_of(chosen)
        assert kind is not None
        # Which of the two answered.  A link the firmware named is reported that
        # way even when a binding exists, so ``resolved_by`` keeps meaning "what
        # actually decided" rather than "what was consulted".
        resolved_by = (BOUND_BY_FIRMWARE if str(chosen.device) in named
                       else BOUND_BY_BINDING)
        return _spec_from_port(chosen, kind, resolved_by,
                               _usb_priority_note(kind, ports))

    # 2) Nothing bound yet: let the firmware's hand marker identify the link.
    #    A Bluetooth dongle hides the glove behind it, so the dongle's own
    #    serial is all the registry could ever bind; this probe is what makes
    #    first-time wireless setup work.
    unbound = detect_unbound_glove_links(registry)
    if config.side in unbound:
        port_name, serial = unbound[config.side]
        chosen = next((p for p in ports if str(p.device) == port_name), None)
        if chosen is not None:
            kind = link_kind_of(chosen)
            assert kind is not None
            return _spec_from_port(chosen, kind, BOUND_BY_FIRMWARE,
                                   _usb_priority_note(kind, ports))

    # Include links without USB serials, which can only identify themselves
    # by the firmware marker and cannot be fixed by binding a serial number.
    conflicting = [
        f"{port.device} ({port.serial_number or 'no-serial'}) reports the "
        f"{named[str(port.device)]} hand"
        for port in ports
        if named.get(str(port.device)) not in (None, config.side)
    ]
    if conflicting:
        raise DeviceNotFoundError(
            f"No glove link matched the requested {config.side} hand. "
            "Firmware side mismatch: " + "; ".join(conflicting) + ".")

    raise DeviceNotFoundError(
        f"No glove link is bound to the {config.side} hand yet. "
        f"Detected: {describe_ports(ports)}. "
        "Run `python main.py devices` to bind it by serial number.")


def _bound_serials(side: str, registry: Path) -> set[str]:
    """Return the serials the registry claims for one hand (never raises)."""

    try:
        data = load_device_registry(registry)
    except (GloveDeviceError, OSError, ValueError):
        return set()
    return {str(serial).strip().upper() for serial in data[side]["usb_serials"]}


def _resolve_explicit_port(config: _RawLinkConfig, port: str) -> LinkSpec:
    ports = list_matching_ports()
    selected = next((item for item in ports
                     if path.normcase(str(item.device)) == path.normcase(port)), None)
    wanted = str(config.serial_number or "").strip().upper()
    if selected is not None:
        kind = link_kind_of(selected)
        assert kind is not None  # list_matching_ports only returns glove links
        if config.link not in (LINK_AUTO, kind):
            raise DeviceNotFoundError(f"{port} is a {kind} link, expected {config.link}")
        serial = str(selected.serial_number or "").strip().upper()
        if wanted and wanted != serial:
            raise DeviceNotFoundError(f"{port} does not match glove serial {wanted}")
        vid, pid = _LINK_IDS[kind]
        return LinkSpec(str(selected.device), serial, vid, pid, kind, BOUND_BY_PORT,
                        _usb_priority_note(kind, ports), _port_identity(selected))
    raise DeviceNotFoundError(f"No supported glove found on {port}")


def _port_identity(port) -> tuple[int, int, str, str | None]:
    return (port.vid, port.pid, str(port.serial_number or "").strip().upper(),
            getattr(port, "location", None))


def _check_explicit_port(link: LinkSpec, generation: int) -> str:
    # A serial-less replacement on the same USB socket may have identical
    # descriptors. Require the caller to start a new session after a drop.
    if generation and not link.serial_number:
        raise StreamClosedError(f"Reopen the raw glove to reconnect serial-less port {link.port}")
    selected = next((item for item in list_matching_ports()
                     if str(item.device) == link.port), None)
    identity = _port_identity(selected) if selected is not None else None
    if identity != link.port_identity:
        raise OSError(f"Selected glove port {link.port} changed identity")
    return link.port


def make_raw_stream(
    link: LinkSpec, *, resolve_port: Callable[[int], str] | None = None,
) -> RawImuStream:
    """Build one transport for the resolved IMU and optional tactile consumer."""
    return RawImuStream(
        serial_port=link.port,
        usb_vid=link.vid,
        usb_pid=link.pid,
        _expected_serial=link.serial_number,
        _resolve_port=resolve_port,
    )


def check_frame_side(side: str, link: LinkSpec, firmware: str | None) -> None:
    reported_side = firmware_hand_side(firmware) if firmware is not None else None
    if reported_side is not None and reported_side != side:
        serial = link.serial_number or "no-serial"
        raise DeviceBindingError(
            f"Firmware side mismatch on {link.kind} ({serial}): "
            f"requested the {side} hand, firmware {firmware} reports the {reported_side} hand.")


class _RawGlove:
    """An owned IMU acquisition session for one hand."""

    def __init__(self, config: _RawLinkConfig, port: str | None):
        self._config = config
        self._requested_port = port
        self._link: LinkSpec | None = None
        self._stream: RawImuStream | None = None
        self._lock = RLock()
        self._closed = False

    @property
    def side(self) -> str:
        return self._config.side

    @property
    def port(self) -> str | None:
        with self._lock:
            return (self._stream.port if self._stream is not None else None) or (
                self._link.port if self._link is not None else self._requested_port)

    @property
    def firmware_version(self) -> str | None:
        """Latest firmware version from the active transport."""
        with self._lock:
            if self._stream is None or self._closed:
                return None
            transport = self._stream._transport
            with transport._state_lock:
                return transport.firmware_version if transport.connected else None

    @property
    def connection_generation(self) -> int:
        """Current transport generation, or zero while disconnected."""
        with self._lock:
            if self._stream is None or self._closed:
                return 0
            transport = self._stream._transport
            with transport._state_lock:
                return transport.connection_generation if transport.connected else 0

    def info(self) -> GloveInfo:
        """Snapshot the raw session's port, USB serial, link, and firmware."""
        with self._lock:
            link = self._link
            firmware = self.firmware_version
            return GloveInfo(
                side=self._config.side,
                sdk_version=get_version(),
                firmware_version=firmware,
                link_kind=link.kind if link is not None else self._config.link,
                port=self.port or "",
                serial_number=link.serial_number if link is not None else "",
                mag_available=mag_available(firmware),
            )

    def start(self) -> _RawGlove:
        with self._lock:
            if self._closed:
                raise StreamClosedError("raw glove is closed")
            if self._stream is None:
                if self._requested_port is None:
                    link = resolve_link(self._config)
                else:
                    link = _resolve_explicit_port(self._config, self._requested_port)
                if self._requested_port is not None or not link.serial_number:
                    self._stream = make_raw_stream(
                        link, resolve_port=lambda generation: _check_explicit_port(link, generation))
                else:
                    self._stream = make_raw_stream(link)
                self._link = link
                try:
                    self._stream.start()
                except BaseException:
                    self.close()
                    raise
        return self

    def read(self, timeout: float | None = None) -> RawImuFrame:
        """Return the next raw packet, or raise StreamTimeoutError on expiry."""
        with self._lock:
            self.start()
            stream, link = self._stream, self._link
        assert stream is not None and link is not None
        frame = stream.read(timeout=timeout)
        with self._lock:
            if self._closed:
                raise StreamClosedError("raw glove is closed")
            try:
                check_frame_side(self.side, link, frame.firmware_version)
            except DeviceBindingError as error:
                try:
                    self.close()
                except StreamClosedError as close_error:
                    raise error from close_error
                raise
        return frame

    def close(self) -> None:
        """Stop the reader and release its serial handle."""
        with self._lock:
            self._closed = True
            if self._stream is not None:
                self._stream.close()
                self._stream = None

    def __enter__(self) -> _RawGlove:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()


def open_raw_glove(
    side: str = "right",
    *,
    port: str | None = None,
    link: str = LINK_AUTO,
    serial_number: str | None = None,
    registry: Path | str | None = None,
    sample_rate_hz: int = 80,
) -> _RawGlove:
    """Acquire physical-order IMU data. First read or context entry starts I/O.

    An explicit port skips hand discovery. With no port, use the same link
    resolution as the live glove. An explicit port stays pinned to its original
    USB identity. A serial-less port requires a new session after a disconnect.
    The device controls the wire cadence;
    sample_rate_hz is the same nominal setting accepted by open_glove.
    """
    if int(sample_rate_hz) <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if port is not None:
        port = str(port).strip()
        if not port:
            raise ValueError("port must not be empty")
    config = _RawLinkConfig(_side(side), _normalise_kind(link), serial_number, registry)
    return _RawGlove(config, port)
