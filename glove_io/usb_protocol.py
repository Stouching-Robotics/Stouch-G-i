"""STM32 glove USB CDC framing and IMU payload decoding.

The firmware multiplexes status, IMU and pressure-matrix messages on one CDC
byte stream.  This module deliberately owns only the wire protocol; serial-port
opening and MANO processing remain in :mod:`modules.hand.hand`.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np


STM32_USB_VID = 0x0483
STM32_USB_PID = 0x5740

USB_MAGIC = b"\xA5\x5A"
# Wire format per ``流程图与数据帧格式_v1.2.2.pdf`` (section 2.1): an 18-byte
# header carrying the complete ``major.minor.patch`` version, then the payload
# and a CRC16-CCITT-FALSE trailer:
#   magic(2) major(1) minor(1) patch(1) type(1)
#   sequence(4) timestamp_us(4) payload_len(2) flags(2)
# ``sequence`` is the full 32-bit ``g_sequence`` counter (little-endian, from
# 0); the v1.2.2 format has no separate magnetometer-ready bit.  The version
# comparison is deliberately semantic rather than a hard-coded value.
USB_NEW_FORMAT_FROM = (1, 2, 2)
USB_HEADER_SIZE = 18
USB_CRC_SIZE = 2
USB_MAX_PAYLOAD = 1100

# Firmware v1.2.10 slims the outer frame: the header drops the 2-byte flags
# word and the sequence shrinks 32 -> 16 bits (its top bit becomes the
# magnetometer-ready flag), so the header is 14 bytes.  The IMU valid mask moves
# into the payload (128 -> 130 B) and the pressure matrix is packed to 12 bits
# per sample (524 -> 384 B).  Detection is version-based.
USB_SLIM_FORMAT_FROM = (1, 2, 10)
USB_SLIM_HEADER_SIZE = 14
IMU_VALID_MASK_SIZE = 2
IMU_SLIM_PAYLOAD_SIZE = 130
ADC_MATRIX_SLIM_PAYLOAD_SIZE = 384

# Interface spec §4.6: the slim header's 16-bit sequence word is a counter
# whose bit 15 the firmware forces to 1 while the magnetometer is not yet
# ready.  That flag is what drives the "move the hand in circles" prompt, so
# the parser must keep it out of the counter it publishes as ``sequence``.
SEQUENCE_COUNTER_MASK = 0x7FFF
MAGNETOMETER_NOT_READY_BIT = 0x8000

USB_TYPE_STATUS = 0x01
USB_TYPE_IMU_Q14 = 0x02
USB_TYPE_ADC_MATRIX = 0x03
USB_TYPE_DELAY_RESPONSE = 0x04
USB_TYPE_HOST_PROBE = 0x50

# v1.2.11 host→device ping probe: ``A5 5A 50 seq(2B) t1(8B)`` (no version, no
# CRC).  The device replies with a normal type 0x04 frame whose 18-byte payload
# is ``seq(2B) t1(8B) t2(4B) t3(4B)``.
USB_HOST_PROBE_SIZE = 13
USB_DELAY_RESPONSE_PAYLOAD_SIZE = 18

# First firmware that understands the probe frame; see
# :func:`supports_latency_probe`.
USB_LATENCY_PROBE_FROM = (1, 2, 11)

TOTAL_IMU_COUNT = 16
IMU_COMPONENT_COUNT = 4
IMU_Q14_PAYLOAD_SIZE = TOTAL_IMU_COUNT * IMU_COMPONENT_COUNT * 2
Q14_SCALE = 1.0 / 16384.0
ADC_MATRIX_ROWS = 16
ADC_MATRIX_COLS = 16
ADC_MATRIX_META_SIZE = 12
ADC_MATRIX_PAYLOAD_SIZE = (
    ADC_MATRIX_META_SIZE + ADC_MATRIX_ROWS * ADC_MATRIX_COLS * 2)

def parse_firmware_version(version: str | tuple[int, ...] | list[int]) -> tuple[int, ...] | None:
    """Return numeric firmware components, or ``None`` for malformed input."""

    try:
        if isinstance(version, str):
            components = tuple(int(part.strip()) for part in version.split("."))
        else:
            components = tuple(int(part) for part in version)
    except (TypeError, ValueError):
        return None
    return components if len(components) == 3 and all(value >= 0 for value in components) else None


def is_supported_version(version: str | tuple[int, ...] | list[int]) -> bool:
    """Return whether ``version`` uses the current 18-byte-header format.

    The PDF documents v1.2.2 itself as the current format.  Accepting it as
    well as all newer versions is important when a board is flashed with the
    documented release; future patch/minor releases remain compatible.
    """

    components = parse_firmware_version(version)
    return components is not None and components >= USB_NEW_FORMAT_FROM


def firmware_real_version(
        version: str | tuple[int, ...] | list[int],
) -> tuple[int, int, int] | None:
    """Return ``(major, minor, patch)`` with the hand-side digit removed.

    v1.2.6+ encodes the hand side in the ones digit of the patch
    (``1.2.61`` -> real ``1.2.6``, ``1.2.101`` -> real ``1.2.10``).  Only a
    valid side digit (1 = left, 2 = right) is stripped: a patch ending in 0 is
    a plain patch, so a firmware reporting the documented ``1.2.10`` keeps its
    real version instead of collapsing to ``1.2.1``.
    """

    components = parse_firmware_version(version)
    if components is None:
        return None
    major, minor, patch = components
    if patch >= 11 and patch % 10 in (1, 2):
        patch //= 10
    return (major, minor, patch)


def is_slim_format(version: str | tuple[int, ...] | list[int]) -> bool:
    """Return whether ``version`` uses the 14-byte slim header (v1.2.10+)."""

    real = firmware_real_version(version)
    return real is not None and real >= USB_SLIM_FORMAT_FROM


def supports_latency_probe(version: str | tuple[int, ...] | list[int]) -> bool:
    """Return whether ``version`` answers the v1.2.11 latency probe.

    v1.2.11 is the first firmware that knows the type 0x50 probe frame, so the
    probe is only ever sent to a device that has announced it or newer; older
    firmware would just see 13 unrecognised bytes injected into its stream.
    """

    real = firmware_real_version(version)
    return real is not None and real >= USB_LATENCY_PROBE_FROM


def firmware_hand_side(version: str) -> str | None:
    """Return the hand side encoded in a firmware version, or ``None``.

    The patch's ones digit carries the side (1 = left, 2 = right), e.g.
    ``1.2.91`` -> "left", ``1.2.62`` -> "right", ``1.2.101`` -> "left",
    ``1.2.102`` -> "right".  A patch ending in 0 is a plain patch and a
    single-digit patch (``1.2.3``) carries no side, so both return ``None``.
    """

    components = parse_firmware_version(version)
    if components is None:
        return None
    patch = components[2]
    if patch < 11:
        return None
    digit = patch % 10
    if digit == 1:
        return "left"
    if digit == 2:
        return "right"
    return None


def decode_sequence_word(raw_sequence: int) -> tuple[int, bool]:
    """Split a slim-header sequence word into (counter, magnetometer_ready).

    ``sequence`` carries only the 15-bit counter; the flag is reported
    separately so consumers counting frames never see the forced bit.  See
    interface spec §4.6.
    """

    word = int(raw_sequence)
    return (word & SEQUENCE_COUNTER_MASK,
            (word & MAGNETOMETER_NOT_READY_BIT) == 0)


@dataclass(frozen=True)
class UsbCdcFrame:
    """One CRC-validated frame extracted from the CDC byte stream."""

    version: str
    message_type: int
    sequence: int
    timestamp_us: int
    flags: int
    payload: bytes
    # Which layout the parser actually matched.  The mask location follows the
    # layout, not the version string: a firmware reporting ``1.2.10`` is not
    # distinguishable from a side-encoded ``1.2.1x`` by version alone, so
    # deriving this from the version would read the mask from the wrong place.
    header_size: int = USB_HEADER_SIZE
    # False while the firmware reports the magnetometer as not yet ready.  The
    # flag only exists in the slim layout, where it rides in bit 15 of the
    # sequence word (see :func:`decode_sequence_word`); the legacy 32-bit
    # sequence has no such bit and is always reported ready, so the prompt
    # never stalls on older firmware.
    mag_ready: bool = True

    @property
    def firmware_version(self) -> str:
        """Compatibility alias used by older SDK callers."""

        return self.version

    @property
    def valid_mask(self) -> int:
        """16-bit IMU valid mask, unified across wire formats.

        v1.2.10+ carries the mask as the first two payload bytes of an IMU
        frame; earlier firmware puts it in the header ``flags`` word.
        """

        if (self.message_type == USB_TYPE_IMU_Q14
                and self.header_size == USB_SLIM_HEADER_SIZE):
            return int.from_bytes(self.payload[:IMU_VALID_MASK_SIZE], "little")
        return self.flags


@dataclass(frozen=True)
class UsbAdcMatrixFrame:
    """Decoded STM32 16x16 tactile matrix payload."""

    sequence: int
    scan_time_us: int
    samples: np.ndarray


@dataclass(frozen=True)
class UsbDelayResponse:
    """Decoded v1.2.11 type 0x04 delay-measurement response payload."""

    sequence: int
    t1_us: int
    t2_us: int
    t3_us: int


@dataclass(frozen=True)
class UsbLinkLatency:
    """One measured host<->device round trip (v1.2.11 ping)."""

    sequence: int
    rtt_us: int
    device_turnaround_us: int
    t1_us: int
    t2_us: int
    t3_us: int
    host_receive_us: int

    @property
    def rtt_ms(self) -> float:
        return self.rtt_us / 1000.0


# CRC16-CCITT-FALSE (poly 0x1021, init 0xFFFF) byte-at-a-time lookup table,
# precomputed once so the per-frame hot path avoids an 8-iteration bit loop.
_CRC16_TABLE = []
for _crc_i in range(256):
    _crc_c = _crc_i << 8
    for _ in range(8):
        _crc_c = (((_crc_c << 1) ^ 0x1021) & 0xFFFF
                  if _crc_c & 0x8000 else (_crc_c << 1) & 0xFFFF)
    _CRC16_TABLE.append(_crc_c)


def crc16_ccitt_false(data: bytes | bytearray | memoryview) -> int:
    """Return CRC16-CCITT-FALSE (poly 0x1021, init 0xFFFF)."""

    crc = 0xFFFF
    for value in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16_TABLE[
            ((crc >> 8) ^ int(value)) & 0xFF]
    return crc


class UsbCdcFrameParser:
    """Incrementally recover framed messages from arbitrary serial chunks."""

    def __init__(self):
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.crc_errors = 0
        self.length_errors = 0

    def reset(self) -> None:
        self.buffer.clear()

    def feed(self, data: bytes | bytearray | memoryview) -> list[UsbCdcFrame]:
        if data:
            self.buffer.extend(data)

        frames: list[UsbCdcFrame] = []
        while True:
            magic_index = self.buffer.find(USB_MAGIC)
            if magic_index < 0:
                # Keep a trailing 0xA5 because it may be the first byte of a
                # magic sequence split across two serial reads.
                keep = 1 if self.buffer.endswith(USB_MAGIC[:1]) else 0
                discard = len(self.buffer) - keep
                if discard > 0:
                    del self.buffer[:discard]
                    self.discarded_bytes += discard
                break

            if magic_index > 0:
                del self.buffer[:magic_index]
                self.discarded_bytes += magic_index

            # Magic(2) + version(3) are enough to read the version, but the
            # version string alone cannot separate the layouts: a firmware
            # reporting patch 10 (``1.2.10``) is indistinguishable from a
            # side-encoded ``1.2.1x``, and choosing wrong silently drops every
            # pressure frame.  The layout is therefore confirmed by CRC, trying
            # the version-implied order first.  The payload length must also be
            # one this layout can produce, which rejects the wrong layout
            # immediately in almost every case.
            # Magic(2) + version(3) + type(1): reading the type at index 5
            # needs six bytes, not five, or a chunk boundary landing exactly
            # there raises IndexError inside the reader thread.
            if len(self.buffer) < 6:
                break
            version = ".".join(str(value) for value in self.buffer[2:5])
            message_type = int(self.buffer[5])
            if is_slim_format(version):
                layouts = (USB_SLIM_HEADER_SIZE, USB_HEADER_SIZE)
            else:
                layouts = (USB_HEADER_SIZE, USB_SLIM_HEADER_SIZE)

            matched = None
            incomplete = False
            length_rejected = 0
            for header_size in layouts:
                if len(self.buffer) < header_size:
                    incomplete = True
                    continue
                if header_size == USB_SLIM_HEADER_SIZE:
                    sequence, mag_ready = decode_sequence_word(
                        int.from_bytes(self.buffer[6:8], "little"))
                    timestamp_us = int.from_bytes(self.buffer[8:12], "little")
                    payload_length = int.from_bytes(self.buffer[12:14], "little")
                    flags = 0
                    expected_lengths = (
                        IMU_SLIM_PAYLOAD_SIZE, ADC_MATRIX_SLIM_PAYLOAD_SIZE)
                else:
                    sequence = int.from_bytes(self.buffer[6:10], "little")
                    timestamp_us = int.from_bytes(self.buffer[10:14], "little")
                    payload_length = int.from_bytes(self.buffer[14:16], "little")
                    flags = int.from_bytes(self.buffer[16:18], "little")
                    mag_ready = True
                    expected_lengths = (
                        IMU_Q14_PAYLOAD_SIZE, ADC_MATRIX_PAYLOAD_SIZE)

                if message_type in (USB_TYPE_IMU_Q14, USB_TYPE_ADC_MATRIX):
                    # Fixed-size payloads: anything else means this is the
                    # wrong layout, or the buffer is not at a frame boundary.
                    if payload_length not in expected_lengths:
                        length_rejected += 1
                        continue
                elif payload_length > USB_MAX_PAYLOAD:
                    # Status reports and the delay response carry variable- or
                    # fixed-length data that never exceeds this bound.
                    length_rejected += 1
                    continue

                total_length = header_size + payload_length + USB_CRC_SIZE
                if len(self.buffer) < total_length:
                    incomplete = True
                    continue

                crc_offset = header_size + payload_length
                expected_crc = int.from_bytes(
                    self.buffer[crc_offset:crc_offset + USB_CRC_SIZE], "little")
                if crc16_ccitt_false(self.buffer[:crc_offset]) != expected_crc:
                    continue

                matched = (header_size, sequence, timestamp_us, flags,
                           total_length, crc_offset, mag_ready)
                break

            if matched is None:
                if incomplete:
                    break
                del self.buffer[0]
                self.discarded_bytes += 1
                if length_rejected == len(layouts):
                    self.length_errors += 1
                else:
                    self.crc_errors += 1
                continue

            (header_size, sequence, timestamp_us, flags,
             total_length, crc_offset, mag_ready) = matched
            frames.append(UsbCdcFrame(
                version=version,
                message_type=message_type,
                sequence=sequence,
                timestamp_us=timestamp_us,
                flags=flags,
                payload=bytes(self.buffer[header_size:crc_offset]),
                header_size=header_size,
                mag_ready=mag_ready,
            ))
            del self.buffer[:total_length]

        return frames


def decode_imu_q14_payload(payload: bytes) -> np.ndarray:
    """Decode an IMU payload into SciPy-order XYZW quaternions.

    v1.2.10+ prepends a 2-byte valid mask (130 B total); the quaternion body is
    always the trailing 128 bytes, so the mask is stripped here.  Use
    :attr:`UsbCdcFrame.valid_mask` for the mask itself.
    """

    if len(payload) == IMU_SLIM_PAYLOAD_SIZE:
        payload = payload[IMU_VALID_MASK_SIZE:]
    elif len(payload) != IMU_Q14_PAYLOAD_SIZE:
        raise ValueError(
            f"STM32 IMU payload must be {IMU_Q14_PAYLOAD_SIZE} or "
            f"{IMU_SLIM_PAYLOAD_SIZE} bytes, got {len(payload)}")

    raw = np.asarray(
        struct.unpack("<64h", payload), dtype=np.float64
    ).reshape(TOTAL_IMU_COUNT, IMU_COMPONENT_COUNT)
    wxyz = raw * Q14_SCALE
    return wxyz[:, [1, 2, 3, 0]]


def _unpack_12bit_matrix(payload: bytes) -> np.ndarray:
    """Unpack a 384-byte 12-bit-packed block into a 16x16 float array.

    Two consecutive 12-bit samples share three bytes, little-endian bit order::

        byte0 = sample0[ 7: 0]
        byte1 = sample0[11: 8] | (sample1[ 3: 0] << 4)
        byte2 = sample1[11: 4]
    """

    data = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    sample0 = data[:, 0] | ((data[:, 1] & 0x0F) << 8)
    sample1 = (data[:, 1] >> 4) | (data[:, 2] << 4)
    samples = np.empty(ADC_MATRIX_ROWS * ADC_MATRIX_COLS, dtype=np.uint32)
    samples[0::2] = sample0
    samples[1::2] = sample1
    return samples.reshape(ADC_MATRIX_ROWS, ADC_MATRIX_COLS).astype(np.float32)


def decode_adc_matrix_payload(
        payload: bytes,
        *,
        sequence: int = 0,
        scan_time_us: int = 0,
) -> UsbAdcMatrixFrame:
    """Decode a pressure-matrix payload into a 16x16 sample array.

    v1.2.10+ sends a bare 384-byte 12-bit-packed block with no metadata, so the
    frame header's sequence/timestamp are passed in by the caller.  Earlier
    firmware sends a 524-byte payload with a 12-byte metadata prefix.
    """

    if len(payload) == ADC_MATRIX_SLIM_PAYLOAD_SIZE:
        samples = _unpack_12bit_matrix(payload)
        return UsbAdcMatrixFrame(
            sequence=sequence,
            scan_time_us=scan_time_us,
            samples=samples,
        )

    if len(payload) != ADC_MATRIX_PAYLOAD_SIZE:
        raise ValueError(
            f"STM32 ADC matrix payload must be {ADC_MATRIX_SLIM_PAYLOAD_SIZE} "
            f"or {ADC_MATRIX_PAYLOAD_SIZE} bytes, got {len(payload)}")
    rows, cols, adc_bits, order = payload[:4]
    if (rows, cols, adc_bits, order) != (
            ADC_MATRIX_ROWS, ADC_MATRIX_COLS, 12, 0):
        raise ValueError(
            "unsupported STM32 ADC matrix metadata: "
            f"rows={rows}, cols={cols}, bits={adc_bits}, order={order}")
    sequence = int.from_bytes(payload[4:8], "little")
    scan_time_us = int.from_bytes(payload[8:12], "little")
    samples = np.frombuffer(
        payload, dtype="<u2", offset=ADC_MATRIX_META_SIZE,
        count=ADC_MATRIX_ROWS * ADC_MATRIX_COLS,
    ).reshape(ADC_MATRIX_ROWS, ADC_MATRIX_COLS).astype(np.float32)
    return UsbAdcMatrixFrame(
        sequence=sequence,
        scan_time_us=scan_time_us,
        samples=samples,
    )


def encode_host_probe(sequence: int, t1_us: int) -> bytes:
    """Encode the v1.2.11 host→device ping probe (type 0x50).

    The 13-byte frame is ``A5 5A 50 seq(2B) t1(8B)`` with no version and no
    CRC, matching the v1.2.11 interface spec §4.5.
    """

    return (
        USB_MAGIC
        + bytes([USB_TYPE_HOST_PROBE])
        + struct.pack("<H", int(sequence) & 0xFFFF)
        + struct.pack("<Q", int(t1_us) & 0xFFFFFFFFFFFFFFFF)
    )


def decode_delay_response(payload: bytes) -> UsbDelayResponse:
    """Decode the 18-byte v1.2.11 type 0x04 delay-measurement response."""

    if len(payload) != USB_DELAY_RESPONSE_PAYLOAD_SIZE:
        raise ValueError(
            f"STM32 delay response payload must be "
            f"{USB_DELAY_RESPONSE_PAYLOAD_SIZE} bytes, got {len(payload)}")
    return UsbDelayResponse(
        sequence=int.from_bytes(payload[0:2], "little"),
        t1_us=int.from_bytes(payload[2:10], "little"),
        t2_us=int.from_bytes(payload[10:14], "little"),
        t3_us=int.from_bytes(payload[14:18], "little"),
    )


def link_latency_from_response(
        response: UsbDelayResponse,
        *,
        host_receive_us: int,
        host_send_us: int | None = None,
) -> UsbLinkLatency:
    """Combine a decoded 0x04 response with the host send/receive times.

    The round trip is measured entirely inside the host clock and needs no
    clock synchronisation with the device.  ``host_send_us`` (the timestamp the
    host actually recorded when it wrote the probe) is preferred over the
    device's echoed ``t1`` so a corrupted echo cannot produce a bogus latency;
    the probe was already matched by sequence number.  ``t2``/``t3`` are device
    timestamps on a 32-bit microsecond counter that wraps after ~71.6 minutes,
    so the turnaround is computed modulo 2^32.
    """

    t1_us = int(response.t1_us)
    sent_us = t1_us if host_send_us is None else int(host_send_us)
    return UsbLinkLatency(
        sequence=int(response.sequence),
        rtt_us=int(host_receive_us) - sent_us,
        device_turnaround_us=(int(response.t3_us) - int(response.t2_us)) & 0xFFFFFFFF,
        t1_us=t1_us,
        t2_us=int(response.t2_us),
        t3_us=int(response.t3_us),
        host_receive_us=int(host_receive_us),
    )


def plausible_quaternion_mask(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Apply the same broad norm gate used by the ESP runtime."""

    quaternions = np.asarray(quaternions_xyzw, dtype=float)
    if quaternions.shape != (TOTAL_IMU_COUNT, IMU_COMPONENT_COUNT):
        raise ValueError("quaternions must have shape (16, 4)")
    norm_squared = np.sum(quaternions * quaternions, axis=1)
    return (
        np.all(np.isfinite(quaternions), axis=1)
        & (norm_squared > 0.25)
        & (norm_squared < 2.25)
    )


def find_stm32_cdc_port(vid: int = STM32_USB_VID, pid: int = STM32_USB_PID) -> str | None:
    """Return the first matching STM32 CDC port, or ``None`` when absent."""

    import serial.tools.list_ports

    for port in serial.tools.list_ports.comports():
        if port.vid == int(vid) and port.pid == int(pid):
            return port.device
    return None


def list_serial_ports() -> list[tuple[str, str, int | None, int | None]]:
    """Return serial-port details for CLI diagnostics."""

    import serial.tools.list_ports

    return [
        (port.device, port.description, port.vid, port.pid)
        for port in serial.tools.list_ports.comports()
    ]
