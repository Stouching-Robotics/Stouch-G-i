"""STM32 glove USB CDC framing and IMU payload decoding.

The firmware multiplexes status, IMU and pressure-matrix messages on one CDC
byte stream.  This module deliberately owns only the wire protocol; serial-port
opening and MANO processing remain in :mod:`modules.hand.hand`.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
import threading
from typing import Iterable

import numpy as np


# Explicit export list: ``glove_io.usb_protocol`` is a thin re-export shim over
# this module, so this is the whole protocol surface both layers see.  Keeping
# it explicit keeps ``from ... import *`` from leaking ``np``/``struct`` into
# callers and makes an accidental rename a loud failure instead of a silent one.
__all__ = [
    # Constants and frame layout
    "STM32_USB_VID", "STM32_USB_PID",
    "USB_MAGIC", "USB_HEADER_SIZE", "USB_CRC_SIZE", "USB_MAX_PAYLOAD",
    "USB_NEW_FORMAT_FROM", "USB_SLIM_FORMAT_FROM", "USB_SLIM_HEADER_SIZE",
    "USB_FULL_SEQUENCE_FROM", "SEQUENCE_COUNTER_MASK",
    "MAGNETOMETER_NOT_READY_BIT",
    "USB_TYPE_STATUS", "USB_TYPE_IMU_Q14", "USB_TYPE_ADC_MATRIX",
    "USB_TYPE_DELAY_RESPONSE", "USB_TYPE_MAG_TELEMETRY", "USB_TYPE_HOST_PROBE",
    "USB_HOST_PROBE_SIZE", "USB_DELAY_RESPONSE_PAYLOAD_SIZE",
    "USB_LATENCY_PROBE_FROM",
    "TOTAL_IMU_COUNT", "IMU_COMPONENT_COUNT", "IMU_Q14_PAYLOAD_SIZE",
    "IMU_VALID_MASK_SIZE", "IMU_SLIM_PAYLOAD_SIZE", "Q14_SCALE",
    "ADC_MATRIX_ROWS", "ADC_MATRIX_COLS", "ADC_MATRIX_META_SIZE",
    "ADC_MATRIX_PAYLOAD_SIZE", "ADC_MATRIX_SLIM_PAYLOAD_SIZE",
    "MAG_TELEMETRY_PAYLOAD_SIZE", "MAG_TELEMETRY_PAYLOAD_SIZES",
    "MAG_TELEMETRY_IMU_COUNT", "MAG_TELEMETRY_VALID_MASK_SIZE",
    "MAG_READY_MIN_NUMERATOR", "MAG_READY_MIN_DENOMINATOR", "mag_gate_met",
    "MAG_LEVEL_SHIFT", "MAG_LEVEL_MASK",
    # Decoded frames
    "UsbCdcFrame", "UsbAdcMatrixFrame", "UsbDelayResponse", "UsbMagTelemetry",
    "UsbLinkLatency", "MagTelemetrySnapshot", "MagTelemetryCache",
    # Parser and codecs
    "UsbCdcFrameParser", "crc16_ccitt_false", "encode_host_probe",
    "decode_imu_q14_payload", "decode_adc_matrix_payload",
    "decode_delay_response", "decode_mag_telemetry_payload",
    "decode_sequence_word", "plausible_quaternion_mask",
    "link_latency_from_response",
    # Version helpers
    "parse_firmware_version", "firmware_real_version", "is_supported_version",
    "is_slim_format", "supports_latency_probe", "firmware_hand_side",
    "display_firmware_version",
    # Channel/hand mapping
    "BLUETOOTH_DONGLE_VID", "BLUETOOTH_DONGLE_PID", "glove_link_kind",
    "STM32_LINK_IDS", "LEGACY_CHANNEL_TO_HAND_BY_SIDE",
    "NEW_FIRMWARE_CHANNEL_TO_HAND_BY_SIDE", "FIRMWARE_OUTPUT_ORDER_BY_SIDE",
    "DEFAULT_CHANNEL_TO_HAND", "validate_channel_to_hand",
    "channel_to_hand_for_firmware", "remap_physical_to_hand",
    "readable_imu_mask",
    # Port discovery
    "find_stm32_cdc_port", "list_serial_ports",
]


STM32_USB_VID = 0x0483
STM32_USB_PID = 0x5740

# The Bluetooth build of the glove firmware sends the same frame stream over
# SPP.  On the host side a BP101Y dongle enumerates as its own USB CDC device
# and relays that stream byte for byte, so a dongle port carries a complete,
# protocol-identical glove link.  Only two things differ from the wired case:
# the USB PID (0x2013 instead of 0x5740) and the serial number, which belongs
# to the dongle rather than to the glove behind it.
BLUETOOTH_DONGLE_VID = 0x0483
BLUETOOTH_DONGLE_PID = 0x2013

# Every (vid, pid, link kind) that can carry a glove stream.  ``glove_link_kind``
# is the single place where the wired and Bluetooth transports are told apart.
STM32_LINK_IDS: tuple[tuple[int, int, str], ...] = (
    (STM32_USB_VID, STM32_USB_PID, "usb"),
    (BLUETOOTH_DONGLE_VID, BLUETOOTH_DONGLE_PID, "bluetooth"),
)


def glove_link_kind(vid: int | None, pid: int | None) -> str | None:
    """Return ``"usb"``/``"bluetooth"`` for a glove link, else ``None``.

    A ``None`` result means the device is not a glove transport at all, so it
    must never be opened as one.
    """

    for link_vid, link_pid, kind in STM32_LINK_IDS:
        if vid == link_vid and pid == link_pid:
            return kind
    return None

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

# Firmware v1.2.10 (``数据接口说明_v1.2.10``) slims the outer frame: the header
# drops the 2-byte flags word and the sequence shrinks 32 -> 16 bits (its top
# bit becomes the magnetometer-ready flag), so the header is 14 bytes.  The IMU
# valid mask moves into the payload (128 -> 130 B) and the pressure matrix is
# packed to 12 bits per sample (524 -> 384 B).  Detection is version-based, so
# both wire formats stay supported in one parser.
USB_SLIM_FORMAT_FROM = (1, 2, 10)
USB_SLIM_HEADER_SIZE = 14
IMU_VALID_MASK_SIZE = 2
IMU_SLIM_PAYLOAD_SIZE = 130
ADC_MATRIX_SLIM_PAYLOAD_SIZE = 384

# Through v1.2.12, bit 15 of the slim sequence word is a magnetometer-not-ready
# flag and the remaining 15 bits are the counter.  Starting with v1.2.13 the
# word is an ordinary 16-bit counter; readiness is reported by type 0x05.
SEQUENCE_COUNTER_MASK = 0x7FFF
MAGNETOMETER_NOT_READY_BIT = 0x8000
USB_FULL_SEQUENCE_FROM = (1, 2, 13)

USB_TYPE_STATUS = 0x01
USB_TYPE_IMU_Q14 = 0x02
USB_TYPE_ADC_MATRIX = 0x03
USB_TYPE_DELAY_RESPONSE = 0x04
USB_TYPE_MAG_TELEMETRY = 0x05
USB_TYPE_HOST_PROBE = 0x50

# v1.2.13: the payload carries one calibration byte per physical IMU (16 of them)
# and the magnetometer vectors are gone.
#
# 18 is the *only* length accepted.  A 23-byte variant used to be tolerated as
# "the old struct with its reserves kept"; it had no source -- the vendor's own
# reference decoder rejects anything but 18 (MAG_TELEMETRY(1).md 6), no firmware
# has ever been seen to send it, and nothing said where those five bytes went.
# Tolerating an unverifiable layout is only safe by luck, so a frame of any other
# length is dropped and the parser resyncs on the next magic.
MAG_TELEMETRY_VALID_MASK_SIZE = 2
MAG_TELEMETRY_IMU_COUNT = 16
MAG_TELEMETRY_PAYLOAD_SIZE = MAG_TELEMETRY_VALID_MASK_SIZE + MAG_TELEMETRY_IMU_COUNT
MAG_TELEMETRY_PAYLOAD_SIZES = (MAG_TELEMETRY_PAYLOAD_SIZE,)
# The readiness gate: three quarters of the channels a 0x05 frame carried must
# read 3.
#
# The denominator is that frame's own valid mask -- the IMUs it read this round
# -- not the 16 physical positions.  There is no usable "is this IMU working"
# signal to divide by instead: the only thing the telemetry licenses is "of the
# channels read this round, this fraction are at 3".  A frame carries about
# twelve of the sixteen, so dividing by 16 would charge the glove for channels
# nobody read, and make the bar unreachable on a hand with any sensor that is
# simply quiet.
#
# Not all of them, because a glove held still never separates hard iron from the
# earth's field on the fingers it does not move: the last few channels stay low
# however long the user circles (MAG_TELEMETRY.md 2.2).  The bar sits at three
# quarters rather than a half so a half-calibrated glove is not called ready; on
# the ~12 channels a frame carries that is 9 of 12.
MAG_READY_MIN_NUMERATOR = 3
MAG_READY_MIN_DENOMINATOR = 4


def mag_gate_met(fresh_ready: int, fresh_count: int) -> bool:
    """Is ``fresh_ready`` of this frame's ``fresh_count`` channels enough?

    A frame that carried nothing cannot pass: "0 of 0" is not a calibrated
    magnetometer, it is an absent one.
    """

    return (fresh_count > 0
            and fresh_ready * MAG_READY_MIN_DENOMINATOR
            >= fresh_count * MAG_READY_MIN_NUMERATOR)

# Where the MAG level sits in each calibration byte.
#
# The byte is the BNO055's CALIB_STAT register (0x35) verbatim, so it is the
# chip's own order -- SYS in the high pair down to MAG in the low pair:
#
#     bit:      7 6     5 4     3 2     1 0
#     content:  SYS     GYR     ACC     MAG
#
# BNO055 datasheet 4.3.54.  ``algorithm/imu_calibrate_cli``'s per-IMU gate in
# ``common/datamodels.py`` has always read it this way ([5:4]>=2 is the gyro,
# [1:0]>=1 the magnetometer).
#
# This was briefly believed to be reversed (MAG in [7:6]), on the strength of
# the rule "SYS = min of the other three" applied to one early capture.  That
# rule does not hold on this firmware -- feed the newer captures through it and
# *every* placement of SYS contradicts itself -- so it was never evidence.  The
# capture that settled it is ``mag_capture/``'s 0x05 dump:
#
#   * [1:0] is 0 for every fresh channel while the glove is cold, then flips to
#     3 within one frame and stays there for ~100% of fresh channels afterwards:
#     the shape of a magnetometer finishing calibration.
#   * [5:4] toggles 0<->3 with hand motion and nowhere else: the gyroscope,
#     which snaps to 3 whenever the glove is briefly still.
#   * [1:0] cannot be SYS: at the moment it reads 3 on every channel, [3:2] is
#     still 1, and SYS=3 requires all three subsystems at 3.
#
# Reading bits [7:6] instead is the trap this constant exists to prevent: it
# returns SYS, which sits near 0 on a calibrated glove and makes it look like
# none of its channels ever reach level 3 -- the gate then never opens.
MAG_LEVEL_SHIFT = 0
MAG_LEVEL_MASK = 0x03

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


def display_firmware_version(version: str | None) -> str | None:
    """Return the firmware version without the hand-side digit.

    The firmware encodes the hand side in the ones digit of the patch
    (``1.2.61`` = left, ``1.2.82`` = right, ``1.2.101`` = left).  That digit is
    only used for auto-detection (see :func:`firmware_hand_side`) and is dropped
    for display: ``1.2.61`` -> ``1.2.6``, ``1.2.101`` -> ``1.2.10``.
    Single-digit patches (``1.2.3``) and malformed input pass through unchanged.
    """
    if not version:
        return version
    real = firmware_real_version(version)
    if real is None:
        return version
    major, minor, patch = real
    return f"{major}.{minor}.{patch}"


# The v1.2.2 firmware emits joint order (wrist, thumb, index, middle, ring,
# pinky).  Keep the v1.0 physical-channel maps here so old JSON configurations
# continue to work: ``channel_to_hand_for_firmware`` composes them with the
# firmware's output permutation.  New configurations use the resulting map
# directly, which is the same for both hand sides.
LEGACY_CHANNEL_TO_HAND_BY_SIDE = {
    "left": (
        14, 13, 8, 15,
        2, 7, 9, 1,
        4, 3, 11, 0,
        5, 12, 10, 6,
    ),
    "right": (
        8, 7, 14, 9,
        11, 13, 15, 10,
        4, 12, 2, 0,
        5, 3, 1, 6,
    ),
}

# USB output slot -> pre-v1.2.2 firmware slot.  These are the ORDER_L/R
# tables shown in section A.3 of the supplied PDF.
FIRMWARE_OUTPUT_ORDER_BY_SIDE = {
    "left": (11, 1, 0, 3, 7, 4, 9, 8, 12, 15, 14, 10, 13, 5, 2, 6),
    "right": (11, 5, 2, 6, 14, 10, 13, 8, 12, 15, 7, 4, 9, 1, 0, 3),
}

NEW_FIRMWARE_CHANNEL_TO_HAND_BY_SIDE = {
    side: tuple(
        LEGACY_CHANNEL_TO_HAND_BY_SIDE[side][source]
        for source in FIRMWARE_OUTPUT_ORDER_BY_SIDE[side]
    )
    for side in FIRMWARE_OUTPUT_ORDER_BY_SIDE
}

DEFAULT_CHANNEL_TO_HAND = NEW_FIRMWARE_CHANNEL_TO_HAND_BY_SIDE["right"]


def decode_sequence_word(
        raw_sequence: int,
        version: str | tuple[int, ...] | list[int] | None = None,
) -> tuple[int, bool]:
    """Split a slim-header sequence word into (counter, magnetometer_ready).

    v1.2.12 and older carry a 15-bit counter plus the readiness flag.  v1.2.13
    and newer use all 16 bits as the counter and report magnetic calibration in
    a separate type-0x05 frame, so ``magnetometer_ready`` is true here.
    """

    word = int(raw_sequence)
    real = firmware_real_version(version) if version is not None else None
    if real is not None and real >= USB_FULL_SEQUENCE_FROM:
        return word & 0xFFFF, True
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
    # False while v1.2.10-v1.2.12 reports the magnetometer as not ready in bit
    # 15 (see :func:`decode_sequence_word`).  Legacy 32-bit frames and v1.2.13+
    # report True here; the latter exposes per-channel state through type 0x05.
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
class UsbMagTelemetry:
    """Decoded type-0x05 telemetry: one MAG level per physical IMU."""

    valid_mask: int
    levels_raw: tuple[int, ...]

    @property
    def level_fresh(self) -> tuple[bool, ...]:
        """Which levels this frame actually carried (bit k = IMU k fresh)."""

        return tuple(bool(self.valid_mask & (1 << k))
                     for k in range(MAG_TELEMETRY_IMU_COUNT))

    @property
    def mag_levels(self) -> tuple[int, ...]:
        return tuple((value >> MAG_LEVEL_SHIFT) & MAG_LEVEL_MASK
                     for value in self.levels_raw)


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
                        int.from_bytes(self.buffer[6:8], "little"), version)
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

                if message_type == USB_TYPE_MAG_TELEMETRY:
                    if (header_size != USB_SLIM_HEADER_SIZE
                            or payload_length not in MAG_TELEMETRY_PAYLOAD_SIZES):
                        length_rejected += 1
                        continue
                elif message_type in (USB_TYPE_IMU_Q14, USB_TYPE_ADC_MATRIX):
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


def validate_channel_to_hand(mapping: Iterable[int]) -> tuple[int, ...]:
    """Validate and freeze a physical-channel to MANO-joint permutation."""

    result = tuple(int(value) for value in mapping)
    if len(result) != TOTAL_IMU_COUNT:
        raise ValueError("channel_to_hand must contain exactly 16 entries")
    if sorted(result) != list(range(TOTAL_IMU_COUNT)):
        raise ValueError("channel_to_hand must be a permutation of 0..15")
    return result


def channel_to_hand_for_firmware(mapping: Iterable[int]) -> tuple[int, ...]:
    """Convert a persisted v1.0 map to the v1.2.2 joint-order map.

    A map already written for the new payload is returned unchanged.  This
    lets users upgrade the SDK without invalidating existing calibration JSON.
    """

    normalized = validate_channel_to_hand(mapping)
    for side, legacy_mapping in LEGACY_CHANNEL_TO_HAND_BY_SIDE.items():
        if normalized == legacy_mapping:
            return NEW_FIRMWARE_CHANNEL_TO_HAND_BY_SIDE[side]
    return normalized


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


def decode_mag_telemetry_payload(payload: bytes) -> UsbMagTelemetry:
    """Decode a type-0x05 payload: ``mask(2B)`` then 16 level bytes."""

    if len(payload) not in MAG_TELEMETRY_PAYLOAD_SIZES:
        raise ValueError(
            "mag telemetry payload must be one of "
            f"{MAG_TELEMETRY_PAYLOAD_SIZES} bytes, got {len(payload)}")
    start = MAG_TELEMETRY_VALID_MASK_SIZE
    valid_mask = int.from_bytes(payload[:start], "little")
    levels = tuple(int(value)
                   for value in payload[start:start + MAG_TELEMETRY_IMU_COUNT])
    return UsbMagTelemetry(valid_mask, levels)


@dataclass(frozen=True)
class MagTelemetrySnapshot:
    """Latest cached magnetic levels for the 16 physical IMUs.

    ``level_fresh`` describes only the most recently received type-0x05 frame.
    A level stays cached when its fresh bit is clear; ``level_seen`` tells that
    stale value apart from one that has never been received at all.
    """

    sequence: int
    device_timestamp_us: int
    host_timestamp_us: int
    levels_raw: tuple[int | None, ...]
    level_fresh: tuple[bool, ...]
    level_seen: tuple[bool, ...]
    # Stored rather than derived: a frame that carried no channels leaves the
    # verdict where it was (see MagTelemetryCache.update).
    ready: bool
    # The newest frame exactly as it came off the wire: its 16 CALIB_STAT bytes
    # with no merge and no decoding, paired with ``level_fresh`` for the mask it
    # arrived under.  ``levels_raw`` above is the cache and cannot be used for
    # this -- a channel the frame skipped keeps its older value there.  Recorded
    # verbatim by the capture so the byte layout can be audited from the file
    # rather than argued about.
    frame_levels_raw: tuple[int, ...] = ()

    @property
    def mag_levels(self) -> tuple[int | None, ...]:
        return tuple(None if value is None
                     else (value >> MAG_LEVEL_SHIFT) & MAG_LEVEL_MASK
                     for value in self.levels_raw)

    @property
    def fresh_count(self) -> int:
        """How many channels the newest frame -- this round -- carried."""

        return sum(1 for fresh in self.level_fresh if fresh)

    @property
    def fresh_ready_count(self) -> int:
        """Of those, how many report MAG level 3."""

        return sum(1 for fresh, level in zip(self.level_fresh, self.mag_levels)
                   if fresh and level == 3)

    @property
    def gate_counts(self) -> tuple[int, int]:
        """``(at 3, carried)`` for the newest 0x05 frame -- what the gate saw.

        What a caller shows next to a "not calibrated" prompt: both numbers are
        about this round, so the pair moves for the same reason the gate does.
        The channels the frame did not carry are in neither of them.
        """

        return self.fresh_ready_count, self.fresh_count

    @property
    def unread_channels(self) -> tuple[int, ...]:
        """The IMUs this round did not report, in physical channel order.

        Not a fault and not a level: the device carries about twelve of the
        sixteen per frame (MAG_TELEMETRY.md 2.4), and which ones is not
        predictable.  Named so a count out of twelve has an explanation on
        screen beside it.
        """

        return tuple(channel for channel, fresh in enumerate(self.level_fresh)
                     if not fresh)

    @property
    def ready_count(self) -> int:
        """How many channels have been seen *and* report MAG level 3.

        A diagnostic over the whole cache, not the gate: it counts cached
        levels too, and its denominator is all 16.
        """

        return sum(1 for seen, level in zip(self.level_seen, self.mag_levels)
                   if seen and level == 3)

    @property
    def uncalibrated_channels(self) -> tuple[int, ...]:
        """The physical IMUs that were read and are still below level 3."""

        return tuple(channel
                     for channel, (seen, level) in enumerate(
                         zip(self.level_seen, self.mag_levels))
                     if seen and level != 3)


class MagTelemetryCache:
    """Per-channel cache merging type-0x05 frames into a snapshot.

    MAG_TELEMETRY.md 2.4: the device reports per-frame freshness rather than a
    guaranteed stream -- not every level is carried by every frame -- so a level
    is cached until its channel is reported again: a stale value stays readable
    while a never-received one stays ``None``, and ``level_seen`` is what tells
    the two apart.
    """

    def __init__(self) -> None:
        # Own lock rather than borrowing the caller's: the two producers (the
        # USB transport reader thread and the calibration receiver thread) each
        # already hold a different outer lock while calling in.
        self._lock = threading.RLock()
        self._snapshot: MagTelemetrySnapshot | None = None

    @property
    def snapshot(self) -> MagTelemetrySnapshot | None:
        with self._lock:
            return self._snapshot

    def reset(self) -> None:
        """Forget everything; called when the link drops."""

        with self._lock:
            self._snapshot = None

    def update(self, frame, host_timestamp_us: int) -> MagTelemetrySnapshot:
        """Merge one parsed type-0x05 frame and return the merged snapshot.

        Raises :class:`ValueError` for a wrong-length payload (the caller
        decides whether that is fatal); the cache is left untouched.
        """

        decoded = decode_mag_telemetry_payload(frame.payload)
        with self._lock:
            previous = self._snapshot
            blank = (None,) * MAG_TELEMETRY_IMU_COUNT
            levels = list(previous.levels_raw if previous is not None else blank)
            seen = list(previous.level_seen if previous is not None
                        else (False,) * MAG_TELEMETRY_IMU_COUNT)
            for channel in range(MAG_TELEMETRY_IMU_COUNT):
                if decoded.level_fresh[channel]:
                    levels[channel] = decoded.levels_raw[channel]
                    seen[channel] = True
            fresh_ready = sum(
                1 for fresh, value in zip(decoded.level_fresh, decoded.levels_raw)
                if fresh and ((value >> MAG_LEVEL_SHIFT) & MAG_LEVEL_MASK) == 3)
            fresh_count = sum(1 for fresh in decoded.level_fresh if fresh)
            if fresh_count:
                ready = mag_gate_met(fresh_ready, fresh_count)
            else:
                # A frame whose mask is empty carried no reading at all, so it
                # is not evidence of anything -- falling to "not ready" here
                # would start a circles countdown off a frame that said nothing.
                ready = previous.ready if previous is not None else False
            self._snapshot = MagTelemetrySnapshot(
                sequence=int(frame.sequence),
                device_timestamp_us=int(frame.timestamp_us),
                host_timestamp_us=int(host_timestamp_us),
                levels_raw=tuple(levels),
                level_fresh=decoded.level_fresh,
                level_seen=tuple(seen),
                ready=ready,
                frame_levels_raw=tuple(int(value)
                                       for value in decoded.levels_raw),
            )
            return self._snapshot


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
    firmware sends a 524-byte payload with a 12-byte metadata prefix that carries
    those values itself.
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


def remap_physical_to_hand(
        quaternions_xyzw: np.ndarray,
        physical_valid_mask: int,
        channel_to_hand: Iterable[int] = DEFAULT_CHANNEL_TO_HAND,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder firmware joint-order slots and valid bits into MANO joint order."""

    quaternions = np.asarray(quaternions_xyzw, dtype=float)
    if quaternions.shape != (TOTAL_IMU_COUNT, IMU_COMPONENT_COUNT):
        raise ValueError("quaternions must have shape (16, 4)")
    mapping = channel_to_hand_for_firmware(channel_to_hand)

    mano_quaternions = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0]), (TOTAL_IMU_COUNT, 1))
    mano_valid = np.zeros(TOTAL_IMU_COUNT, dtype=bool)
    plausible = plausible_quaternion_mask(quaternions)

    for physical_channel, mano_joint in enumerate(mapping):
        mano_quaternions[mano_joint] = quaternions[physical_channel]
        mano_valid[mano_joint] = bool(
            physical_valid_mask & (1 << physical_channel)) and plausible[physical_channel]

    return mano_quaternions, mano_valid


def readable_imu_mask(
        quaternions_xyzw: np.ndarray,
        channel_to_hand: Iterable[int] = DEFAULT_CHANNEL_TO_HAND,
) -> np.ndarray:
    """Return a per-MANO-joint mask of which IMUs produced readable data.

    "Readable" means the decoded quaternion is finite with a plausible norm,
    i.e. the firmware actually produced usable data for that channel this
    frame.  Unlike :func:`remap_physical_to_hand`, this does not consult the
    frame's flags word: a channel whose flags bit is still clear (for example
    an IMU the firmware marked offline) recovers as soon as its quaternion
    becomes readable again, instead of staying reported missing until a
    device restart.
    """

    mapping = channel_to_hand_for_firmware(channel_to_hand)
    plausible = plausible_quaternion_mask(quaternions_xyzw)
    readable = np.zeros(TOTAL_IMU_COUNT, dtype=bool)
    for physical_channel, mano_joint in enumerate(mapping):
        readable[mano_joint] = plausible[physical_channel]
    return readable


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
