"""STM32 glove USB CDC framing and IMU payload decoding.

The firmware multiplexes status, IMU and pressure-matrix messages on one CDC
byte stream.  This module deliberately owns only the wire protocol; serial-port
opening and MANO processing remain in :mod:`modules.hand.hand`.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

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

USB_TYPE_STATUS = 0x01
USB_TYPE_IMU_Q14 = 0x02
USB_TYPE_ADC_MATRIX = 0x03

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


def firmware_hand_side(version: str) -> str | None:
    """Return the hand side encoded in a firmware version, or ``None``.

    A two-digit patch carries the side in its ones digit (1 = left, 2 = right),
    e.g. ``1.2.91`` -> "left", ``1.2.62`` -> "right".  A single-digit patch
    (``1.2.3``) carries no side, so this returns ``None``.
    """

    components = parse_firmware_version(version)
    if components is None:
        return None
    patch = components[2]
    if not (10 <= patch <= 99):
        return None
    digit = patch % 10
    if digit == 1:
        return "left"
    if digit == 2:
        return "right"
    return None


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


@dataclass(frozen=True)
class UsbCdcFrame:
    """One CRC-validated frame extracted from the CDC byte stream."""

    version: str
    message_type: int
    sequence: int
    timestamp_us: int
    flags: int
    payload: bytes

    @property
    def firmware_version(self) -> str:
        """Compatibility alias used by older SDK callers."""

        return self.version


@dataclass(frozen=True)
class UsbAdcMatrixFrame:
    """Decoded STM32 16x16 tactile matrix payload."""

    sequence: int
    scan_time_us: int
    samples: np.ndarray


def crc16_ccitt_false(data: bytes | bytearray | memoryview) -> int:
    """Return CRC16-CCITT-FALSE (poly 0x1021, init 0xFFFF)."""

    crc = 0xFFFF
    for value in data:
        crc ^= int(value) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
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

            if len(self.buffer) < USB_HEADER_SIZE:
                break

            payload_length = int.from_bytes(self.buffer[14:16], "little")
            if payload_length > USB_MAX_PAYLOAD:
                del self.buffer[0]
                self.discarded_bytes += 1
                self.length_errors += 1
                continue

            total_length = USB_HEADER_SIZE + payload_length + USB_CRC_SIZE
            if len(self.buffer) < total_length:
                break

            crc_offset = USB_HEADER_SIZE + payload_length
            expected_crc = int.from_bytes(
                self.buffer[crc_offset:crc_offset + USB_CRC_SIZE], "little")
            actual_crc = crc16_ccitt_false(self.buffer[:crc_offset])
            if actual_crc != expected_crc:
                del self.buffer[0]
                self.discarded_bytes += 1
                self.crc_errors += 1
                continue

            frames.append(UsbCdcFrame(
                version=".".join(
                    str(value) for value in self.buffer[2:5]),
                message_type=int(self.buffer[5]),
                sequence=int.from_bytes(self.buffer[6:10], "little"),
                timestamp_us=int.from_bytes(self.buffer[10:14], "little"),
                flags=int.from_bytes(self.buffer[16:18], "little"),
                payload=bytes(self.buffer[USB_HEADER_SIZE:crc_offset]),
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
    """Decode the fixed 128-byte payload into SciPy-order XYZW quaternions."""

    if len(payload) != IMU_Q14_PAYLOAD_SIZE:
        raise ValueError(
            f"STM32 IMU payload must be {IMU_Q14_PAYLOAD_SIZE} bytes, "
            f"got {len(payload)}")

    raw = np.asarray(
        struct.unpack("<64h", payload), dtype=np.float64
    ).reshape(TOTAL_IMU_COUNT, IMU_COMPONENT_COUNT)
    wxyz = raw * Q14_SCALE
    return wxyz[:, [1, 2, 3, 0]]


def decode_adc_matrix_payload(payload: bytes) -> UsbAdcMatrixFrame:
    """Decode the STM32 12-byte metadata plus 256 little-endian ADC values."""

    if len(payload) != ADC_MATRIX_PAYLOAD_SIZE:
        raise ValueError(
            f"STM32 ADC matrix payload must be {ADC_MATRIX_PAYLOAD_SIZE} bytes, "
            f"got {len(payload)}")
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
