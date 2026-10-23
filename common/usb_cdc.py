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
import zlib

import numpy as np


# Explicit export list: ``sdk.io.usb_protocol`` is a thin re-export shim over
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
    "USB_TYPE_PRESSURE_FIT_PARAMS", "USB_TYPE_PARAM_ACK",
    "USB_TYPE_PARAM_COMMAND", "USB_PARAM_COMMAND_SIZE",
    "PARAM_RAW_COMMAND_SIZE", "PARAM_RAW_ACK_PAYLOAD_SIZE",
    "PARAM_RAW_BLOCK_MAX_SIZE",
    "USB_PRESSURE_FIT_PARAMS_FROM",
    "PRESSURE_FIT_SET_COUNT", "PRESSURE_FIT_CHANNEL_COUNT",
    "PRESSURE_FIT_META_SIZE", "PRESSURE_FIT_WRITE_INDEX_SIZE",
    "PRESSURE_FIT_BLOB_SIZE", "PRESSURE_FIT_PAYLOAD_SIZE",
    "PRESSURE_FIT_PAYLOAD_SIZES", "PRESSURE_FIT_FORMAT_VERSION",
    "PRESSURE_FIT_FLAG_VALID", "PRESSURE_FIT_Q12_SCALE",
    "PRESSURE_FIT_Q12_MIN", "PRESSURE_FIT_Q12_MAX",
    "PRESSURE_FIT_SHARD_SIZE", "PRESSURE_FIT_SHARD_COUNT",
    "PARAM_ACK_PAYLOAD_SIZE", "PARAM_ACK_PAYLOAD_SIZES",
    "PARAM_SUBCMD_WRITE_SHARD", "PARAM_SUBCMD_READ_SET",
    "PARAM_SUBCMD_CLEAR_ALL", "PARAM_SUBCMD_QUERY_STATUS",
    "PARAM_SUBCMD_START_FIT",
    "PARAM_SUBCMDS",
    "PARAM_STATUS_OK", "PARAM_STATUS_RANGE", "PARAM_STATUS_FLASH_FAIL",
    "PARAM_STATUS_BAD_SET", "PARAM_STATUS_CRC_MISMATCH", "PARAM_STATUS_NAMES",
    "hand_side_from_code", "pressure_fit_crc32", "pressure_fit_quantize",
    "TOTAL_IMU_COUNT", "IMU_COMPONENT_COUNT", "IMU_Q14_PAYLOAD_SIZE",
    "IMU_VALID_MASK_SIZE", "IMU_SLIM_PAYLOAD_SIZE", "Q14_SCALE",
    "ADC_MATRIX_ROWS", "ADC_MATRIX_COLS", "ADC_MATRIX_CELLS",
    "ADC_MATRIX_META_SIZE",
    "ADC_MATRIX_PAYLOAD_SIZE", "ADC_MATRIX_SLIM_PAYLOAD_SIZE",
    "ADC_MATRIX_TIER_SPARSE", "ADC_MATRIX_TIER_BITMAP",
    "ADC_MATRIX_TIER_FULL", "ADC_MATRIX_TIER_LEGACY",
    "carries_matrix_patches",
    "ADC_MATRIX_SPARSE_MAX_CELLS", "ADC_MATRIX_BITMAP_MAX_CELLS",
    "ADC_MATRIX_BITMAP_MARKER", "ADC_MATRIX_BITMAP_SIZE",
    "adc_matrix_bitmap_value_size",
    "ADC_MATRIX_SPARSE_MIN_SIZE", "ADC_MATRIX_SPARSE_MAX_SIZE",
    "ADC_MATRIX_BITMAP_MIN_SIZE", "ADC_MATRIX_BITMAP_MAX_SIZE",
    "MAG_TELEMETRY_PAYLOAD_SIZE", "MAG_TELEMETRY_PAYLOAD_SIZES",
    "MAG_TELEMETRY_IMU_COUNT", "MAG_TELEMETRY_VALID_MASK_SIZE",
    "MAG_READY_MIN_NUMERATOR", "MAG_READY_MIN_DENOMINATOR", "mag_gate_met",
    "MAG_LEVEL_SHIFT", "MAG_LEVEL_MASK",
    # Decoded frames
    "UsbCdcFrame", "UsbAdcMatrixFrame",
    "UsbDelayResponse", "UsbMagTelemetry",
    "UsbLinkLatency", "MagTelemetrySnapshot", "MagTelemetryCache",
    "UsbPressureFitParams", "UsbParamAck", "UsbRawParamAck",
    "PressureFitSnapshot", "PressureFitCache",
    # Parser and codecs
    "UsbCdcFrameParser", "crc16_ccitt_false", "encode_host_probe",
    "decode_imu_q14_payload", "decode_adc_matrix_payload",
    "decode_adc_matrix_patch", "adc_matrix_payload_tier",
    "is_adc_matrix_payload_size", "is_adc_matrix_keyframe",
    "AdcMatrixAssembler",
    "decode_delay_response", "decode_mag_telemetry_payload",
    "decode_sequence_word", "plausible_quaternion_mask",
    "link_latency_from_response",
    "encode_param_command", "encode_raw_param_command",
    "decode_pressure_fit_payload", "decode_param_ack_payload",
    "decode_raw_param_ack_payload", "encode_pressure_fit_blob",
    "SYNTHETIC_PRESSURE_FIT_SEED", "synthetic_pressure_fit_blob",
    "pressure_fit_set_summary",
    # Version helpers
    "parse_firmware_version", "firmware_real_version", "is_supported_version",
    "is_slim_format", "supports_latency_probe", "firmware_hand_side",
    "display_firmware_version", "supports_pressure_fit_params",
    # Channel/hand mapping
    "BLUETOOTH_DONGLE_VID", "BLUETOOTH_DONGLE_PID", "glove_link_kind",
    "LINK_KIND_USB", "LINK_KIND_BLUETOOTH",
    "STM32_LINK_IDS", "LEGACY_CHANNEL_TO_HAND_BY_SIDE",
    "NEW_FIRMWARE_CHANNEL_TO_HAND_BY_SIDE", "FIRMWARE_OUTPUT_ORDER_BY_SIDE",
    "DEFAULT_CHANNEL_TO_HAND", "validate_channel_to_hand",
    "channel_to_hand_for_firmware", "remap_physical_to_hand",
    "readable_imu_mask", "mag_ready_mask",
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

# The two link kinds, by name.  They are the same values
# :func:`glove_link_kind` returns and the same ones ``common.types`` exports as
# ``LINK_USB``/``LINK_BLUETOOTH``; this module cannot import that one (it
# imports *this* one), so the pair is written twice and ``self_test``'s
# ``unified_frame_selftest`` asserts the two agree rather than trusting them to.
LINK_KIND_USB = "usb"
LINK_KIND_BLUETOOTH = "bluetooth"

# Every (vid, pid, link kind) that can carry a glove stream.  ``glove_link_kind``
# is the single place where the wired and Bluetooth transports are told apart.
STM32_LINK_IDS: tuple[tuple[int, int, str], ...] = (
    (STM32_USB_VID, STM32_USB_PID, LINK_KIND_USB),
    (BLUETOOTH_DONGLE_VID, BLUETOOTH_DONGLE_PID, LINK_KIND_BLUETOOTH),
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


def carries_matrix_patches(link_kind: str | None) -> bool:
    """Does this link's firmware compress the pressure matrix into patches?

    The one place the two 0x03 dialects are told apart.  ``"bluetooth"`` is the
    build that sends the sparse and bitmap tiers; everything else -- the wired
    ``"usb"`` build, and a ``None`` the enumerator could not classify -- gets
    the documented fixed-picture protocol (v1.2.16 4.4: 256 12-bit samples,
    exactly 384 bytes, every scan, no patches).

    The fallback is the *strict* dialect on purpose.  A link nobody could name
    is not evidence that somebody compressed it, and guessing "patches" there
    would let a one-byte payload be published as a scan the device never took;
    guessing "pictures" costs at worst a frozen matrix on a link whose own
    firmware never needed compressing anyway (see the 0x03 comment above the
    tier constants).  The caller that *knows* the link -- every live reader --
    passes it.
    """

    return link_kind == LINK_KIND_BLUETOOTH

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

# The 0x03 payload is not one layout but three, and the frame type alone does
# not say which: a tier is identified by its own shape, not by a version or a
# flag byte.
#
#   sparse  count(1) + count x [index(1) + value(2)]        len = 1 + 3*count
#   bitmap  0xFF + 32-byte presence map + 12-bit-packed     len = 33 + packed
#           values for the set bits, two per three bytes
#   full    256 12-bit samples, two per three bytes         384 bytes
#
# Which of them a link may carry is decided by the *link*, not by the payload:
# the Bluetooth build (v2.0.7) is the one that compresses the matrix, because
# RF hopping channels that go bad drop whole blocks and a smaller scan leaves
# the IMU frames alone (``firmware/bluetooth/README.md``), while the wired build
# (v1.2.16) documents one fixed 384-byte picture per scan and no patches at all
# (``数据采集手套_数据接口说明_v1.2.16`` 4.4).  So every tier-aware entry point
# below takes ``link_kind`` and is, in effect, two parsers: see
# :func:`carries_matrix_patches`.  Reading a link with the other one's dialect
# is not a harmless widening -- a one-byte sparse payload is a perfectly
# well-formed *frame* under the wrong dialect and would be published as a scan
# the device never took, while under the right one it is a length error worth
# counting.
#
# On the Bluetooth link the device picks between them by how many cells moved --
# the sparse list for a handful, the presence map for more, and a full picture
# when so much moved that a patch would not pay for itself, at least every 30
# scans and at the start of the stream.  That crossover is the *device's*
# policy; the host does not gate on it, only on the shapes.  See
# :func:`is_adc_matrix_payload_size` for why.
#
# Sparse and bitmap are patches, not pictures: they carry only the cells that
# moved, and a consumer has to merge them onto the last picture it saw (see
# :class:`AdcMatrixAssembler`).  Values are absolute 12-bit ADC samples in both
# tiers -- only the sparse tier spends 16 bits on one -- so merging is an
# overwrite, never an accumulation.  The contract makes merging safe: a scan is
# preceded by a full frame at the start of the stream and at least every 30
# frames after that, so a patch can never be older than one keyframe.

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
# v1.2.16 adds two uplink types and one downlink command byte:
#   0x06  one stored pressure-fit parameter set (k[256], b[256])
#   0x07  the answer to one 0x51 parameter command
#   0x51  the parameter command itself (host -> device)
# 0x51 is a *downlink command byte only*, exactly like 0x50 above it: it is
# never compared against a frame's ``message_type``, only its 0x07 answer is.
USB_TYPE_PRESSURE_FIT_PARAMS = 0x06
USB_TYPE_PARAM_ACK = 0x07
USB_TYPE_HOST_PROBE = 0x50
USB_TYPE_PARAM_COMMAND = 0x51

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


def hand_side_from_code(code: int) -> str | None:
    """Map the wire hand code (1 = left, 2 = right) to a side name."""

    return {1: "left", 2: "right"}.get(int(code))


def pressure_fit_crc32(data: bytes | bytearray | memoryview) -> int:
    """Return the CRC-32/ISO-HDLC (zlib/PKZIP) of a parameter blob.

    Deliberately *not* :func:`crc16_ccitt_false`: that is the frame trailer's
    algorithm, over different bytes.  The v1.2.16 parameter CRC32 covers the
    1024-byte k+b blob alone -- no frame header, no set number, no command
    wording (spec 5.2).
    """

    return zlib.crc32(bytes(data)) & 0xFFFFFFFF


# The largest magnitude Q12 can hold, so the smallest physical value a stored
# k or b can express is about -8.0 and the largest +7.9998.
PRESSURE_FIT_Q12_MAX = 32767
PRESSURE_FIT_Q12_MIN = -32768


def pressure_fit_quantize(values, *, name: str = "values") -> np.ndarray:
    """Q12-encode physical k/b into the signed int16 words the device stores.

    Refuses anything int16 cannot hold instead of wrapping.  That refusal is the
    whole point: a silent ``astype(np.int16)`` wraps modulo 2**16 and the
    resulting garbage still passes the device's CRC32 check, because the CRC
    covers exactly the bytes the host itself sent -- the corruption would only
    surface later as pressure readings that are simply wrong.
    """

    scaled = np.rint(np.asarray(values, dtype=np.float64) * 4096.0)
    if not np.all(np.isfinite(scaled)):
        raise ValueError(f"{name} contains non-finite values")
    if scaled.size and (scaled.min() < PRESSURE_FIT_Q12_MIN
                        or scaled.max() > PRESSURE_FIT_Q12_MAX):
        raise ValueError(
            f"{name} does not fit Q12 (int16/4096): the representable range is "
            f"[{PRESSURE_FIT_Q12_MIN / 4096.0}, "
            f"{PRESSURE_FIT_Q12_MAX / 4096.0}]"
        )
    return scaled.astype(np.int16)


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

# v1.2.16 type 0x06: one stored fit-parameter set per frame, always 1032 bytes.
#   套号(1) 标志(1) 格式(1) 手型(1) 写入序号(4) k[256] int16 b[256] int16
# The 256 channels are in the same order as a type-0x03 pressure frame (16x16
# row-major) and, unlike every other uplink type, are NOT remapped by hand side
# -- left and right send the same channel order (spec 4.7/6.2).
PRESSURE_FIT_SET_COUNT = 6
PRESSURE_FIT_CHANNEL_COUNT = 256
PRESSURE_FIT_META_SIZE = 8
PRESSURE_FIT_WRITE_INDEX_SIZE = 4
PRESSURE_FIT_BLOB_SIZE = PRESSURE_FIT_CHANNEL_COUNT * 2 * 2  # k + b, int16
PRESSURE_FIT_PAYLOAD_SIZE = PRESSURE_FIT_META_SIZE + PRESSURE_FIT_BLOB_SIZE
PRESSURE_FIT_PAYLOAD_SIZES = (PRESSURE_FIT_PAYLOAD_SIZE,)
PRESSURE_FIT_FORMAT_VERSION = 1
# flags bit0: this set currently holds parameters.  A set that was never
# written reports 0 here and carries an all-zero k/b that must not be used.
PRESSURE_FIT_FLAG_VALID = 0x01
# k and b are signed Q12: physical value = raw / 4096.
PRESSURE_FIT_Q12_SCALE = 1.0 / 4096.0
# One shard carries at most 256 bytes, so a set is always four shards.
PRESSURE_FIT_SHARD_SIZE = 256
PRESSURE_FIT_SHARD_COUNT = PRESSURE_FIT_BLOB_SIZE // PRESSURE_FIT_SHARD_SIZE

# Legacy v1.2.16 type 0x07: one answer per 0x51 command, 12 bytes.
#   子命令(1) 套号(1) 状态(1) 有效掩码(1) 写入序号(4) CRC32(4)
# ``valid_mask`` here is "which of the six parameter sets are stored" -- it is
# unrelated to ``UsbCdcFrame.valid_mask`` (the IMU joint mask) despite the name.
PARAM_ACK_PAYLOAD_SIZE = 12
PARAM_RAW_ACK_PAYLOAD_SIZE = 8
# The current firmware uses the compact raw-block acknowledgement:
#   subcommand(1) status(1) stored_length(2) crc32(4)
# Keep the legacy 12-byte shape accepted as well because older firmware and
# the command-line compatibility tools still use it.
PARAM_ACK_PAYLOAD_SIZES = (PARAM_RAW_ACK_PAYLOAD_SIZE, PARAM_ACK_PAYLOAD_SIZE)

# v1.2.16 host -> device 0x51 subcommands.
PARAM_SUBCMD_WRITE_SHARD = 0x01
PARAM_SUBCMD_READ_SET = 0x02
PARAM_SUBCMD_CLEAR_ALL = 0x03
PARAM_SUBCMD_QUERY_STATUS = 0x04
# "Start fitting": the host asks the device to run a fit of its own.  This is a
# *contract with the firmware team*, not a reading of the spec -- v1.2.16 5.2
# lists only 01/02/03/04, and 4.7 says the firmware "only stores and returns, it
# does no fitting of any kind".  The command is a bare 13-byte 0x51 with no data
# area, and it is a *send*: one downlink and its 0x07 ack, nothing read back.
#
# It has the same number as the *uplink* frame type 0x06 (a set of parameters)
# by arithmetic accident, not by design: equal numbers, unrelated fields.  Both
# are pinned by ``tools/self_test.py::pressure_fit_start_selftest`` so that
# "tidying up" either one is a visible change rather than a silent one.
PARAM_SUBCMD_START_FIT = 0x06
PARAM_SUBCMDS = (
    PARAM_SUBCMD_WRITE_SHARD,
    PARAM_SUBCMD_READ_SET,
    PARAM_SUBCMD_CLEAR_ALL,
    PARAM_SUBCMD_QUERY_STATUS,
    PARAM_SUBCMD_START_FIT,
)

# Legacy v1.2.16 0x51 command: 13 bytes of header/suffix, plus any data area.  Same
# shape as the 0x50 probe above.
USB_PARAM_COMMAND_SIZE = 13
# Current raw-block command: A5 5A 51 subcommand length(2) crc32(4) data.
PARAM_RAW_COMMAND_SIZE = 10
# v1.2.16(2) reserves at most 780 bytes in the firmware parameter page.
PARAM_RAW_BLOCK_MAX_SIZE = 780

# v1.2.16 0x07 status values.  Only 0 is success.
PARAM_STATUS_OK = 0
PARAM_STATUS_RANGE = 1
PARAM_STATUS_FLASH_FAIL = 2
PARAM_STATUS_BAD_SET = 3
PARAM_STATUS_CRC_MISMATCH = 4
PARAM_STATUS_NAMES = {
    PARAM_STATUS_OK: "ok",
    PARAM_STATUS_RANGE: "offset/length out of range",
    PARAM_STATUS_FLASH_FAIL: "flash write failed",
    PARAM_STATUS_BAD_SET: "invalid set index",
    PARAM_STATUS_CRC_MISMATCH: "blob crc32 mismatch",
}

# First firmware that knows the 0x51 parameter command and emits 0x06/0x07;
# see :func:`supports_pressure_fit_params`.
USB_PRESSURE_FIT_PARAMS_FROM = (1, 2, 16)

TOTAL_IMU_COUNT = 16
IMU_COMPONENT_COUNT = 4
IMU_Q14_PAYLOAD_SIZE = TOTAL_IMU_COUNT * IMU_COMPONENT_COUNT * 2
Q14_SCALE = 1.0 / 16384.0
ADC_MATRIX_ROWS = 16
ADC_MATRIX_COLS = 16
ADC_MATRIX_CELLS = ADC_MATRIX_ROWS * ADC_MATRIX_COLS
ADC_MATRIX_META_SIZE = 12
ADC_MATRIX_PAYLOAD_SIZE = (
    ADC_MATRIX_META_SIZE + ADC_MATRIX_CELLS * 2)

# Tier names, used as values (not flags) so a consumer can tell a keyframe from
# a patch without re-deriving the length arithmetic.
ADC_MATRIX_TIER_SPARSE = "sparse"
ADC_MATRIX_TIER_BITMAP = "bitmap"
ADC_MATRIX_TIER_FULL = "full"
# The legacy 18-byte-header layout carries a 524-byte block with a 12-byte
# metadata prefix and no patches at all; it keeps its own tier name so a
# keyframe from it is not mistaken for a slim one when its header is read back.
ADC_MATRIX_TIER_LEGACY = "legacy"

# Tier ceilings, and they are *shape* ceilings rather than the device's own
# crossover (which is a handful of cells for the sparse list and more for the
# map, with a full picture above that).  What bounds them here is that a patch
# has to be strictly shorter than a full picture, or its length would no longer
# say which tier it is: the sparse list spends three bytes per cell plus one and
# so stops at 127 cells, and the presence map's 234th point would pack to
# exactly the full picture's 384 bytes.  233 is therefore the last count the map
# can carry -- a 234-point patch is not a shape the host could recognise, which
# is why the device must never send one.
ADC_MATRIX_SPARSE_MAX_CELLS = 127
ADC_MATRIX_BITMAP_MAX_CELLS = 233
ADC_MATRIX_BITMAP_MARKER = 0xFF
ADC_MATRIX_BITMAP_SIZE = 32


def adc_matrix_bitmap_value_size(count: int) -> int:
    """Bytes the packed 12-bit values of a ``count``-point bitmap occupy.

    Pairs share three bytes and an odd point left over spends only the two its
    own 12 bits need -- the same truncation as stopping a pair early, not a
    second encoding.
    """

    count = int(count)
    return 3 * (count // 2) + 2 * (count % 2)


# `1 + 3 * count` for a sparse payload, and marker + map + packed values for a
# bitmap one.  The sparse minimum of one byte is ``count = 0`` -- a scan in
# which nothing moved, which the device does send and which is a reading like
# any other: it says the picture is unchanged, and a host that dropped it would
# turn a still hand's 60 scans a second into the two keyframes a second that
# the patches were never supposed to cost.
ADC_MATRIX_SPARSE_MIN_SIZE = 1
ADC_MATRIX_SPARSE_MAX_SIZE = 1 + 3 * ADC_MATRIX_SPARSE_MAX_CELLS
# The map's floor is the sentinel, the 32-byte map, and the two bytes one
# changed cell's value needs -- not the all-clear map's 33.  A scan in which
# nothing moved is what the *sparse* shape says (its count of zero), and the
# firmware sends it that way, so an all-clear map is not a shape this contract
# produces and the gate does not claim its length.
ADC_MATRIX_BITMAP_MIN_SIZE = (
    1 + ADC_MATRIX_BITMAP_SIZE + adc_matrix_bitmap_value_size(1))
ADC_MATRIX_BITMAP_MAX_SIZE = (
    1 + ADC_MATRIX_BITMAP_SIZE
    + adc_matrix_bitmap_value_size(ADC_MATRIX_BITMAP_MAX_CELLS))


def is_adc_matrix_payload_size(size: int, *, link_kind: str | None) -> bool:
    """Can this link's 0x03 contract put ``size`` bytes on the wire?

    This is the parser's gate, so it answers from the length alone: the tiers
    have to be told apart *after* the frame is known to be well formed, and a
    gate that needed the payload's first byte could not run until the whole
    frame had arrived.

    Two answers, one per dialect (see :func:`carries_matrix_patches`):

    * a Bluetooth link can put down 384 bytes, the sparse list's 4k+1 up to a
      byte short of a picture, or the map's marker-and-packed lengths below it;
    * a wired link can put down 384 bytes and nothing else.  v1.2.16 4.4 has one
      shape, one size, every scan, so any other length there is a frame that is
      not this protocol's -- a resynchronisation, not a reading.

    On the Bluetooth side the range is deliberately **wider than the device's
    crossover**.  The device chooses its tier by how many cells moved, but that
    is its policy, not the wire format, and the two are not the same question: a
    firmware that retuned the crossover would keep sending perfectly well-formed
    patches, and a gate that encoded the old thresholds would charge every one
    of them as an impossible length -- a clean link reported as a broken one,
    with a phantom lost frame per scan to go with it.
    """

    size = int(size)
    if size == ADC_MATRIX_SLIM_PAYLOAD_SIZE:
        return True
    if not carries_matrix_patches(link_kind):
        return False
    if (ADC_MATRIX_SPARSE_MIN_SIZE <= size <= ADC_MATRIX_SPARSE_MAX_SIZE
            and (size - 1) % 3 == 0):
        return True
    return (ADC_MATRIX_BITMAP_MIN_SIZE <= size <= ADC_MATRIX_BITMAP_MAX_SIZE
            and size % 3 in (0, 2))


def _bitmap_point_count(payload: bytes | bytearray | memoryview) -> int | None:
    """Number of changed cells a bitmap tier's presence map claims, or ``None``."""

    if len(payload) < 1 + ADC_MATRIX_BITMAP_SIZE:
        return None
    return sum(
        bin(int(byte)).count("1")
        for byte in payload[1:1 + ADC_MATRIX_BITMAP_SIZE])


def adc_matrix_payload_tier(
        payload: bytes | bytearray | memoryview,
        *,
        link_kind: str | None) -> str | None:
    """Name the 0x03 tier a payload belongs to, or ``None`` if it is none of them.

    Each tier is checked against its *own* arithmetic rather than its leading
    byte alone, and the checks are ordered so that no tier's shape can be
    claimed by another:

    * sparse needs ``1 + 3 * count`` bytes for the count in its first byte, so a
      384-byte keyframe -- whose first byte is a sample's low bits -- can never
      land here: 384 is not ``1 mod 3``, whatever the byte happens to say;
    * bitmap needs the ``0xFF`` marker *and* a presence map whose popcount
      agrees exactly with the length of the value section behind it, so a full
      picture that happens to start ``FF`` is not read as a patch by accident:
      its next 32 bytes would have to carry precisely the right number of set
      bits *and* the length would have to match that count's packing;

    anything else is ``None``, including the 524-byte legacy block, which is a
    keyframe with its own metadata prefix and is reported as
    :data:`ADC_MATRIX_TIER_LEGACY`.

    The count ceilings here are the shapes' own, not the device's crossover: a
    list of 30 cells is a well-formed sparse payload even though this firmware
    would have sent a map for it.  See :func:`is_adc_matrix_payload_size`.

    The two patch tiers exist only on a link whose firmware compresses at all,
    so a wired link answers ``full`` or ``None`` and never a patch -- see
    :func:`carries_matrix_patches`.  This is the second gate, not a duplicate of
    the first: the parser's is about what a length could be, and this one is
    about what these particular bytes are, so a patch shape arriving on a wired
    link is refused here even by a caller that fed the payload in directly.
    """

    size = len(payload)
    if size == ADC_MATRIX_PAYLOAD_SIZE:
        return ADC_MATRIX_TIER_LEGACY
    if size < 1:
        return None

    if carries_matrix_patches(link_kind):
        marker = int(payload[0])
        if size == 1 + 3 * marker and size <= ADC_MATRIX_SPARSE_MAX_SIZE:
            return ADC_MATRIX_TIER_SPARSE

        if marker == ADC_MATRIX_BITMAP_MARKER:
            points = _bitmap_point_count(payload)
            if (points is not None
                    and 1 <= points <= ADC_MATRIX_BITMAP_MAX_CELLS
                    and size == (1 + ADC_MATRIX_BITMAP_SIZE
                                 + adc_matrix_bitmap_value_size(points))):
                return ADC_MATRIX_TIER_BITMAP

    if size == ADC_MATRIX_SLIM_PAYLOAD_SIZE:
        return ADC_MATRIX_TIER_FULL
    return None


def is_adc_matrix_keyframe(tier: str | None) -> bool:
    """Is ``tier`` a self-contained picture rather than a patch?"""

    return tier in (ADC_MATRIX_TIER_FULL, ADC_MATRIX_TIER_LEGACY)

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


def supports_pressure_fit_params(
        version: str | tuple[int, ...] | list[int]) -> bool:
    """Return whether ``version`` knows the v1.2.16 0x51 parameter command.

    v1.2.16 is the first firmware that answers 0x51, so the command is only
    ever sent to a device that has announced it or newer; older firmware would
    see those 13+ bytes injected into its stream as garbage.

    The gate inherits :func:`firmware_real_version`'s side-digit rule, so both
    spellings of this release land on the same answer: the side-encoded
    ``1.2.161``/``1.2.162`` strip to ``(1, 2, 16)`` and a bare ``1.2.16`` is
    already there.  A bare patch whose ones digit happens to be 1 or 2 would be
    read as side-encoded and strip -- ``1.2.21`` -> ``(1, 2, 2)`` -- but this
    firmware has encoded the side since v1.2.6, so that spelling never occurs
    on the wire.
    """

    real = firmware_real_version(version)
    return real is not None and real >= USB_PRESSURE_FIT_PARAMS_FROM


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
class UsbPressureFitParams:
    """Decoded v1.2.16 type 0x06: one stored fit-parameter set (k[256], b[256]).

    ``k``/``b`` are the raw signed-16 Q12 words exactly as stored -- the wire
    truth, so a decode/encode round trip is byte-exact.  Callers wanting
    physical values use :attr:`k_scaled`/:attr:`b_scaled`.

    The 256 channels follow the 0x03 pressure frame's order (16x16 row-major,
    no hand-side remap), so left and right store the same channel order.
    """

    set_index: int
    flags: int
    format_version: int
    hand: int
    write_index: int
    k: np.ndarray
    b: np.ndarray

    @property
    def valid(self) -> bool:
        """Whether this set actually holds parameters (flags bit0)."""

        return bool(self.flags & PRESSURE_FIT_FLAG_VALID)

    @property
    def hand_side(self) -> str | None:
        return hand_side_from_code(self.hand)

    @property
    def supported_format(self) -> bool:
        return self.format_version == PRESSURE_FIT_FORMAT_VERSION

    @property
    def k_scaled(self) -> np.ndarray:
        return self.k.astype(np.float64) * PRESSURE_FIT_Q12_SCALE

    @property
    def b_scaled(self) -> np.ndarray:
        return self.b.astype(np.float64) * PRESSURE_FIT_Q12_SCALE

    @property
    def blob(self) -> bytes:
        """The 1024 bytes the device stores: k little-endian int16, then b."""

        return self.k.astype("<i2").tobytes() + self.b.astype("<i2").tobytes()

    @property
    def crc32(self) -> int:
        """The whole-set CRC32 the device would report for this content."""

        return pressure_fit_crc32(self.blob)


@dataclass(frozen=True)
class UsbParamAck:
    """Decoded v1.2.16 type 0x07: the answer to one 0x51 parameter command.

    ``subcommand``/``set_index`` echo the command that was sent.  That echo is
    the *only* correlation the protocol offers -- no nonce and no offset -- so
    only one command may be in flight at a time.

    ``valid_mask`` is "which of the six sets currently hold parameters".  It is
    unrelated to :attr:`UsbCdcFrame.valid_mask`, which is the IMU joint mask.
    """

    subcommand: int
    set_index: int
    status: int
    valid_mask: int
    write_index: int
    crc32: int

    @property
    def ok(self) -> bool:
        return self.status == PARAM_STATUS_OK

    @property
    def status_name(self) -> str:
        return PARAM_STATUS_NAMES.get(
            self.status, f"unknown status {self.status}")

    def set_valid(self, set_index: int) -> bool:
        """Whether the device says set ``set_index`` currently holds parameters.

        Every ack carries the current mask, not just a status query's (spec
        4.8), so this reads the device's view at the moment the command was
        answered.  A write's intermediate shard acks therefore answer for the
        *previous* contents -- the set only becomes valid on the last shard.
        """

        return bool(self.valid_mask & (1 << int(set_index)))


@dataclass(frozen=True)
class UsbRawParamAck:
    """Decoded compact type-0x07 acknowledgement for a raw parameter block.

    The current firmware does not echo a set number.  It acknowledges the
    complete byte block sent with ``0x51/0x01`` or the block returned by
    ``0x51/0x02``:

    ``subcommand(1) | status(1) | stored_length(2) | crc32(4)``.
    """

    subcommand: int
    status: int
    length: int
    crc32: int

    @property
    def ok(self) -> bool:
        return self.status == PARAM_STATUS_OK

    @property
    def status_name(self) -> str:
        return PARAM_STATUS_NAMES.get(
            self.status, f"unknown status {self.status}")


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
    """Incrementally recover framed messages from arbitrary serial chunks.

    ``link_kind`` names the link this parser is reading and is required, because
    type 0x03 is the one type whose *length gate* depends on it: a compressing
    link may carry the sparse and bitmap patches, a wired one may carry nothing
    but the fixed 384-byte picture (see :func:`carries_matrix_patches`).  The
    parser cannot work it out for itself -- both firmwares frame identically --
    so every construction site states it, and a missed one is a loud ``TypeError``
    in the reader thread rather than a silently frozen pressure panel.
    """

    def __init__(self, *, link_kind: str | None):
        self._link_kind = link_kind
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
                elif message_type in (USB_TYPE_PRESSURE_FIT_PARAMS,
                                      USB_TYPE_PARAM_ACK):
                    # For the 0x07 ack, same reasoning as the 0x05 branch
                    # above, and the same reason this cannot be left to the
                    # generic bound further down: a fixed length *is* the
                    # type's identity, while the generic branch only rejects
                    # payloads over USB_MAX_PAYLOAD (1100).  A 12+-1-byte ack
                    # would sail through it and be caught by the CRC at a
                    # byte-shifted offset, which files the failure under
                    # crc_errors instead of length_errors and misdirects anyone
                    # reading link_errors.  Sitting *before* the generic branch
                    # is what keeps that verdict honest.
                    if message_type == USB_TYPE_PRESSURE_FIT_PARAMS:
                        # 0x06 is the exception: the compact firmware returns
                        # one complete raw block whose size is defined by the
                        # frame header, not by the legacy 1032-byte per-set
                        # layout.  Keep accepting the legacy size too (it is
                        # included by this range).
                        fit_length_ok = 0 < payload_length <= 0xFFFF
                    else:
                        fit_length_ok = payload_length in PARAM_ACK_PAYLOAD_SIZES
                    if (header_size != USB_SLIM_HEADER_SIZE
                            or not fit_length_ok):
                        length_rejected += 1
                        continue
                elif message_type == USB_TYPE_ADC_MATRIX:
                    # The slim layout's accepted lengths are the link's own:
                    # three tiers -- a 384-byte full picture, a sparse index
                    # list, and a bitmap patch -- on a link whose firmware
                    # compresses, and only the 384-byte picture on a wired one.
                    # The length is the only thing that can tell them apart
                    # before the payload is complete, so the gate is the shapes'
                    # arithmetic rather than one pinned size, and the link kind
                    # is what says which set of shapes applies.  The legacy
                    # layout predates the patches and keeps its own single size.
                    if header_size == USB_SLIM_HEADER_SIZE:
                        length_ok = is_adc_matrix_payload_size(
                            payload_length, link_kind=self._link_kind)
                    else:
                        length_ok = payload_length == ADC_MATRIX_PAYLOAD_SIZE
                    if not length_ok:
                        length_rejected += 1
                        continue
                elif message_type == USB_TYPE_IMU_Q14:
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


def decode_pressure_fit_payload(payload: bytes) -> UsbPressureFitParams:
    """Decode a type-0x06 payload: 8 metadata bytes then k[256] and b[256].

    1032 is the only length accepted; a frame of any other length is a bug
    somewhere, not a variant to tolerate (see the parser's own branch).
    """

    if len(payload) != PRESSURE_FIT_PAYLOAD_SIZE:
        raise ValueError(
            f"pressure fit payload must be {PRESSURE_FIT_PAYLOAD_SIZE} bytes, "
            f"got {len(payload)}")
    set_index, flags, format_version, hand = payload[0:4]
    write_index = int.from_bytes(payload[4:8], "little")
    k_start = PRESSURE_FIT_META_SIZE
    b_start = k_start + PRESSURE_FIT_CHANNEL_COUNT * 2
    # ``astype`` rather than the raw ``frombuffer`` view: the view is read-only
    # and would pin the whole wire buffer for as long as this object lives.
    k = np.frombuffer(
        payload, dtype="<i2", offset=k_start,
        count=PRESSURE_FIT_CHANNEL_COUNT).astype(np.int16)
    b = np.frombuffer(
        payload, dtype="<i2", offset=b_start,
        count=PRESSURE_FIT_CHANNEL_COUNT).astype(np.int16)
    return UsbPressureFitParams(
        set_index=set_index,
        flags=flags,
        format_version=format_version,
        hand=hand,
        write_index=write_index,
        k=k,
        b=b,
    )


def decode_param_ack_payload(payload: bytes) -> UsbParamAck:
    """Decode a type-0x07 payload: the answer to one 0x51 command.

    An unrecognised ``status`` is carried through rather than rejected, so a
    newer firmware that adds one still reports something readable through
    :attr:`UsbParamAck.status_name`.
    """

    if len(payload) != PARAM_ACK_PAYLOAD_SIZE:
        raise ValueError(
            f"parameter ack payload must be {PARAM_ACK_PAYLOAD_SIZE} bytes, "
            f"got {len(payload)}")
    return UsbParamAck(
        subcommand=payload[0],
        set_index=payload[1],
        status=payload[2],
        valid_mask=payload[3],
        write_index=int.from_bytes(payload[4:8], "little"),
        crc32=int.from_bytes(payload[8:12], "little"),
    )


def decode_raw_param_ack_payload(payload: bytes) -> UsbRawParamAck:
    """Decode the current compact type-0x07 raw-block acknowledgement."""

    if len(payload) != PARAM_RAW_ACK_PAYLOAD_SIZE:
        raise ValueError(
            f"raw parameter ack payload must be "
            f"{PARAM_RAW_ACK_PAYLOAD_SIZE} bytes, got {len(payload)}")
    return UsbRawParamAck(
        subcommand=payload[0],
        status=payload[1],
        length=int.from_bytes(payload[2:4], "little"),
        crc32=int.from_bytes(payload[4:8], "little"),
    )


def _pressure_fit_words(values, name: str) -> np.ndarray:
    """Coerce one 256-channel k or b into little-endian int16 wire words.

    Integer input is taken as the raw Q12 wire truth and only range-checked;
    float input is quantised through :func:`pressure_fit_quantize`.  Either way
    a value the wire cannot hold raises instead of wrapping.
    """

    array = np.asarray(values)
    if array.size != PRESSURE_FIT_CHANNEL_COUNT:
        raise ValueError(
            f"{name} must hold exactly {PRESSURE_FIT_CHANNEL_COUNT} channels, "
            f"got {array.size}")
    array = array.reshape(PRESSURE_FIT_CHANNEL_COUNT)
    if np.issubdtype(array.dtype, np.integer):
        if array.size and (int(array.min()) < PRESSURE_FIT_Q12_MIN
                           or int(array.max()) > PRESSURE_FIT_Q12_MAX):
            raise ValueError(f"{name} does not fit signed 16-bit")
        return array.astype("<i2")
    return pressure_fit_quantize(array, name=name).astype("<i2")


def encode_pressure_fit_blob(k, b) -> bytes:
    """Pack k[256] and b[256] into the 1024-byte blob the device stores."""

    return (_pressure_fit_words(k, "k").tobytes()
            + _pressure_fit_words(b, "b").tobytes())


SYNTHETIC_PRESSURE_FIT_SEED = 20260922


def synthetic_pressure_fit_blob(
        seed: int = SYNTHETIC_PRESSURE_FIT_SEED) -> bytes:
    """A deterministic 1024-byte test set, for exercising the write path.

    Every stored set on a real glove holds real calibration, so anything that
    tests a write has to supply its own content.  Built from raw Q12 words
    rather than floats: integer input takes the range-checked branch of
    :func:`_pressure_fit_words` and so cannot be defeated by float rounding at
    the edge of the Q12 range, which the float path would raise on rather than
    wrap.

    Values stay 7/8 of full scale away from both rails, so a truncated, shifted
    or byte-swapped read is loud rather than subtle, and ``k`` never holds a
    zero because the device's own conversion (spec 4.7) divides by it.
    """

    limit = 28672  # 7.0 in Q12, against a full scale of 7.9998
    rng = np.random.default_rng(int(seed))
    k = rng.integers(-limit, limit + 1, PRESSURE_FIT_CHANNEL_COUNT,
                     dtype=np.int64)
    k[k == 0] = limit
    b = rng.integers(-limit, limit + 1, PRESSURE_FIT_CHANNEL_COUNT,
                     dtype=np.int64)
    return encode_pressure_fit_blob(k, b)


def _q12_word_summary(words) -> dict:
    """Min/max/nonzero/median/rail-count of one 256-word k or b array.

    Raw Q12 words, and deliberately *not* through ``abs()``: on int16
    ``np.abs`` wraps at -32768, which is one of the two rails this exists to
    count.
    """

    words = np.asarray(words).reshape(-1)
    info = np.iinfo(np.int16)
    return {
        "min": int(words.min()),
        "max": int(words.max()),
        "nonzero": int(np.count_nonzero(words)),
        "median": int(np.median(words)),
        "rails": int(np.count_nonzero(
            (words == info.min) | (words == info.max))),
    }


def pressure_fit_set_summary(params: "UsbPressureFitParams") -> dict:
    """One set's identity plus its k/b statistics, as primitive values only.

    Plain numbers and strings on purpose: this is what the live viewer's solver
    process ships to the GUI through a ``multiprocessing.Queue``, and that
    boundary carries dicts rather than decoded snapshot objects.
    """

    return {
        "set_index": int(params.set_index),
        "valid": bool(params.valid),
        "flags": int(params.flags),
        "format_version": int(params.format_version),
        "supported_format": bool(params.supported_format),
        "hand": int(params.hand),
        "hand_side": params.hand_side,
        "write_index": int(params.write_index),
        "crc32": int(params.crc32),
        "k": _q12_word_summary(params.k),
        "b": _q12_word_summary(params.b),
    }


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


@dataclass(frozen=True)
class PressureFitSnapshot:
    """The newest cached type-0x06 frame for one set, with its frame metadata."""

    set_index: int
    sequence: int
    device_timestamp_us: int
    host_timestamp_us: int
    params: UsbPressureFitParams

    @property
    def crc32(self) -> int:
        return self.params.crc32


class PressureFitCache:
    """Latest type-0x06 parameters, one entry per set index.

    A 0x06 frame is a *complete* block -- every frame carries all 256 channels
    of one set -- so unlike :class:`MagTelemetryCache` there is no per-channel
    freshness to merge: a new frame simply replaces that set's entry.  No
    ``level_seen``-style bookkeeping is needed, and none is offered.
    """

    def __init__(self) -> None:
        # Own lock, for the same reason MagTelemetryCache has one: the producer
        # is the USB reader thread and callers come from anywhere.
        self._lock = threading.RLock()
        self._by_set: dict[int, PressureFitSnapshot] = {}

    def reset(self) -> None:
        """Drop everything; called when the link is (re)opened or lost."""

        with self._lock:
            self._by_set.clear()

    def snapshot(self, set_index: int) -> PressureFitSnapshot | None:
        with self._lock:
            return self._by_set.get(int(set_index))

    def forget(self, set_index: int) -> None:
        """Drop one set's entry after the device's copy of it changed.

        A write emits no 0x06 (spec 4.7), so a cached entry survives a write
        that replaced exactly the bytes it describes -- old ``write_index``,
        old CRC32, old coefficients.  Dropping the entry leaves "not read this
        session" visible instead of a value that is quietly wrong, and it is
        the same reasoning :meth:`reset` applies to a whole-device erase.
        """

        with self._lock:
            self._by_set.pop(int(set_index), None)

    def snapshots(self) -> dict[int, PressureFitSnapshot]:
        """A shallow copy, safe to iterate while the reader thread updates."""

        with self._lock:
            return dict(self._by_set)

    def valid_mask(self) -> int:
        """Bits of the sets whose cached frame says they hold valid data."""

        with self._lock:
            mask = 0
            for set_index, snapshot in self._by_set.items():
                if snapshot.params.valid:
                    mask |= 1 << int(set_index)
            return mask

    def update(self, frame, host_timestamp_us: int) -> PressureFitSnapshot:
        """Decode one 0x06 frame and replace that set's cached entry.

        A ``ValueError`` from the decoder propagates to the reader thread's own
        handler; the cache is left untouched.
        """

        params = decode_pressure_fit_payload(frame.payload)
        snapshot = PressureFitSnapshot(
            set_index=int(params.set_index),
            sequence=int(frame.sequence),
            device_timestamp_us=int(frame.timestamp_us),
            host_timestamp_us=int(host_timestamp_us),
            params=params,
        )
        with self._lock:
            self._by_set[snapshot.set_index] = snapshot
        return snapshot


def _unpack_12bit_values(
        data: bytes | bytearray | memoryview, count: int) -> np.ndarray:
    """Unpack ``count`` consecutive 12-bit samples packed two per three bytes.

    Little-endian bit order, the same in every tier that packs at all::

        byte0 = sample0[ 7: 0]
        byte1 = sample0[11: 8] | (sample1[ 3: 0] << 4)
        byte2 = sample1[11: 4]

    An odd ``count`` leaves the last sample with only the two bytes its own 12
    bits need, which is the same truncation a reader of the pair layout gets by
    stopping early rather than a second encoding.
    """

    values = np.empty(int(count), dtype=np.uint32)
    pairs = int(count) // 2
    if pairs:
        block = np.frombuffer(
            bytes(data[:pairs * 3]), dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        values[0:2 * pairs:2] = block[:, 0] | ((block[:, 1] & 0x0F) << 8)
        values[1:2 * pairs:2] = (block[:, 1] >> 4) | (block[:, 2] << 4)
    if int(count) % 2:
        tail = int(pairs) * 3
        values[-1] = int(data[tail]) | ((int(data[tail + 1]) & 0x0F) << 8)
    return values


def _unpack_12bit_matrix(payload: bytes) -> np.ndarray:
    """Unpack a 384-byte 12-bit-packed block into a 16x16 float array."""

    samples = _unpack_12bit_values(payload, ADC_MATRIX_CELLS)
    return samples.reshape(ADC_MATRIX_ROWS, ADC_MATRIX_COLS).astype(np.float32)


def decode_adc_matrix_patch(
        payload: bytes | bytearray | memoryview,
        *,
        link_kind: str | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split a sparse or bitmap tier into ``(flat indices, sample values)``.

    Returns ``None`` when the payload is neither, or when a sparse index falls
    outside the 256 cells -- an index the contract cannot produce, so the frame
    is dropped rather than allowed to write off the end of a picture.

    A bitmap's values are emitted in ascending bit order, so both tiers come
    back in one shape and the merge in :class:`AdcMatrixAssembler` stays a
    single write.

    ``link_kind`` reaches :func:`adc_matrix_payload_tier`, so a patch payload on
    a link whose firmware sends pictures only is ``None`` here too, for the same
    reason as there.
    """

    tier = adc_matrix_payload_tier(payload, link_kind=link_kind)
    if tier == ADC_MATRIX_TIER_SPARSE:
        count = int(payload[0])
        body = np.frombuffer(
            bytes(payload[1:1 + 3 * count]), dtype=np.uint8).reshape(count, 3)
        indices = np.array(body[:, 0], dtype=np.intp)
        if indices.size and int(indices.max()) >= ADC_MATRIX_CELLS:
            return None
        values = (body[:, 1].astype(np.uint32)
                  | (body[:, 2].astype(np.uint32) << 8))
        return indices, values.astype(np.float32)

    if tier == ADC_MATRIX_TIER_BITMAP:
        bitmap = bytes(payload[1:1 + ADC_MATRIX_BITMAP_SIZE])
        indices = np.array(
            [index * 8 + bit
             for index, byte in enumerate(bitmap)
             for bit in range(8)
             if byte & (1 << bit)],
            dtype=np.intp)
        values = _unpack_12bit_values(
            payload[1 + ADC_MATRIX_BITMAP_SIZE:], indices.size)
        return indices, values.astype(np.float32)

    return None


class AdcMatrixAssembler:
    """Rebuild the 16x16 pressure picture from the three-tier 0x03 stream.

    On a compressing link the device no longer sends a picture every scan.  It
    sends a full frame when the stream starts and at least every 30 scans after
    that, and between them only the cells that changed -- so a consumer that
    wants a picture has to keep the last one and apply the patches to it.

    That is all this class is: :meth:`update` takes one payload and returns
    either a complete picture (:class:`UsbAdcMatrixFrame`) or ``None``, and the
    picture it returns for a patch is the merged result, not the patch.  A patch
    that arrives before the first full frame is dropped rather than merged onto
    zeros, since a picture invented from nothing would be indistinguishable
    downstream from one the device actually reported; the contract's opening
    keyframe and its 30-frame refresh bound how long that can matter, and
    :attr:`baseline_waits` counts the wait.

    ``link_kind`` picks the dialect (:func:`carries_matrix_patches`) and is
    required rather than defaulted: on a wired link there is nothing to assemble
    -- every payload is the whole picture -- so the class degrades to a decoder
    that refuses patches, and a caller that forgot the link would otherwise
    silently get the Bluetooth reading of a wired stream.

    One instance per reader loop, and :meth:`reset` alongside the frame parser
    whenever the port is reopened: a picture carried across a reconnect belongs
    to a link that no longer exists.
    """

    def __init__(self, *, link_kind: str | None) -> None:
        self._link_kind = link_kind
        self._picture: np.ndarray | None = None
        self._lock = threading.Lock()
        # Both cumulative like the parser's own counters, and both diagnostics
        # rather than errors: an orphan is *expected* for the first scans after
        # a link opens mid-stream, and an untiered payload cannot reach here
        # from the wire at all -- the parser's gate has already refused any
        # length that is no shape.  What a rising count means is that the host
        # and the device disagree about the contract, which is worth a number
        # rather than an exception in a reader thread.
        #
        # ``baseline_waits`` is published as
        # ``link_errors["matrix_baseline_waits"]``: it is the only way to tell a
        # frozen pressure panel from a quiet hand.
        self.baseline_waits = 0
        self.rejected = 0

    def reset(self) -> None:
        """Drop the picture.  Call on every port (re)open."""

        with self._lock:
            self._picture = None

    @property
    def has_baseline(self) -> bool:
        """True once a full frame has seeded the picture."""

        with self._lock:
            return self._picture is not None

    def update(
            self,
            payload: bytes | bytearray | memoryview,
            *,
            sequence: int = 0,
            scan_time_us: int = 0,
    ) -> UsbAdcMatrixFrame | None:
        """Fold one 0x03 payload in; return a picture, or ``None`` when there is
        no picture to hand back yet.

        ``None`` means a patch arrived before the keyframe it is a delta
        against, or a payload that fits no tier on this link -- never an empty
        patch, which on a compressing link is a scan in which nothing moved and
        is published like any other.
        """

        tier = adc_matrix_payload_tier(payload, link_kind=self._link_kind)
        if is_adc_matrix_keyframe(tier):
            try:
                frame = decode_adc_matrix_payload(
                    payload, sequence=sequence, scan_time_us=scan_time_us)
            except ValueError:
                self.rejected += 1
                return None
            with self._lock:
                self._picture = frame.samples
            # The *copy* is what leaves.  The array kept here is what the next
            # patch merges onto, and a consumer that wrote into the one it was
            # handed would be editing the host's own state -- which the old
            # one-picture-per-scan decoder could never expose.  The panel does
            # exactly that, drawing straight from the picture it was given
            # while the next scan lands.
            return UsbAdcMatrixFrame(
                sequence=frame.sequence,
                scan_time_us=frame.scan_time_us,
                samples=frame.samples.copy(),
            )

        patch = (None if tier is None
                 else decode_adc_matrix_patch(payload, link_kind=self._link_kind))
        if patch is None:
            self.rejected += 1
            return None

        indices, values = patch
        with self._lock:
            if self._picture is None:
                self.baseline_waits += 1
                return None
            merged = self._picture.copy()
            merged.reshape(-1)[indices] = values
            self._picture = merged
        return UsbAdcMatrixFrame(
            sequence=int(sequence),
            scan_time_us=int(scan_time_us),
            samples=merged.copy(),
        )


def decode_adc_matrix_payload(
        payload: bytes,
        *,
        sequence: int = 0,
        scan_time_us: int = 0,
) -> UsbAdcMatrixFrame:
    """Decode a *keyframe* pressure-matrix payload into a 16x16 sample array.

    v1.2.10+ sends a bare 384-byte 12-bit-packed block with no metadata, so the
    frame header's sequence/timestamp are passed in by the caller.  Earlier
    firmware sends a 524-byte payload with a 12-byte metadata prefix that carries
    those values itself.

    This is only the self-contained tier.  A sparse or bitmap payload is not a
    smaller version of a picture and has nothing to decode on its own -- it
    raises here, and callers fed by a live link want
    :meth:`AdcMatrixAssembler.feed` instead, which merges those onto the last
    picture rather than refusing them.
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
            f"or {ADC_MATRIX_PAYLOAD_SIZE} bytes, got {len(payload)}"
            f" (tier {adc_matrix_payload_tier(payload)!r})")
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


def encode_param_command(
        subcommand: int,
        *,
        set_index: int = 0,
        offset: int = 0,
        length: int = 0,
        crc32: int = 0,
        data: bytes = b"",
) -> bytes:
    """Encode one v1.2.16 host->device 0x51 parameter command.

    The frame is ``A5 5A 51 subcmd(1) set(1) offset(2) length(2) crc32(4)`` --
    13 bytes plus the data area, with no version and no CRC16 shell, matching
    the command family that 0x50 already belongs to (spec 5.1/5.2).  The
    first four bytes are therefore always ``A5 5A 51 subcmd`` and the next is
    the set number; callers pairing an answer back to a command rely on that.

    ``crc32`` is the whole 1024-byte set's CRC-32 (see
    :func:`pressure_fit_crc32`), not a per-shard value, and every shard of a
    write carries the same one.  Commands that carry no data pass 0, which is
    a legitimate field value rather than "unset".

    The checks here mirror the statuses the device would otherwise return
    (1 = range, 3 = bad set), so an obviously malformed command never reaches
    the wire.
    """

    subcommand = int(subcommand) & 0xFF
    if subcommand not in PARAM_SUBCMDS:
        raise ValueError(f"unknown parameter subcommand {subcommand:#x}")
    set_index = int(set_index)
    if not 0 <= set_index < PRESSURE_FIT_SET_COUNT:
        raise ValueError(
            f"set_index must be 0..{PRESSURE_FIT_SET_COUNT - 1}, "
            f"got {set_index}")
    offset, length, crc32 = int(offset), int(length), int(crc32)
    if length != len(data):
        raise ValueError(
            f"length {length} does not match {len(data)} data bytes")
    if length and subcommand != PARAM_SUBCMD_WRITE_SHARD:
        raise ValueError("only a write shard carries a data area")
    if length > PRESSURE_FIT_SHARD_SIZE:
        raise ValueError(
            f"a shard carries at most {PRESSURE_FIT_SHARD_SIZE} bytes, "
            f"got {length}")
    if offset + length > PRESSURE_FIT_BLOB_SIZE:
        raise ValueError(
            f"offset {offset} + length {length} exceeds the "
            f"{PRESSURE_FIT_BLOB_SIZE}-byte set")

    return (
        USB_MAGIC
        + bytes([USB_TYPE_PARAM_COMMAND])
        + struct.pack("<BBHH", subcommand, set_index, offset, length)
        + struct.pack("<I", crc32 & 0xFFFFFFFF)
        + bytes(data)
    )


def encode_raw_param_command(
        subcommand: int, data: bytes = b"", *, crc32: int = 0) -> bytes:
    """Encode the current whole-block ``0x51`` command.

    Wire layout, exactly as used by the current firmware::

        A5 5A 51 subcommand length(le16) crc32(le32) data...

    A read is ``subcommand=0x02`` with zero length, zero CRC and no data.  A
    write is ``subcommand=0x01`` and carries the complete raw block in one
    command.  The older set/offset/shard encoder remains available as
    :func:`encode_param_command` for compatibility with the previous firmware.
    """

    subcommand = int(subcommand) & 0xFF
    if subcommand not in (PARAM_SUBCMD_WRITE_SHARD, PARAM_SUBCMD_READ_SET):
        raise ValueError(
            f"raw parameter command only supports subcommands 0x01/0x02, "
            f"got {subcommand:#x}")
    data = bytes(data)
    if subcommand == PARAM_SUBCMD_READ_SET and data:
        raise ValueError("raw parameter read must not carry data")
    if len(data) > PARAM_RAW_BLOCK_MAX_SIZE:
        raise ValueError(
            f"raw parameter block is larger than the firmware limit "
            f"({PARAM_RAW_BLOCK_MAX_SIZE} bytes)")
    if subcommand == PARAM_SUBCMD_READ_SET:
        crc32 = 0
    return (
        USB_MAGIC
        + bytes([USB_TYPE_PARAM_COMMAND, subcommand])
        + struct.pack("<H", len(data))
        + struct.pack("<I", int(crc32) & 0xFFFFFFFF)
        + data
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


def mag_ready_mask(
        levels: Iterable[int | None],
        channel_to_hand: Iterable[int] = DEFAULT_CHANNEL_TO_HAND,
) -> tuple[bool, ...]:
    """Collapse 0x05 MAG levels into per-MANO-joint "initialized" flags.

    ``levels`` is :attr:`~common.types.MagStatus.levels` -- the MAG field of each
    channel's CALIB_STAT byte, 0..3, or ``None`` for a channel that has never
    reported -- in the **physical** order 0x05 carries.  The result is in MANO
    joint order (wrist, then three IMUs per finger), the same order
    :func:`remap_physical_to_hand` produces and :attr:`~common.types.HandFrame.
    imu_xyzw` uses; 0x05 is not hand-remapped on the wire, so the reorder
    happens here.

    A channel counts as initialized at level 3 and nothing else -- level 0, 1,
    2 and a never-reported ``None`` all read ``False``.  There is deliberately no
    third state per channel: a caller asking "is this IMU initialized" wants a
    yes or no, and the distinction between "reported as not yet calibrated" and
    "has not reported yet" lives on :class:`~common.types.MagStatus`
    (``seen`` / ``fresh`` / ``unread_channels``) for callers that need it.
    """

    mapping = channel_to_hand_for_firmware(channel_to_hand)
    values = tuple(levels)
    if len(values) != TOTAL_IMU_COUNT:
        raise ValueError(
            f"mag levels must have {TOTAL_IMU_COUNT} channels, got {len(values)}")

    ready = [False] * TOTAL_IMU_COUNT
    for physical_channel, mano_joint in enumerate(mapping):
        value = values[physical_channel]
        if value is None:
            continue
        # 3 is MAG_READY_LEVEL in sdk.mag.  Hard-coded here for the same reason
        # the MagStatus count properties hard-code it: this module is protocol
        # level and must not import the SDK layer.
        ready[mano_joint] = ((int(value) >> MAG_LEVEL_SHIFT) & MAG_LEVEL_MASK) == 3
    return tuple(ready)


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
