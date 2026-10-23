"""Pressure-fit parameters: the host<->device parameter channel (firmware >= 1.2.16).

A pressure-fit set is what turns a glove's raw pressure ADC into newtons.  One
set is 512 int16 coefficients covering all 256 channels of the 16x16 matrix --
256 ``k`` slopes followed by 256 ``b`` intercepts -- held in Q12 fixed point:

    physical = raw / 4096        range [-8.0, +7.9998]

The device stores up to :data:`PRESSURE_FIT_SET_COUNT` (6) of them.  A set is
written as four 256-byte shards, each carrying the CRC-32 of the *whole* 1024
byte blob, and the device commits the set only when the final shard's ack marks
it valid.

This module is a facade over :mod:`sdk.io.streams` and
:mod:`sdk.io.usb_protocol`, in the same shape as :mod:`sdk._streams`: it
re-exports and writes no logic of its own.  Reach the channel through a running
glove rather than by opening a port here --

    from sdk import open_glove

    with open_glove("right") as glove:
        if glove.supports_pressure_fit:
            glove.wait_for_firmware_version()      # 0x01 has to land first
            outcome = glove.query_pressure_fit()   # 0x04: ask the device to report
            for set_index in sorted(outcome.params or {}):
                params = outcome.params[set_index]
                read = glove.read_pressure_fit(set_index)
                glove.write_pressure_fit(set_index, read.params.blob)

:meth:`~sdk.glove.Glove.start_pressure_fit` is the odd one out: it asks the
device to fit for itself (subcommand ``0x06``, a contract with the firmware
team rather than a spec entry) and reads *nothing* back -- no parameter set
comes out of it, and any 0x06 frames it happens to see belong to whatever query
asks next.  A firmware that has not implemented it says nothing at all, which
is why the failure names the subcommand instead of reporting a generic timeout.

:meth:`~sdk.glove.Glove.write_pressure_fit` and its siblings block until the
device answers (a write is four shards plus retries, so seconds are normal).
Do not call them from a render thread; the IMU reader is unaffected, since the
channel takes its own lock inside the transport.

The coefficient values themselves come from a bench calibration file, and
turning such a file into a blob is the caller's job -- the SDK moves bytes, it
does not fit curves.
"""

from sdk.io.streams import (
    PARAM_ACK_TIMEOUT_S,
    PARAM_COMMAND_ATTEMPTS,
    PARAM_SHARD_RETRIES,
    PARAM_VERSION_WAIT_S,
    PressureFitCommandOutcome,
    PressureFitReadOutcome,
    PressureFitSnapshot,
    PressureFitWriteOutcome,
    RawParamReadOutcome,
    RawParamWriteOutcome,
    write_pressure_fit_shards,
)
from sdk.io.usb_protocol import (
    PARAM_ACK_PAYLOAD_SIZE,
    PARAM_STATUS_BAD_SET,
    PARAM_STATUS_CRC_MISMATCH,
    PARAM_STATUS_FLASH_FAIL,
    PARAM_STATUS_NAMES,
    PARAM_STATUS_OK,
    PARAM_STATUS_RANGE,
    PARAM_SUBCMD_CLEAR_ALL,
    PARAM_SUBCMD_QUERY_STATUS,
    PARAM_SUBCMD_READ_SET,
    PARAM_SUBCMD_START_FIT,
    PARAM_SUBCMD_WRITE_SHARD,
    PARAM_SUBCMDS,
    PRESSURE_FIT_BLOB_SIZE,
    PRESSURE_FIT_CHANNEL_COUNT,
    PRESSURE_FIT_FLAG_VALID,
    PRESSURE_FIT_FORMAT_VERSION,
    PRESSURE_FIT_META_SIZE,
    PRESSURE_FIT_PAYLOAD_SIZE,
    PRESSURE_FIT_Q12_MAX,
    PRESSURE_FIT_Q12_MIN,
    PRESSURE_FIT_Q12_SCALE,
    PRESSURE_FIT_SET_COUNT,
    PRESSURE_FIT_SHARD_COUNT,
    PRESSURE_FIT_SHARD_SIZE,
    SYNTHETIC_PRESSURE_FIT_SEED,
    USB_PARAM_COMMAND_SIZE,
    USB_PRESSURE_FIT_PARAMS_FROM,
    USB_TYPE_PARAM_ACK,
    USB_TYPE_PARAM_COMMAND,
    USB_TYPE_PRESSURE_FIT_PARAMS,
    UsbParamAck,
    UsbPressureFitParams,
    decode_param_ack_payload,
    decode_pressure_fit_payload,
    encode_param_command,
    encode_pressure_fit_blob,
    hand_side_from_code,
    pressure_fit_crc32,
    pressure_fit_quantize,
    pressure_fit_set_summary,
    supports_pressure_fit_params,
    synthetic_pressure_fit_blob,
)

__all__ = [
    # Wire constants
    "PRESSURE_FIT_SET_COUNT", "PRESSURE_FIT_CHANNEL_COUNT",
    "PRESSURE_FIT_META_SIZE", "PRESSURE_FIT_BLOB_SIZE",
    "PRESSURE_FIT_PAYLOAD_SIZE", "PRESSURE_FIT_FORMAT_VERSION",
    "PRESSURE_FIT_FLAG_VALID", "PRESSURE_FIT_SHARD_SIZE",
    "PRESSURE_FIT_SHARD_COUNT", "PRESSURE_FIT_Q12_SCALE",
    "PRESSURE_FIT_Q12_MIN", "PRESSURE_FIT_Q12_MAX",
    "USB_TYPE_PRESSURE_FIT_PARAMS", "USB_TYPE_PARAM_ACK",
    "USB_TYPE_PARAM_COMMAND", "USB_PARAM_COMMAND_SIZE",
    "USB_PRESSURE_FIT_PARAMS_FROM",
    "PARAM_ACK_PAYLOAD_SIZE", "PARAM_SUBCMD_WRITE_SHARD",
    "PARAM_SUBCMD_READ_SET", "PARAM_SUBCMD_CLEAR_ALL",
    "PARAM_SUBCMD_QUERY_STATUS", "PARAM_SUBCMD_START_FIT", "PARAM_SUBCMDS",
    "PARAM_STATUS_OK", "PARAM_STATUS_RANGE", "PARAM_STATUS_FLASH_FAIL",
    "PARAM_STATUS_BAD_SET", "PARAM_STATUS_CRC_MISMATCH", "PARAM_STATUS_NAMES",
    # Timing the public entry points default to
    "PARAM_ACK_TIMEOUT_S", "PARAM_SHARD_RETRIES", "PARAM_VERSION_WAIT_S",
    "PARAM_COMMAND_ATTEMPTS",
    # Decoded values
    "UsbPressureFitParams", "UsbParamAck", "PressureFitSnapshot",
    "PressureFitReadOutcome", "PressureFitCommandOutcome",
    "PressureFitWriteOutcome",
    "RawParamReadOutcome", "RawParamWriteOutcome",
    # Codecs and helpers
    "encode_param_command", "encode_pressure_fit_blob",
    "decode_pressure_fit_payload", "decode_param_ack_payload",
    "pressure_fit_crc32", "pressure_fit_quantize",
    "pressure_fit_set_summary", "supports_pressure_fit_params",
    "hand_side_from_code", "write_pressure_fit_shards",
    "SYNTHETIC_PRESSURE_FIT_SEED", "synthetic_pressure_fit_blob",
]
