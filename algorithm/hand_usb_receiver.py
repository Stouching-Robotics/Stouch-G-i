"""USB CDC receiver shared by the Friston wizard and legacy hand adapters."""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from algorithm.friston_rotations import correct_imu_quaternion_axes
from algorithm.hand.safety import DeviceTimestampClock, QuaternionStreamFilter
from common.datamodels import ImuMsg
from common.ring_buffer import RingBuffer
from common.usb_cdc import (
    AdcMatrixAssembler, BLUETOOTH_DONGLE_PID, LINK_KIND_BLUETOOTH, LINK_KIND_USB,
    USB_TYPE_ADC_MATRIX, USB_TYPE_IMU_Q14, USB_TYPE_STATUS, UsbCdcFrameParser,
    decode_imu_q14_payload, find_stm32_cdc_port, is_supported_version,
    readable_imu_mask, remap_physical_to_hand,
)

EXPECTED_IMU_IDS = [f"imu_{i}" for i in range(16)]


_SERIAL_READ_LIMIT = 4096


def _read_serial_available(serial_handle) -> bytes:
    """Read whatever bytes the CDC driver already has buffered.

    ``serial_handle.read(4096)`` waits for a nearly full 4 KiB block (or the
    200 ms timeout) on Windows.  IMU and tactile packets share this CDC
    stream, so that batches roughly 6-8 IMU samples into each host read and
    collapses the 80 Hz device stream to ~10-16 Hz.  Block for one byte when
    idle, then immediately drain the driver's current buffer; the frame parser
    accepts partial packets, so throughput is preserved without the batching
    delay.
    """
    available = int(serial_handle.in_waiting)
    read_size = min(max(available, 1), _SERIAL_READ_LIMIT)
    return serial_handle.read(read_size)


def hand_usb_receiver(
        kill_switch: threading.Event,
        serial_port: str | None,
        usb_vid: int,
        usb_pid: int,
        channel_to_hand: list[int],
        buffer: RingBuffer,
        status_callback: Callable[[list[str]], None] | None = None,
        pressure_buffer: RingBuffer | None = None,
        firmware_callback: Callable[[str], None] | None = None,
        diagnostics_state: dict | None = None,
        *, on_stream_reset: Callable[[], None] | None = None,
):
    """Receive STM32 Q14 IMU frames and publish normalized ``ImuMsg`` data.

    ``pressure_buffer`` follows the callback-free pattern: the receiver is handed
    a shared object to fill rather than a callback, so a consumer that only wants
    the data never has to run inside the receive loop.

    Type-0x05 magnetometer telemetry is not consumed here.  Calibration reads IMU,
    tactile and status frames only; the live viewer has its own receiver
    (``glove_io.streams``) and that is the one that caches 0x05.
    """

    try:
        import serial
    except ImportError:
        logger.error("STM32 USB CDC requires pyserial; run: pip install pyserial")
        if status_callback:
            status_callback(EXPECTED_IMU_IDS)
        return

    n_frame = 0
    diag_t = time.time()
    diag_n_frame = 0
    stream_filter = QuaternionStreamFilter()
    device_clock = DeviceTimestampClock()
    # The pressure dialect is the link's: the Bluetooth firmware compresses the
    # matrix into sparse/bitmap patches, the wired one sends a fixed 384-byte
    # picture every scan.  Same pid test ``glove_io.streams`` makes, on the ids
    # the calibration carries for this hand (the live-link rebind above is what
    # keeps those pointing at the link the hand is actually on).
    link_kind = (LINK_KIND_BLUETOOTH if usb_pid == BLUETOOTH_DONGLE_PID
                 else LINK_KIND_USB)
    parser = UsbCdcFrameParser(link_kind=link_kind)
    matrix_assembler = AdcMatrixAssembler(link_kind=link_kind)
    absent_streak = np.zeros(16, dtype=np.int32)
    rejected_streak = np.zeros(16, dtype=np.int32)
    calibration_status = np.full(16, 0xFF, dtype=np.uint8)
    last_missing_ids: list[str] | None = None
    last_payload_error: str | None = None
    reported_status: str | None = None
    reported_firmware_version: str | None = None
    pressure_frame_count = 0
    diag_pressure_count = 0
    serial_handle = None
    active_port: str | None = None

    while not kill_switch.is_set():
        if serial_handle is None:
            try:
                active_port = serial_port or find_stm32_cdc_port(usb_vid, usb_pid)
            except Exception as exc:
                logger.warning(f"[IMU] Serial-port discovery failed: {exc}")
                active_port = None

            if not active_port:
                if status_callback and last_missing_ids != EXPECTED_IMU_IDS:
                    status_callback(EXPECTED_IMU_IDS)
                    last_missing_ids = EXPECTED_IMU_IDS.copy()
                logger.warning(
                    f"[IMU] STM32 CDC not found (VID={usb_vid:04X}, PID={usb_pid:04X})")
                kill_switch.wait(1.0)
                continue

            try:
                serial_handle = serial.Serial(
                    active_port, baudrate=115200, timeout=0.2)
                parser.reset()
                if on_stream_reset is not None:
                    stream_filter = QuaternionStreamFilter()
                    device_clock = DeviceTimestampClock()
                    absent_streak.fill(0)
                    rejected_streak.fill(0)
                    last_missing_ids = None
                    reported_firmware_version = None
                    on_stream_reset()
                logger.info(f"[IMU] STM32 USB CDC connected: {active_port}")
            except (serial.SerialException, OSError) as exc:
                logger.warning(f"[IMU] Could not open serial port ({active_port}): {exc}")
                serial_handle = None
                kill_switch.wait(1.0)
                continue

        try:
            raw = _read_serial_available(serial_handle)
        except (serial.SerialException, OSError) as exc:
            logger.warning(f"[IMU] STM32 USB CDC disconnected ({active_port}): {exc}")
            try:
                serial_handle.close()
            except Exception:
                pass
            serial_handle = None
            if on_stream_reset is not None:
                on_stream_reset()
            if status_callback:
                status_callback(EXPECTED_IMU_IDS)
            last_missing_ids = EXPECTED_IMU_IDS.copy()
            kill_switch.wait(0.5)
            continue

        if not raw:
            continue

        for usb_frame in parser.feed(raw):
            # The firmware version is present in the header of every valid
            # frame.  Publish it as soon as the first frame arrives so GUI
            # clients can show the actual hardware version.
            if (firmware_callback is not None
                    and usb_frame.version != reported_firmware_version):
                firmware_callback(usb_frame.version)
                reported_firmware_version = usb_frame.version

            if not is_supported_version(usb_frame.version):
                logger.error(
                    f"[IMU] Unsupported STM32 USB protocol version: {usb_frame.version}")
                continue

            if usb_frame.message_type == USB_TYPE_STATUS:
                status_text = usb_frame.payload.decode("utf-8", errors="replace").strip()
                if status_text and status_text != reported_status:
                    logger.info(f"[STM32] Startup status report:\n{status_text}")
                    reported_status = status_text
                continue
            if usb_frame.message_type == USB_TYPE_ADC_MATRIX:
                # Full frames and the sparse/bitmap patches between them come
                # out of the assembler as complete pictures, so the pressure
                # ring buffer never holds a picture with only the changed cells
                # written into it.  ``None`` is a patch with no keyframe behind
                # it yet, which is normal for the first scans of a link that
                # opened mid-stream and is not worth an error line.
                matrix_frame = matrix_assembler.update(
                    usb_frame.payload,
                    sequence=usb_frame.sequence,
                    scan_time_us=usb_frame.timestamp_us,
                )
                if matrix_frame is not None and pressure_buffer is not None:
                    pressure_buffer.push(matrix_frame.samples)
                    pressure_frame_count += 1
                continue
            if usb_frame.message_type != USB_TYPE_IMU_Q14:
                continue

            try:
                physical_quaternions = decode_imu_q14_payload(usb_frame.payload)
                # The flags-gated valid mask is deliberately ignored below in
                # favour of ``readable`` so a recovered IMU is accepted again
                # even while the firmware flags bit is still clear.
                quaternion_frame, _ = remap_physical_to_hand(
                    physical_quaternions,
                    usb_frame.valid_mask,
                    channel_to_hand,
                )
                # "Readable" is purely data-driven (quaternion is valid), so a
                # previously-broken IMU keeps being re-evaluated every frame and
                # is un-flagged the moment its data becomes readable again.
                readable = readable_imu_mask(physical_quaternions, channel_to_hand)
            except ValueError as exc:
                error_text = str(exc)
                if error_text != last_payload_error:
                    logger.error(f"[IMU] Incompatible STM32 payload: {error_text}")
                    last_payload_error = error_text
                continue

            # Keep the exact axis conversion used by the 2.1 WebSocket path.
            for joint_id in range(16):
                quaternion_frame[joint_id] = correct_imu_quaternion_axes(
                    quaternion_frame[joint_id])

            arrival_s = time.time()
            filter_timestamp_s = device_clock.update(
                usb_frame.timestamp_us, arrival_s)
            # Device power-cycle / counter restart: the recovered IMU's
            # quaternions start over, so drop the stale filter state and
            # accumulated missing counters.  Otherwise an IMU that was absent
            # before the restart keeps its old ``absent_streak``/rejected state
            # and stays reported as missing even after its data recovers.
            if device_clock.consume_reset():
                if on_stream_reset is not None:
                    on_stream_reset()
                stream_filter = QuaternionStreamFilter()
                absent_streak = np.zeros(16, dtype=np.int32)
                rejected_streak = np.zeros(16, dtype=np.int32)
                last_missing_ids = None
            filtered = stream_filter.update(
                quaternion_frame, readable, filter_timestamp_s)

            sink = diagnostics_state.get("sink") if diagnostics_state is not None else None
            if sink is not None:
                try:
                    sink({
                        "host_time_ns": time.time_ns(), "host_monotonic_s": time.monotonic(),
                        "timestamp_us": int(usb_frame.timestamp_us),
                        "sequence": int(usb_frame.sequence), "version": str(usb_frame.version),
                        "message_type": int(usb_frame.message_type),
                        "flags": int(usb_frame.flags),
                        "payload_hex": usb_frame.payload.hex(),
                        "physical_xyzw": physical_quaternions,
                        "physical_valid_mask": usb_frame.valid_mask,
                        "channel_to_hand": list(channel_to_hand),
                        "logical_filtered_xyzw": filtered.quaternions,
                        "logical_valid_mask": filtered.valid_mask,
                        "logical_sensor_age_s": filtered.sensor_age_s,
                        "logical_rejected_mask": filtered.rejected_mask,
                    })
                except Exception as exc:
                    logger.warning(f"Calibration diagnostics rejected a frame: {exc}")

            absent_streak = np.where(readable, 0, absent_streak + 1)
            rejected_streak = np.where(
                filtered.rejected_mask, rejected_streak + 1, 0)
            missing_ids = sorted(
                {
                    *(f"imu_{idx}" for idx in np.flatnonzero(absent_streak >= 3)),
                    *(f"imu_{idx}" for idx in np.flatnonzero(rejected_streak >= 3)),
                },
                key=lambda value: int(value.split("_")[1]),
            )
            if missing_ids != last_missing_ids:
                if missing_ids:
                    logger.warning(
                        f"IMU stream has {len(missing_ids)} invalid sensors: "
                        f"{missing_ids}")
                else:
                    logger.info("IMU stream recovered: all 16 sensors are ready")
                if status_callback:
                    status_callback(missing_ids)
                last_missing_ids = missing_ids

            buffer.push(ImuMsg(
                sys_ticks=int(arrival_s * 1_000_000),
                imu_rotation=R.from_quat(filtered.quaternions),
                seq=n_frame,
                valid_mask=filtered.valid_mask,
                sensor_age_s=filtered.sensor_age_s,
                rejected_mask=filtered.rejected_mask,
                # Current STM32 payload has no CALIB_STAT bytes.  0xFF is the
                # 2.1 data model's explicit "not supplied" value.
                calibration_status=calibration_status,
            ))
            n_frame += 1

            if (time.time() - diag_t) >= 10.0:
                elapsed = time.time() - diag_t
                fps = (n_frame - diag_n_frame) / elapsed
                logger.info(
                    f"IMU USB CDC fps: {fps:.1f} | CRC errors={parser.crc_errors} "
                    f"| discarded bytes={parser.discarded_bytes}")
                if pressure_buffer is not None:
                    pressure_fps = (
                        (pressure_frame_count - diag_pressure_count) / elapsed)
                    logger.info(f"Tactile USB CDC fps: {pressure_fps:.1f}")
                    diag_pressure_count = pressure_frame_count
                diag_n_frame = n_frame
                diag_t = time.time()

    if serial_handle is not None:
        try:
            serial_handle.close()
        except Exception:
            pass
    logger.info("[IMU] STM32 USB CDC receiver stopped")
