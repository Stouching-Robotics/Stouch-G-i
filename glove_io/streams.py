"""Plaintext STM32 USB acquisition used by the public sensor interfaces.

The transport only validates and decodes the firmware wire format.  It does
not remap sensors, correct axes, filter orientations, calibrate a glove, or
solve hand keypoints.  Those operations belong to the protected core behind
the public 21-keypoint interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
import threading
import time
from typing import Callable, Iterator

import numpy as np

from glove_io.usb_protocol import (
    STM32_USB_PID,
    STM32_USB_VID,
    is_supported_version,
    USB_TYPE_ADC_MATRIX,
    USB_TYPE_DELAY_RESPONSE,
    USB_TYPE_IMU_Q14,
    USB_TYPE_STATUS,
    UsbCdcFrameParser,
    UsbLinkLatency,
    decode_adc_matrix_payload,
    decode_delay_response,
    decode_imu_q14_payload,
    encode_host_probe,
    find_stm32_cdc_port,
    link_latency_from_response,
    plausible_quaternion_mask,
    supports_latency_probe,
)
from common.errors import StreamClosedError, StreamTimeoutError
from common.types import RawImuFrame, TactileFrame


StatusCallback = Callable[[str], None]
SERIAL_READ_LIMIT = 4096
# Bit masks for the 16 IMU presence bits packed into a frame's flags word.
_IMU_PRESENT_BITS = (1 << np.arange(16)).astype(np.uint32)
# How often the reader thread probes the link for round-trip latency, and how
# long an unanswered probe is kept before being discarded.
LATENCY_PROBE_INTERVAL_S = 2.0
LATENCY_PROBE_EXPIRY_S = 1.5


def _monotonic_us() -> int:
    """High-resolution monotonic microseconds, for round-trip measurement.

    ``time.time_ns()`` is far too coarse here: on Windows its smallest step was
    measured at ~995 us, so a sub-millisecond round trip lands in a single tick
    and reads as exactly 0.  ``perf_counter_ns`` is QPC-backed (100 ns steps on
    the same host) and monotonic, so a wall-clock adjustment cannot corrupt a
    measurement already in flight.  Only differences are used, so the arbitrary
    epoch is irrelevant.
    """

    return time.perf_counter_ns() // 1_000


@dataclass
class _PendingPing:
    """One in-flight v1.2.11 probe, resolved by the reader thread."""

    sequence: int
    sent_us: int
    event: threading.Event
    result: UsbLinkLatency | None = None
    # Monotonic deadline for the automatic probes; 0 disables expiry, which is
    # what on-demand pings want since they clean up in their own ``finally``.
    expires_s: float = 0.0


def _read_available(serial_handle) -> bytes:
    """Read promptly without waiting for a large CDC block to fill."""

    available = int(serial_handle.in_waiting)
    read_size = min(max(available, 1), SERIAL_READ_LIMIT)
    return serial_handle.read(read_size)


def _put_latest(queue: Queue, value) -> None:
    """Put a value without allowing a slow consumer to block acquisition."""

    while True:
        try:
            queue.put_nowait(value)
            return
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                return


def _drain(queue: Queue, maximum: int | None = None) -> list:
    values = []
    limit = None if maximum is None else max(0, int(maximum))
    while limit is None or len(values) < limit:
        try:
            values.append(queue.get_nowait())
        except Empty:
            break
    return values


class _UsbSensorTransport:
    """One serial owner that multiplexes raw IMU and tactile frames."""

    def __init__(
        self,
        serial_port: str | None,
        usb_vid: int,
        usb_pid: int,
        imu_queue_size: int,
        tactile_queue_size: int,
        status_callback: StatusCallback | None,
    ):
        self.requested_port = serial_port
        self.usb_vid = int(usb_vid)
        self.usb_pid = int(usb_pid)
        self.imu_queue: Queue[RawImuFrame] = Queue(
            maxsize=max(2, int(imu_queue_size)))
        self.tactile_queue: Queue[TactileFrame] = Queue(
            maxsize=max(2, int(tactile_queue_size)))
        self.status_callback = status_callback
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.active_port: str | None = None
        self.connected = False
        self.last_error: str | None = None
        self.last_status: str = "idle"
        self.firmware_version: str | None = None
        self._state_lock = threading.RLock()
        # v1.2.11 latency probes.  The reader thread owns the serial handle's
        # read side; probes are written from the caller's thread, so both the
        # handle reference and the in-flight table are lock-protected.
        self._serial_handle = None
        self._write_lock = threading.Lock()
        self._ping_lock = threading.Lock()
        self._pending_pings: dict[int, _PendingPing] = {}
        self._ping_sequence = 0
        # Latest link round trip, refreshed by the reader thread's own probes.
        self.latency: UsbLinkLatency | None = None
        self._next_probe_s = 0.0

    def _set_status(self, value: str, error: str | None = None) -> None:
        callback = None
        with self._state_lock:
            changed = value != self.last_status or error != self.last_error
            self.last_status = value
            self.last_error = error
            if changed:
                callback = self.status_callback
        if callback is not None:
            callback(value)

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="stouch-usb-sensor",
            daemon=True,
        )
        self.thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self.stop_event.set()
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_s)))
        with self._state_lock:
            self.connected = False
            self._serial_handle = None
        # Release any caller still waiting on a probe so it cannot hang for the
        # remainder of its timeout after the stream is gone.
        self._fail_pending_pings()
        self._set_status("stopped")

    def _maybe_probe_latency(self) -> None:
        """Send a latency probe on schedule, if the firmware understands one.

        Runs on the reader thread, which owns the serial handle's read side and
        already consumes the type 0x04 answer, so the measurement costs the
        caller nothing and never blocks a consumer.  Probing is skipped unless
        the device has announced v1.2.11+, because older firmware does not know
        the probe frame.
        """

        now = time.monotonic()
        self._expire_pending_pings(now)
        if now < self._next_probe_s:
            return
        self._next_probe_s = now + LATENCY_PROBE_INTERVAL_S

        version = self.firmware_version
        if version is None or not supports_latency_probe(version):
            return
        with self._state_lock:
            handle = self._serial_handle
        if handle is None:
            return

        sent_us = _monotonic_us()
        with self._ping_lock:
            self._ping_sequence = (self._ping_sequence + 1) & 0xFFFF
            sequence = self._ping_sequence
            self._pending_pings[sequence] = _PendingPing(
                sequence=sequence,
                sent_us=sent_us,
                event=threading.Event(),
                expires_s=now + LATENCY_PROBE_EXPIRY_S,
            )
        try:
            with self._write_lock:
                handle.write(encode_host_probe(sequence, sent_us))
        except Exception as exc:
            with self._ping_lock:
                self._pending_pings.pop(sequence, None)
            self._set_status("error", f"latency probe write failed: {exc}")

    def _expire_pending_pings(self, now: float) -> None:
        """Drop probes the device never answered so the table cannot grow."""

        with self._ping_lock:
            stale = [
                sequence for sequence, item in self._pending_pings.items()
                if item.expires_s and now >= item.expires_s
            ]
            for sequence in stale:
                self._pending_pings.pop(sequence, None)

    def _fail_pending_pings(self) -> None:
        with self._ping_lock:
            pending = list(self._pending_pings.values())
            self._pending_pings.clear()
        for item in pending:
            item.event.set()

    def _resolve_ping(self, payload: bytes) -> None:
        """Match a type 0x04 response to its probe and publish the round trip."""

        try:
            response = decode_delay_response(payload)
        except ValueError as exc:
            self._set_status("error", f"invalid delay response: {exc}")
            return
        with self._ping_lock:
            pending = self._pending_pings.get(int(response.sequence))
        if pending is None:
            return
        pending.result = link_latency_from_response(
            response,
            host_receive_us=_monotonic_us(),
            host_send_us=pending.sent_us,
        )
        self.latency = pending.result
        pending.event.set()

    def ping(self, timeout_s: float = 1.0) -> UsbLinkLatency | None:
        """Measure the host<->device round trip using a v1.2.11 probe.

        Returns ``None`` when the stream is down, when the device has not
        announced v1.2.11+ (older firmware does not know the probe frame, so no
        probe is sent), or when no answer arrives within ``timeout_s``.  For
        display purposes read :attr:`latency` instead, which the reader thread
        refreshes on its own without blocking anything.
        """

        with self._state_lock:
            handle = self._serial_handle
            version = self.firmware_version
        if handle is None:
            return None
        if version is None or not supports_latency_probe(version):
            return None

        sent_us = _monotonic_us()
        with self._ping_lock:
            self._ping_sequence = (self._ping_sequence + 1) & 0xFFFF
            sequence = self._ping_sequence
            pending = _PendingPing(
                sequence=sequence, sent_us=sent_us, event=threading.Event())
            self._pending_pings[sequence] = pending

        try:
            with self._write_lock:
                handle.write(encode_host_probe(sequence, sent_us))
        except Exception as exc:
            with self._ping_lock:
                self._pending_pings.pop(sequence, None)
            self._set_status("error", f"ping write failed: {exc}")
            return None

        try:
            if not pending.event.wait(max(0.0, float(timeout_s))):
                return None
            return pending.result
        finally:
            with self._ping_lock:
                self._pending_pings.pop(sequence, None)

    def _run(self) -> None:
        try:
            import serial
        except ImportError as exc:
            self._set_status("error", "pyserial is not installed")
            return

        parser = UsbCdcFrameParser()
        serial_handle = None
        while not self.stop_event.is_set():
            if serial_handle is None:
                try:
                    port = self.requested_port or find_stm32_cdc_port(
                        self.usb_vid, self.usb_pid)
                    if not port:
                        self._set_status(
                            "waiting",
                            f"STM32 USB CDC {self.usb_vid:04X}:{self.usb_pid:04X} not found",
                        )
                        self.stop_event.wait(0.5)
                        continue
                    serial_handle = serial.Serial(
                        port, baudrate=115200, timeout=0.2)
                    parser.reset()
                    with self._state_lock:
                        self.active_port = str(port)
                        self.connected = True
                        self._serial_handle = serial_handle
                    self._set_status("connected", None)
                except (OSError, serial.SerialException) as exc:
                    with self._state_lock:
                        self.connected = False
                        self._serial_handle = None
                    self._set_status("waiting", f"cannot open serial port: {exc}")
                    serial_handle = None
                    self.stop_event.wait(0.5)
                    continue

            try:
                # ``read(4096)`` waits for a nearly full 4 KiB block (or the
                # 200 ms timeout) on Windows.  Since IMU and tactile packets
                # share this CDC stream, that batches roughly 6-8 IMU samples
                # into each host read.  Real-time consumers intentionally keep
                # only the newest queued IMU frame, so the batching turns an
                # 80 Hz device stream into about 10 Hz of solved hand poses.
                #
                # Block for one byte when idle, then immediately drain whatever
                # the driver already has.  The frame parser accepts partial
                # packets, so this preserves throughput while removing the
                # host-side batching delay.
                data = _read_available(serial_handle)
            except (OSError, serial.SerialException) as exc:
                self._set_status("disconnected", f"serial read failed: {exc}")
                with self._state_lock:
                    self.connected = False
                    self._serial_handle = None
                self._fail_pending_pings()
                try:
                    serial_handle.close()
                except Exception:
                    pass
                serial_handle = None
                self.stop_event.wait(0.2)
                continue

            # Time-driven, so it still fires on a link that is idle right now.
            self._maybe_probe_latency()

            if not data:
                continue
            for wire_frame in parser.feed(data):
                host_timestamp_us = time.time_ns() // 1_000
                with self._state_lock:
                    self.firmware_version = wire_frame.version
                if not is_supported_version(wire_frame.version):
                    self._set_status(
                        "error",
                        f"unsupported USB protocol version {wire_frame.version}",
                    )
                    continue
                try:
                    if wire_frame.message_type == USB_TYPE_IMU_Q14:
                        quaternions = decode_imu_q14_payload(wire_frame.payload)
                        present = (wire_frame.valid_mask & _IMU_PRESENT_BITS) != 0
                        plausible = plausible_quaternion_mask(quaternions)
                        _put_latest(self.imu_queue, RawImuFrame(
                            sequence=int(wire_frame.sequence),
                            device_timestamp_us=int(wire_frame.timestamp_us),
                            host_timestamp_us=int(host_timestamp_us),
                            quaternions_xyzw=quaternions,
                            present_mask=present,
                            valid_mask=present & plausible,
                            mag_ready=bool(wire_frame.mag_ready),
                        ))
                    elif wire_frame.message_type == USB_TYPE_ADC_MATRIX:
                        matrix = decode_adc_matrix_payload(
                            wire_frame.payload,
                            sequence=wire_frame.sequence,
                            scan_time_us=wire_frame.timestamp_us,
                        )
                        _put_latest(self.tactile_queue, TactileFrame(
                            sequence=int(matrix.sequence),
                            timestamp_us=int(matrix.scan_time_us),
                            samples=matrix.samples,
                            processed=False,
                        ))
                    elif wire_frame.message_type == USB_TYPE_DELAY_RESPONSE:
                        self._resolve_ping(wire_frame.payload)
                    elif wire_frame.message_type == USB_TYPE_STATUS:
                        status = wire_frame.payload.decode(
                            "utf-8", errors="replace").strip()
                        if status:
                            self._set_status(f"device: {status}", None)
                except ValueError as exc:
                    self._set_status("error", f"invalid USB payload: {exc}")

        if serial_handle is not None:
            try:
                serial_handle.close()
            except Exception:
                pass
        with self._state_lock:
            self.connected = False
            self._serial_handle = None
        self._fail_pending_pings()


class RawImuStream:
    """Public physical-channel IMU stream.

    Frames are exactly the decoded Q14 values in STM32 physical channel order.
    No sensor mapping, axis conversion, smoothing, calibration, or FK is
    applied.  Use :class:`runtime.HandSolver.process` for 21-keypoint output.
    """

    def __init__(
        self,
        serial_port: str | None = None,
        *,
        usb_vid: int = STM32_USB_VID,
        usb_pid: int = STM32_USB_PID,
        buffer_size: int = 256,
        tactile_buffer_size: int = 2048,
        status_callback: StatusCallback | None = None,
        _transport: _UsbSensorTransport | None = None,
    ):
        self._transport = _transport or _UsbSensorTransport(
            serial_port,
            usb_vid,
            usb_pid,
            buffer_size,
            tactile_buffer_size,
            status_callback,
        )

    def start(self) -> "RawImuStream":
        self._transport.start()
        return self

    def stop(self, timeout_s: float = 3.0) -> None:
        self._transport.stop(timeout_s)

    close = stop

    def __enter__(self) -> "RawImuStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def connected(self) -> bool:
        return self._transport.connected

    @property
    def port(self) -> str | None:
        return self._transport.active_port

    @property
    def status(self) -> str:
        return self._transport.last_status

    @property
    def firmware_version(self) -> str | None:
        return self._transport.firmware_version

    @property
    def last_error(self) -> str | None:
        return self._transport.last_error

    def read(self, timeout: float | None = None) -> RawImuFrame:
        self.start()
        try:
            return self._transport.imu_queue.get(timeout=timeout)
        except Empty as exc:
            if self._transport.stop_event.is_set():
                raise StreamClosedError("raw IMU stream is closed") from exc
            raise StreamTimeoutError("timed out waiting for a raw IMU frame") from exc

    def poll(self, maximum: int | None = None) -> list[RawImuFrame]:
        """Return all currently buffered frames without blocking."""

        return _drain(self._transport.imu_queue, maximum)

    def frames(self, timeout: float = 0.25) -> Iterator[RawImuFrame]:
        self.start()
        while not self._transport.stop_event.is_set():
            try:
                yield self.read(timeout=timeout)
            except StreamTimeoutError:
                continue

    def tactile_stream(self) -> "TactileStream":
        """Return a tactile interface sharing this stream's serial handle."""

        return TactileStream(imu=self)

    def poll_tactile(self, maximum: int | None = None) -> list[TactileFrame]:
        return _drain(self._transport.tactile_queue, maximum)

    @property
    def latency(self) -> UsbLinkLatency | None:
        """Latest link round trip, refreshed in the background every ~2 s.

        ``None`` until the first answer arrives, which for firmware older than
        v1.2.11 is permanent -- that firmware does not know the probe frame.
        """

        return self._transport.latency

    def ping(self, timeout_s: float = 1.0) -> UsbLinkLatency | None:
        """Measure the host<->device round trip (firmware v1.2.11+).

        The same call works on the wired USB CDC link and on a Bluetooth dongle
        port, since a dongle relays the SPP stream byte for byte.  Returns
        ``None`` when the link is down, when the device has not announced
        v1.2.11+, or when no answer arrives in time.
        """

        self.start()
        return self._transport.ping(timeout_s)


class TactileStream:
    """Public raw 16x16 tactile stream.

    Pass ``imu=raw_stream`` to share the same USB connection.  A standalone
    tactile stream owns its transport and discards unconsumed IMU frames.
    """

    def __init__(
        self,
        imu: RawImuStream | None = None,
        *,
        serial_port: str | None = None,
        usb_vid: int = STM32_USB_VID,
        usb_pid: int = STM32_USB_PID,
        buffer_size: int = 2048,
        status_callback: StatusCallback | None = None,
    ):
        self._owner = imu is None
        self._imu = imu or RawImuStream(
            serial_port,
            usb_vid=usb_vid,
            usb_pid=usb_pid,
            buffer_size=16,
            tactile_buffer_size=buffer_size,
            status_callback=status_callback,
        )

    def start(self) -> "TactileStream":
        self._imu.start()
        return self

    def stop(self, timeout_s: float = 3.0) -> None:
        if self._owner:
            self._imu.stop(timeout_s)

    close = stop

    def __enter__(self) -> "TactileStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def connected(self) -> bool:
        return self._imu.connected

    def read(self, timeout: float | None = None) -> TactileFrame:
        self.start()
        try:
            return self._imu._transport.tactile_queue.get(timeout=timeout)
        except Empty as exc:
            if self._imu._transport.stop_event.is_set():
                raise StreamClosedError("tactile stream is closed") from exc
            raise StreamTimeoutError("timed out waiting for a tactile frame") from exc

    def poll(self, maximum: int | None = None) -> list[TactileFrame]:
        return self._imu.poll_tactile(maximum)

    def frames(self, timeout: float = 0.25) -> Iterator[TactileFrame]:
        self.start()
        while not self._imu._transport.stop_event.is_set():
            try:
                yield self.read(timeout=timeout)
            except StreamTimeoutError:
                continue


# Explicit name for callers that want one object for both sensor families.
SensorStream = RawImuStream


__all__ = ["RawImuStream", "SensorStream", "TactileStream"]
