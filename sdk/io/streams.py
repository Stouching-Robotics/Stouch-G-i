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

from sdk.io.usb_protocol import (
    BLUETOOTH_DONGLE_PID,
    LINK_KIND_BLUETOOTH,
    LINK_KIND_USB,
    STM32_USB_PID,
    STM32_USB_VID,
    is_supported_version,
    MagTelemetryCache,
    MagTelemetrySnapshot,
    PARAM_RAW_ACK_PAYLOAD_SIZE,
    PARAM_STATUS_OK,
    PARAM_SUBCMD_CLEAR_ALL,
    PARAM_SUBCMD_QUERY_STATUS,
    PARAM_SUBCMD_READ_SET,
    PARAM_SUBCMD_START_FIT,
    PARAM_SUBCMD_WRITE_SHARD,
    PRESSURE_FIT_BLOB_SIZE,
    PRESSURE_FIT_SET_COUNT,
    PRESSURE_FIT_SHARD_COUNT,
    PRESSURE_FIT_SHARD_SIZE,
    PressureFitCache,
    PressureFitSnapshot,
    USB_TYPE_ADC_MATRIX,
    USB_TYPE_DELAY_RESPONSE,
    USB_TYPE_IMU_Q14,
    USB_TYPE_MAG_TELEMETRY,
    USB_TYPE_PARAM_ACK,
    USB_TYPE_PRESSURE_FIT_PARAMS,
    USB_TYPE_STATUS,
    AdcMatrixAssembler,
    UsbCdcFrameParser,
    UsbLinkLatency,
    UsbParamAck,
    UsbPressureFitParams,
    UsbRawParamAck,
    decode_delay_response,
    decode_imu_q14_payload,
    decode_param_ack_payload,
    decode_raw_param_ack_payload,
    encode_host_probe,
    encode_param_command,
    encode_raw_param_command,
    find_stm32_cdc_port,
    link_latency_from_response,
    plausible_quaternion_mask,
    pressure_fit_crc32,
    supports_latency_probe,
    supports_pressure_fit_params,
)
from common.errors import StreamClosedError, StreamTimeoutError
from common.latency_probe import stamp as _latency_stamp
from common.link_quality import SequenceDropCounter, sequence_mask
from common.types import RawImuFrame, TactileFrame


StatusCallback = Callable[[str], None]
SERIAL_READ_LIMIT = 4096
# A write must not be able to park the thread holding it.  The reader thread is
# the only thing draining this port, and it also sends the latency probe, so a
# ``write()`` that blocks until the OS accepts the bytes stops acquisition
# outright: nothing is read while it waits, the dongle's buffer backs up, and
# the next read hands over a burst after a gap -- which is exactly the shape a
# congested link is diagnosed by.  Bounding the write turns that into a dropped
# probe.  Generous on purpose: a backstop against a wedged handle, not a path a
# healthy link should ever reach.
SERIAL_WRITE_TIMEOUT_S = 1.0
# Bit masks for the 16 IMU presence bits packed into a frame's flags word.
_IMU_PRESENT_BITS = (1 << np.arange(16)).astype(np.uint32)
# How often the reader thread probes the link for round-trip latency, and how
# long an unanswered probe is kept before being discarded.
LATENCY_PROBE_INTERVAL_S = 2.0
LATENCY_PROBE_EXPIRY_S = 1.5
# A silent Bluetooth link is reopened independently by the reader which owns
# that COM port.  CRC/loss spikes are deliberately *not* used to bounce it:
# hardware testing proved those spikes come from two-radio interference, and
# repeatedly reopening the same port cannot cure interference.
BT_VALID_FRAME_SILENCE_S = 2.0
BT_RECOVERY_COOLDOWN_S = 12.0
BT_REOPEN_DELAY_S = 1.0

# v1.2.16 parameter commands.  A shard gets one write attempt per retry, and an
# unanswered one is resent byte for byte -- the device overwrites the same
# staging offset, so a duplicate is harmless.  The timeout is generous because
# writing a set stalls the device's stream for ~23 ms (spec 5.3) and an answer
# must not be charged against a command that did arrive.
PARAM_ACK_TIMEOUT_S = 0.5
PARAM_SHARD_RETRIES = 5
# Spec 5.1 applies to the whole command channel, not just the write shards:
# "每条命令都要发到收到应答为止 … 未收到应答就重发同一条命令即可（本命令通道是幂等的）".
# A single dropped byte or frame must not turn a read, a query or an erase into
# a reported failure, so each command is sent up to this many times.
PARAM_COMMAND_ATTEMPTS = 3
# After a 0x04 query's own 0x07 ack, any 0x06 broadcast still in flight arrives
# within this window (sets are announced ~30 ms apart, spec 4.7).
PARAM_BROADCAST_SLACK_S = 0.5
# How many times a whole set is restarted from offset 0 after a device-side
# rejection.  Bounded so a device that answers "flash write failed" forever
# cannot turn into a busy loop.
PARAM_SET_RESTARTS = 2
# How long the public parameter entry points wait for the device to announce a
# firmware version before judging it.  The version arrives with the first frame
# the reader thread parses, which on a port opened before the glove was powered
# up is 2 to 3 seconds later (spec 7.2), so this is sized for that boot rather
# than for a command round trip.
PARAM_VERSION_WAIT_S = 3.0
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


@dataclass
class _PendingParamAck:
    """One in-flight v1.2.16 0x51 command, resolved by the reader thread.

    Unlike a ping, this carries no unique key.  A 0x07 answer echoes only the
    subcommand and the set number -- no offset, no nonce -- so two commands for
    the same ``(subcommand, set_index)`` pair are indistinguishable on the wire.
    That is why the transport keeps a single slot rather than a table and
    serialises the channel instead: the protocol itself is "send until it
    answers", one command at a time.
    """

    subcommand: int
    set_index: int
    event: threading.Event
    result: UsbParamAck | None = None


@dataclass
class _PendingRawParamAck:
    """One in-flight compact whole-block parameter command."""

    subcommand: int
    event: threading.Event
    result: UsbRawParamAck | None = None


@dataclass(frozen=True)
class PressureFitReadOutcome:
    """Result of reading one stored set (0x02 -> 0x06 + 0x07).

    ``ack`` is the device's answer, and it is the *only* place a whole-set
    CRC32 appears: a 0x06 payload carries 套号/标志/格式/手型/写入序号 and then
    k and b, with no checksum field of its own (spec 4.7).  The CRC32 the host
    can genuinely compare against its own computation is the one in the 0x07
    ack (spec 4.8), which is why it is handed back rather than kept internal.
    """

    set_index: int
    ok: bool
    params: UsbPressureFitParams | None = None
    ack: UsbParamAck | None = None
    message: str = ""

    @property
    def device_crc32(self) -> int | None:
        """The CRC32 the device says it holds for this set, or ``None``.

        On a successful read this is comparable with
        ``pressure_fit_crc32(outcome.params.blob)``; a mismatch means the bytes
        that arrived are not the bytes stored.
        """

        return None if self.ack is None else self.ack.crc32


@dataclass(frozen=True)
class PressureFitCommandOutcome:
    """Result of a parameter command that returns no coefficient payload.

    ``clear`` (0x03) uses ``ok``/``ack``/``message``; ``query`` (0x04) adds the
    sets whose 0x06 broadcasts arrived (``params``) and the ones the device's
    ``valid_mask`` claimed but that never came (``missing``).  ``message`` is
    what keeps "the device answered 套号非法" apart from "nothing came back at
    all" -- both used to reach the caller as the same empty result.
    """

    ok: bool
    ack: UsbParamAck | None = None
    params: dict[int, UsbPressureFitParams] | None = None
    missing: tuple[int, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class PressureFitWriteOutcome:
    """Result of writing one whole parameter set (four shards, spec 5.3)."""

    set_index: int
    ok: bool
    crc32: int
    write_index: int | None = None
    valid_mask: int | None = None
    shards_sent: int = 0
    retries_used: int = 0
    restarts: int = 0
    message: str = ""


@dataclass(frozen=True)
class RawParamReadOutcome:
    """Result of the current ``0x51/0x02`` raw-block read."""

    ok: bool
    data: bytes | None = None
    ack: UsbRawParamAck | None = None
    message: str = ""


@dataclass(frozen=True)
class RawParamWriteOutcome:
    """Result of the current one-command ``0x51/0x01`` raw-block write."""

    ok: bool
    length: int
    crc32: int
    ack: UsbRawParamAck | None = None
    message: str = ""


def write_pressure_fit_shards(
        blob: bytes,
        set_index: int,
        send: Callable[[bytes, float], "UsbParamAck | None"],
        *,
        timeout_s: float = PARAM_ACK_TIMEOUT_S,
        retries: int = PARAM_SHARD_RETRIES,
        restarts: int = PARAM_SET_RESTARTS,
        progress: Callable[[int, bool, int], None] | None = None,
) -> PressureFitWriteOutcome:
    """Drive the spec-5.3 four-shard write of one 1024-byte parameter set.

    ``send`` is the entire dependency on the outside world: it takes an encoded
    0x51 command plus a timeout and returns the matching 0x07 answer, or
    ``None`` when nothing arrived.  Keeping the retry and restart policy here,
    rather than inline in the transport, is what makes the shard timing
    testable without a glove attached.

    The rules, all from spec 5.3:

    * four 256-byte shards at offsets 0/256/512/768, every one carrying the
      same whole-set CRC32;
    * a shard with no answer is resent *unchanged* -- same offset, same bytes,
      a plain overwrite on the device, hence idempotent;
    * a non-zero status restarts the whole set from offset 0, because offset 0
      is the "a new set begins here" marker that clears the device's staging
      buffer, making it the only correct recovery point;
    * the final shard's ack must report the CRC32 that was sent and set the
      corresponding bit in ``valid_mask``; intermediate acknowledgements are
      not a commit result.

    The two recovery paths are deliberately different and must not be mixed:
    silence resends one shard, a rejection restarts the set.
    """

    blob = bytes(blob)
    if len(blob) != PRESSURE_FIT_BLOB_SIZE:
        raise ValueError(
            f"a parameter set is {PRESSURE_FIT_BLOB_SIZE} bytes, "
            f"got {len(blob)}")
    set_index = int(set_index)
    if not 0 <= set_index < PRESSURE_FIT_SET_COUNT:
        raise ValueError(
            f"set_index must be 0..{PRESSURE_FIT_SET_COUNT - 1}, "
            f"got {set_index}")

    crc = pressure_fit_crc32(blob)
    shards_sent = 0
    retries_used = 0
    restarts_used = 0
    # The final shard's ack is the only one that describes the set *after* the
    # write: it carries the write index the device just stamped and the mask
    # that should now include this set.  Intermediate acks answer for the
    # previous contents, so they are not kept.
    final_ack = None

    while True:
        failure = None
        for shard in range(PRESSURE_FIT_SHARD_COUNT):
            offset = shard * PRESSURE_FIT_SHARD_SIZE
            chunk = blob[offset:offset + PRESSURE_FIT_SHARD_SIZE]
            command = encode_param_command(
                PARAM_SUBCMD_WRITE_SHARD, set_index=set_index,
                offset=offset, length=len(chunk), crc32=crc, data=chunk)
            last = shard == PRESSURE_FIT_SHARD_COUNT - 1

            for attempt in range(retries + 1):
                if attempt:
                    retries_used += 1
                ack = send(command, timeout_s)
                shards_sent += 1
                if progress is not None:
                    progress(shard, ack is not None and ack.ok, attempt)
                if ack is None:
                    # Link silence, not a verdict: resend this same shard.
                    continue
                if ack.ok:
                    if last and ack.crc32 != crc:
                        # A zero CRC plus a clear mask means the device never
                        # committed the complete four-shard staging buffer.
                        # Restart the whole set so a dropped final shard cannot
                        # be reported as a successful write.
                        if (ack.crc32 == 0
                                and not (ack.valid_mask & (1 << set_index))):
                            failure = (
                                f"final shard was acknowledged but set "
                                f"{set_index} was not committed")
                            break
                        # A non-zero mismatch is a real storage discrepancy;
                        # preserve it as the finding rather than hiding it
                        # behind another write attempt.
                        return PressureFitWriteOutcome(
                            set_index=set_index, ok=False, crc32=crc,
                            write_index=ack.write_index,
                            valid_mask=ack.valid_mask,
                            shards_sent=shards_sent,
                            retries_used=retries_used,
                            restarts=restarts_used,
                            message=(
                                f"device stored a different blob: its crc32 is "
                                f"{ack.crc32:#010x}, sent {crc:#010x}"),
                        )
                    if last and not (ack.valid_mask & (1 << set_index)):
                        failure = (
                            f"final shard ack did not mark set {set_index} "
                            f"valid (mask=0x{ack.valid_mask:02x})")
                        break
                    if last:
                        final_ack = ack
                    break
                failure = (f"shard {shard} rejected: {ack.status_name}")
                break
            else:
                failure = (
                    f"shard {shard} got no answer after {retries + 1} attempts")

            if failure is not None:
                break

        if failure is None:
            return PressureFitWriteOutcome(
                set_index=set_index, ok=True, crc32=crc,
                write_index=(final_ack.write_index
                             if final_ack is not None else None),
                valid_mask=(final_ack.valid_mask
                            if final_ack is not None else None),
                shards_sent=shards_sent, retries_used=retries_used,
                restarts=restarts_used)

        if restarts_used >= restarts:
            return PressureFitWriteOutcome(
                set_index=set_index, ok=False, crc32=crc,
                shards_sent=shards_sent, retries_used=retries_used,
                restarts=restarts_used,
                message=f"{failure}; giving up after {restarts} restart(s)")
        restarts_used += 1


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
        self._is_bluetooth = self.usb_pid == BLUETOOTH_DONGLE_PID
        # The pressure dialect follows the same test.  It is read from the
        # transport's own pid -- the one ``sdk.transport`` rebinds to the link
        # the port is on *now* -- rather than from the port table at open time,
        # so a parser, an assembler and the recovery policy below can never end
        # up describing different links.
        self._link_kind = (LINK_KIND_BLUETOOTH if self._is_bluetooth
                           else LINK_KIND_USB)
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
        # Set once a frame has announced the version, so a caller that needs to
        # know what this device speaks can wait for it instead of reading the
        # attribute before the reader thread has parsed anything -- see
        # :meth:`wait_for_version`.
        self._version_event = threading.Event()
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
        # Latest v1.2.13 magnetometer telemetry, merged by the reader thread.
        # The cache owns its own lock, so it needs no ``_state_lock`` here.
        self._mag_cache = MagTelemetryCache()
        # Latest v1.2.16 stored parameters, one entry per set.  Same contract
        # as the magnetometer cache: it owns its own lock and is written only
        # by the reader thread.
        self._fit_cache = PressureFitCache()
        # The pressure picture the sparse/bitmap 0x03 patches are deltas
        # against.  Same contract as the caches above -- reader-thread-owned,
        # reset on reopen -- with the one difference that matters: it is *per
        # transport*, so two hands on two links can never patch each other's
        # picture.
        #
        # Only a Bluetooth link sends patches at all; on a wired one this
        # assembler is a decoder that refuses them, which is what keeps a frame
        # the wired firmware cannot produce from being merged as a reading.
        self._matrix_assembler = AdcMatrixAssembler(link_kind=self._link_kind)
        # v1.2.16 parameter commands.  A 0x07 answer echoes only the subcommand
        # and set number, so the channel is serialised and holds at most one
        # waiter -- see _PendingParamAck.
        self._param_lock = threading.Lock()
        self._param_command_lock = threading.Lock()
        self._pending_param: _PendingParamAck | None = None
        self._raw_param_lock = threading.Lock()
        self._pending_raw_param: _PendingRawParamAck | None = None
        # Latest complete type-0x06 payload, regardless of whether it uses the
        # legacy per-set layout or the current opaque raw-block layout.
        self._raw_param_frame: tuple[object, bytes] | None = None
        # 0x07 frames that matched no waiter, or the wrong one.  The protocol
        # cannot rule mispairing out on its own, so this is the diagnostic that
        # says whether it is happening; non-zero is not itself an error.
        self.param_unmatched = 0
        self._next_probe_s = 0.0
        # The frame parser's error tally, republished here so it can be read
        # from outside: the parser itself is a local of ``_run``, so its
        # counters used to be unreachable.  Cumulative for the lifetime of this
        # transport -- callers wanting a rate take deltas.  Replaced wholesale
        # (never mutated) so a reader on another thread always sees a
        # consistent triple.
        self.link_errors: dict[str, int] = {
            "crc_errors": 0,
            "length_errors": 0,
            "discarded_bytes": 0,
            "recoveries": 0,
        }
        self._recovery_count = 0
        # Sequence-gap loss, same publishing contract as ``link_errors``.  This
        # is the one measurement that separates a link that is dropping frames
        # from a host that is discarding them, so it is fed every frame the
        # parser produces -- all five types share the device's counter.
        self._drop_counter = SequenceDropCounter()
        self.link_drops: dict[str, int] = {"frames": 0, "dropped": 0}

    @property
    def mag_telemetry(self) -> MagTelemetrySnapshot | None:
        """Latest independently cached v1.2.13 magnetic telemetry."""

        return self._mag_cache.snapshot

    @property
    def pressure_fit_snapshots(self) -> dict[int, PressureFitSnapshot]:
        """Cached v1.2.16 parameters, keyed by set index.  No I/O."""

        return self._fit_cache.snapshots()

    def pressure_fit_snapshot(
            self, set_index: int) -> PressureFitSnapshot | None:
        """Latest cached parameters for one set, or ``None``.  No I/O."""

        return self._fit_cache.snapshot(set_index)

    def _update_pressure_fit(
            self, wire_frame, host_timestamp_us: int) -> None:
        """Store a type-0x06 frame for both legacy and raw-block firmware."""

        payload = bytes(wire_frame.payload)
        with self._raw_param_lock:
            self._raw_param_frame = (object(), payload)
        # Keep the established six-set cache when the payload is the legacy
        # 1032-byte layout.  The current firmware's opaque block has no set
        # metadata, so it is retained separately for the raw read API.
        try:
            self._fit_cache.update(wire_frame, host_timestamp_us)
        except ValueError:
            pass

    def _resolve_raw_param_ack(self, payload: bytes) -> None:
        """Resolve a compact 8-byte acknowledgement for a raw block."""

        try:
            ack = decode_raw_param_ack_payload(payload)
        except ValueError as exc:
            self._set_status("error", f"invalid raw parameter ack: {exc}")
            return
        with self._raw_param_lock:
            pending = self._pending_raw_param
            if pending is None or ack.subcommand != pending.subcommand:
                self.param_unmatched += 1
                return
            pending.result = ack
            pending.event.set()

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

    def _sync_link_errors(self, parser) -> None:
        """Republish the reader's parser tally after one drained read.

        Called once per read rather than once per frame: the parser is owned
        and consumed entirely by the reader thread, so this is the cheapest
        point that still keeps the published triple within one read of the
        truth.  The counters are cumulative, so a consumer that wants a rate
        has to keep the previous triple and subtract.
        """

        self.link_errors = {
            "crc_errors": parser.crc_errors,
            "length_errors": parser.length_errors,
            "discarded_bytes": parser.discarded_bytes,
            "recoveries": self._recovery_count,
            # Not a parser error, but the only way to tell a frozen pressure
            # panel from a quiet hand: pressure patches dropped for want of a
            # keyframe to be a delta against.  Zero on firmware that sends full
            # frames every scan, which is what makes a non-zero count mean the
            # stream opened mid-difference rather than something worse.
            "matrix_baseline_waits": self._matrix_assembler.baseline_waits,
        }
        self.link_drops = self._drop_counter.snapshot()

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
        self._fail_pending_param()
        # Same for a caller waiting on the version announcement: no frame is
        # coming once the stream is gone, so let it out now.
        self._version_event.set()
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
        # The BP101Y link is half-duplex at the application boundary and the
        # gloves already fill most of its UART budget with IMU + pressure data.
        # Automatic two-second probes add reverse traffic exactly when a weak
        # link is congested.  Explicit ``ping()`` remains available for a
        # deliberate diagnostic, but live Bluetooth acquisition stays one-way.
        if self._is_bluetooth:
            return
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

    def _resolve_param_ack(self, payload: bytes) -> None:
        """Match a type 0x07 answer to the command that is waiting for it."""

        if len(payload) == PARAM_RAW_ACK_PAYLOAD_SIZE:
            self._resolve_raw_param_ack(payload)
            return

        try:
            ack = decode_param_ack_payload(payload)
        except ValueError as exc:
            self._set_status("error", f"invalid parameter ack: {exc}")
            return
        with self._param_lock:
            pending = self._pending_param
            if (pending is None
                    or ack.subcommand != pending.subcommand
                    or ack.set_index != pending.set_index):
                self.param_unmatched += 1
                return
            pending.result = ack
            pending.event.set()

    def _fail_pending_param(self) -> None:
        """Release a waiting command when the link goes away.

        Called next to every :meth:`_fail_pending_pings` call site, so a caller
        never sits out its whole timeout after the port has already gone.  Takes
        only ``_param_lock`` and never touches the write lock, so it cannot
        deadlock against the reader thread.
        """

        with self._param_lock:
            pending = self._pending_param
            self._pending_param = None
        if pending is not None:
            pending.event.set()
        with self._raw_param_lock:
            raw_pending = self._pending_raw_param
            self._pending_raw_param = None
        if raw_pending is not None:
            raw_pending.event.set()

    def _send_param_command(
            self,
            command: bytes,
            timeout_s: float = PARAM_ACK_TIMEOUT_S,
    ) -> UsbParamAck | None:
        """Write one 0x51 command and wait for its 0x07 answer.

        One attempt only: the caller owns the retry policy (see
        :func:`write_pressure_fit_shards`).  Returns ``None`` when the link is
        down, when the firmware predates the command, or when nothing arrived
        in time.

        ``timeout_s`` is positional-or-keyword rather than keyword-only like
        the public methods here, because this one is passed *as* the ``send``
        callback :func:`write_pressure_fit_shards` calls as ``send(cmd, t)``.
        Making it keyword-only silently broke that wiring, and only on real
        hardware: the shard policy is tested with an injected ``send``, which
        never goes through this method.
        """

        with self._state_lock:
            handle = self._serial_handle
            version = self.firmware_version
        if handle is None:
            return None
        if version is None or not supports_pressure_fit_params(version):
            return None

        # ``A5 5A | 51 | subcmd | set`` is the encoder's guaranteed layout, so
        # the echo keys can be read straight off the bytes we are about to
        # send.  They sit one byte further along than the type byte: byte 2 is
        # the 0x51 type itself, which the device never echoes, so reading the
        # subcommand from there would match nothing and every command would
        # time out with ``param_unmatched`` climbing.
        pending = _PendingParamAck(
            subcommand=command[3], set_index=command[4],
            event=threading.Event())
        with self._param_lock:
            self._pending_param = pending
        try:
            with self._write_lock:
                handle.write(command)
        except Exception as exc:
            with self._param_lock:
                if self._pending_param is pending:
                    self._pending_param = None
            self._set_status("error", f"parameter command write failed: {exc}")
            return None

        try:
            if not pending.event.wait(max(0.0, float(timeout_s))):
                return None
            return pending.result
        finally:
            # Identity check rather than a keyed pop: a retry of the same shard
            # reuses the same key, so only this attempt's own slot may be
            # cleared or a newer waiter's would be clobbered.
            with self._param_lock:
                if self._pending_param is pending:
                    self._pending_param = None

    def _send_raw_param_command(
            self, command: bytes, timeout_s: float = PARAM_ACK_TIMEOUT_S,
    ) -> UsbRawParamAck | None:
        """Send one compact whole-block command and await its 8-byte ack."""

        with self._state_lock:
            handle = self._serial_handle
            version = self.firmware_version
        if handle is None:
            return None
        if version is None or not supports_pressure_fit_params(version):
            return None
        pending = _PendingRawParamAck(
            subcommand=command[3], event=threading.Event())
        with self._raw_param_lock:
            self._pending_raw_param = pending
        try:
            with self._write_lock:
                handle.write(command)
        except Exception as exc:
            with self._raw_param_lock:
                if self._pending_raw_param is pending:
                    self._pending_raw_param = None
            self._set_status("error", f"raw parameter command write failed: {exc}")
            return None
        try:
            if not pending.event.wait(max(0.0, float(timeout_s))):
                return None
            return pending.result
        finally:
            with self._raw_param_lock:
                if self._pending_raw_param is pending:
                    self._pending_raw_param = None

    def _update_mag_telemetry(
            self, wire_frame, host_timestamp_us: int) -> None:
        """Merge one type-0x05 frame into the per-channel caches."""

        self._mag_cache.update(wire_frame, host_timestamp_us)

    def wait_for_version(self, timeout_s: float = 3.0) -> str | None:
        """Block until a frame has announced the firmware version.

        :attr:`firmware_version` is filled in by the reader thread as soon as
        the first valid frame arrives, which on a just-opened port is a moment
        later.  A caller that gates on the version -- the parameter tooling
        refuses to send 0x51 to firmware that predates it -- must therefore
        wait rather than sample the attribute right after :meth:`start`, or it
        reports "no link" on a perfectly healthy device.

        Returns the version, or ``None`` if nothing arrived within
        ``timeout_s`` (a closed port, a silent device, or a port opened before
        the glove finished booting).  Waiting again is cheap and safe.
        """

        if self.firmware_version is not None:
            return self.firmware_version
        self._version_event.wait(max(0.0, float(timeout_s)))
        return self.firmware_version

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

    def _wait_pressure_fit(
            self, set_index: int, *, baseline, timeout_s: float,
    ) -> UsbPressureFitParams | None:
        """Wait for the reader thread to cache a *new* 0x06 for ``set_index``.

        ``baseline`` is whatever the cache held for this set before the command
        was written, and a new frame always replaces that entry with a fresh
        object, so waiting for a different object is both exact and free of any
        clock comparison.  Timestamping an accepted frame instead would mean
        trusting that the reader's stamp and the caller's share a clock, and
        that a frame already in the OS buffer when the command was sent cannot
        be stamped later than it.
        """

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            snapshot = self._fit_cache.snapshot(set_index)
            if snapshot is not None and snapshot is not baseline:
                return snapshot.params
            if time.monotonic() >= deadline:
                return None
            self.stop_event.wait(0.005)

    def _wait_raw_param_frame(
            self, baseline, *, timeout_s: float,
    ) -> bytes | None:
        """Wait for a new complete type-0x06 raw payload."""

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            with self._raw_param_lock:
                current = self._raw_param_frame
            if current is not None and current[0] is not baseline:
                return current[1]
            if time.monotonic() >= deadline:
                return None
            self.stop_event.wait(0.005)

    def _run_param_command(
            self, command: bytes, *, timeout_s: float,
            attempts: int = PARAM_COMMAND_ATTEMPTS,
    ) -> tuple[UsbParamAck | None, str]:
        """Send one command until the device answers it (spec 5.1).

        Returns ``(ack, problem)``: exactly one of the two is meaningful.  The
        distinction matters downstream -- a silent link and a device that
        answered "套号非法" are different faults, and reporting both as "no
        answer" hides the second one.
        """

        for _ in range(max(1, int(attempts))):
            ack = self._send_param_command(command, timeout_s=timeout_s)
            if ack is None:
                continue
            if ack.ok:
                return ack, ""
            return None, f"device rejected the command: {ack.status_name}"
        return None, f"no answer after {max(1, int(attempts))} attempts"

    def _run_raw_param_command(
            self, command: bytes, *, timeout_s: float,
            attempts: int = PARAM_COMMAND_ATTEMPTS,
    ) -> tuple[UsbRawParamAck | None, str]:
        """Send one compact command until its 8-byte ack arrives."""

        for _ in range(max(1, int(attempts))):
            ack = self._send_raw_param_command(command, timeout_s=timeout_s)
            if ack is None:
                continue
            if ack.ok:
                return ack, ""
            return None, f"device rejected the raw parameter command: {ack.status_name}"
        return None, f"no answer after {max(1, int(attempts))} attempts"

    def _param_version_problem(self) -> str | None:
        """Why the v1.2.16 parameter commands cannot be used, or ``None``.

        Waits for the version announcement first.  ``firmware_version`` is
        filled in by the reader thread as the first frame arrives, so sampling
        the attribute on a just-opened port reports "does not support the
        parameter command (firmware None)" -- wrong for any glove, and actively
        misleading on a v1.2.16 one.  The four public entry points share this
        so the "not announced yet" and "too old" cases cannot drift apart.
        """

        version = self.wait_for_version(PARAM_VERSION_WAIT_S)
        if version is None:
            return "no link: the device has not announced a firmware version"
        if not supports_pressure_fit_params(version):
            return (f"device does not support the parameter command "
                    f"(firmware {version})")
        return None

    def read_pressure_fit_set(
            self, set_index: int, *, timeout_s: float = 2.0,
            attempts: int = PARAM_COMMAND_ATTEMPTS,
    ) -> PressureFitReadOutcome:
        """Ask the device for one stored set (0x02) and return its 0x06.

        ``outcome.ack`` carries the device's whole-set CRC32, which is the only
        checksum available for this content (see
        :class:`PressureFitReadOutcome`).

        The whole command session holds ``_param_command_lock``: two
        overlapping sessions could otherwise have their 0x07 answers swapped,
        since the echo carries no nonce.
        """

        set_index = int(set_index)
        with self._param_command_lock:
            problem = self._param_version_problem()
            if problem is not None:
                return PressureFitReadOutcome(
                    set_index=set_index, ok=False, message=problem)
            baseline = self._fit_cache.snapshot(set_index)
            command = encode_param_command(
                PARAM_SUBCMD_READ_SET, set_index=set_index)
            ack, problem = self._run_param_command(
                command, timeout_s=timeout_s, attempts=attempts)
            if ack is None:
                return PressureFitReadOutcome(
                    set_index=set_index, ok=False, message=problem)
            params = self._wait_pressure_fit(
                set_index, baseline=baseline, timeout_s=timeout_s)
            if params is None:
                return PressureFitReadOutcome(
                    set_index=set_index, ok=False, ack=ack,
                    message=(f"the device acknowledged the read but sent no "
                             f"0x06 for set {set_index} (nothing stored?)"))
            return PressureFitReadOutcome(
                set_index=set_index, ok=True, params=params, ack=ack)

    def query_pressure_fit(
            self, *, timeout_s: float = 5.0,
            attempts: int = PARAM_COMMAND_ATTEMPTS,
    ) -> PressureFitCommandOutcome:
        """Ask for every stored set (0x04): ack, parameters, missing sets.

        The device broadcasts each stored set as a 0x06 and answers with a
        single 0x07 *after* them, so once the ack is in, any broadcast still in
        flight is given a short grace period before the cache is read.  The
        ack's ``valid_mask`` is what says which sets to expect; a set it claims
        is stored but that never arrived is reported in ``missing`` rather than
        silently omitted.

        "Never arrived" is judged against what the cache held *before* the
        query, because a stale entry from an earlier read looks exactly like a
        fresh one.  A write emits no 0x06 at all, so
        :meth:`write_pressure_fit_set` drops the written set's entry; without
        that, the leftover previous contents would be indistinguishable from a
        broadcast answering this query.
        """

        with self._param_command_lock:
            problem = self._param_version_problem()
            if problem is not None:
                return PressureFitCommandOutcome(ok=False, message=problem)
            baseline = {set_index: self._fit_cache.snapshot(set_index)
                        for set_index in range(PRESSURE_FIT_SET_COUNT)}
            command = encode_param_command(PARAM_SUBCMD_QUERY_STATUS)
            ack, problem = self._run_param_command(
                command, timeout_s=timeout_s, attempts=attempts)
            if ack is None:
                return PressureFitCommandOutcome(ok=False, message=problem)
            time.sleep(PARAM_BROADCAST_SLACK_S)

            params: dict[int, UsbPressureFitParams] = {}
            for set_index in range(PRESSURE_FIT_SET_COUNT):
                if not ack.set_valid(set_index):
                    continue
                snapshot = self._fit_cache.snapshot(set_index)
                if snapshot is not None and snapshot is not baseline[set_index]:
                    params[set_index] = snapshot.params
            missing = tuple(
                set_index for set_index in range(PRESSURE_FIT_SET_COUNT)
                if ack.set_valid(set_index) and set_index not in params)
            return PressureFitCommandOutcome(
                ok=True, ack=ack, params=params, missing=missing)

    def read_raw_param_block(
            self, *, timeout_s: float = 5.0,
            attempts: int = PARAM_COMMAND_ATTEMPTS,
    ) -> RawParamReadOutcome:
        """Read the current firmware's opaque parameter block.

        The command is exactly ``A5 5A 51 02 00 00 00 00 00 00``.  The
        firmware first acknowledges it with an 8-byte type-0x07 payload and
        then emits one type-0x06 frame containing the complete raw block.
        """

        with self._param_command_lock:
            problem = self._param_version_problem()
            if problem is not None:
                return RawParamReadOutcome(ok=False, message=problem)
            with self._raw_param_lock:
                baseline = (self._raw_param_frame[0]
                            if self._raw_param_frame is not None else object())
            command = encode_raw_param_command(PARAM_SUBCMD_READ_SET)
            ack, problem = self._run_raw_param_command(
                command, timeout_s=timeout_s, attempts=attempts)
            if ack is None:
                return RawParamReadOutcome(ok=False, message=problem)
            data = self._wait_raw_param_frame(
                baseline, timeout_s=timeout_s)
            if data is None:
                return RawParamReadOutcome(
                    ok=False, ack=ack,
                    message="device acknowledged the read but sent no type-0x06 raw block")
            if ack.length != len(data):
                return RawParamReadOutcome(
                    ok=False, data=data, ack=ack,
                    message=(f"raw block length mismatch: ack says {ack.length} bytes, "
                             f"type-0x06 carries {len(data)} bytes"))
            actual_crc = pressure_fit_crc32(data)
            if ack.crc32 not in (0, actual_crc):
                return RawParamReadOutcome(
                    ok=False, data=data, ack=ack,
                    message=(f"raw block CRC32 mismatch: ack says 0x{ack.crc32:08X}, "
                             f"received data calculates 0x{actual_crc:08X}"))
            return RawParamReadOutcome(ok=True, data=data, ack=ack)

    def write_raw_param_block(
            self, data: bytes, *, timeout_s: float = PARAM_ACK_TIMEOUT_S,
            attempts: int = PARAM_COMMAND_ATTEMPTS,
    ) -> RawParamWriteOutcome:
        """Write one complete raw parameter block with the current protocol."""

        data = bytes(data)
        crc32 = pressure_fit_crc32(data)
        with self._param_command_lock:
            problem = self._param_version_problem()
            if problem is not None:
                return RawParamWriteOutcome(
                    ok=False, length=len(data), crc32=crc32, message=problem)
            command = encode_raw_param_command(
                PARAM_SUBCMD_WRITE_SHARD, data, crc32=crc32)
            ack, problem = self._run_raw_param_command(
                command, timeout_s=timeout_s, attempts=attempts)
            if ack is None:
                return RawParamWriteOutcome(
                    ok=False, length=len(data), crc32=crc32, message=problem)
            if ack.length != len(data) or ack.crc32 != crc32:
                return RawParamWriteOutcome(
                    ok=False, length=len(data), crc32=crc32, ack=ack,
                    message=(f"device ack mismatch: length={ack.length}, "
                             f"crc32=0x{ack.crc32:08X}"))
            return RawParamWriteOutcome(
                ok=True, length=len(data), crc32=crc32, ack=ack)

    def write_pressure_fit_set(
            self, set_index: int, blob: bytes, *,
            timeout_s: float = PARAM_ACK_TIMEOUT_S,
            retries: int = PARAM_SHARD_RETRIES,
            progress=None,
    ) -> PressureFitWriteOutcome:
        """Write one stored set (four shards, spec 5.3).

        ``blob`` is the 1024-byte image (k then b).  Returns the outcome
        rather than raising on a link failure; inspect ``outcome.ok``.

        On success the set's cached entry is dropped: a write emits no 0x06
        (spec 4.7), so the cache would otherwise keep describing the
        coefficients the write just replaced -- including a ``write_index``
        and CRC32 that no longer exist on the device.
        """

        with self._param_command_lock:
            problem = self._param_version_problem()
            if problem is not None:
                return PressureFitWriteOutcome(
                    set_index=int(set_index), ok=False,
                    crc32=pressure_fit_crc32(bytes(blob)),
                    message=problem)
            outcome = write_pressure_fit_shards(
                blob, set_index, self._send_param_command,
                timeout_s=timeout_s, retries=retries, progress=progress)
            if outcome.ok:
                self._fit_cache.forget(set_index)
            return outcome

    def clear_pressure_fit(
            self, *, timeout_s: float = 2.0,
            attempts: int = PARAM_COMMAND_ATTEMPTS,
    ) -> PressureFitCommandOutcome:
        """Erase every stored set (0x03).

        On success the cache is dropped: every set is gone, so the entries it
        holds describe parameters the device no longer has.  Leaving them would
        let the next :meth:`query_pressure_fit` treat them as pre-command
        baselines and report them as freshly broadcast.
        """

        with self._param_command_lock:
            problem = self._param_version_problem()
            if problem is not None:
                return PressureFitCommandOutcome(ok=False, message=problem)
            command = encode_param_command(PARAM_SUBCMD_CLEAR_ALL)
            ack, problem = self._run_param_command(
                command, timeout_s=timeout_s, attempts=attempts)
            if ack is None:
                return PressureFitCommandOutcome(ok=False, message=problem)
            self._fit_cache.reset()
            return PressureFitCommandOutcome(ok=True, ack=ack)

    def start_pressure_fit(
            self, *, timeout_s: float = PARAM_ACK_TIMEOUT_S,
            attempts: int = PARAM_COMMAND_ATTEMPTS,
    ) -> PressureFitCommandOutcome:
        """Ask the device to fit for itself (0x51/0x06), and read nothing back.

        This is a *send*: one 13-byte downlink with no data area, and the single
        0x07 that answers it.  Spec 4.7 gives a device no reason to emit 0x06
        frames unprompted, so any that happen to follow the command are not its
        answer -- they take the ordinary broadcast path into the cache, which is
        where :meth:`query_pressure_fit` reads them.  ``params`` is therefore
        always empty here, and nothing this call does can be mistaken for the
        read-back.

        Silence and a rejection are different faults and are reported
        differently.  A firmware that has not implemented 0x06 is the state
        every glove is in today, so the silent case says so and names the
        subcommand; a device that *did* answer with a non-zero status rejected
        the command, and blaming the firmware for a missing subcommand there
        would point at the wrong side.

        The whole session holds ``_param_command_lock``, like its siblings: the
        0x07 echo carries no nonce, so two overlapping sessions could otherwise
        swap answers.
        """

        with self._param_command_lock:
            problem = self._param_version_problem()
            if problem is not None:
                return PressureFitCommandOutcome(ok=False, message=problem)
            command = encode_param_command(PARAM_SUBCMD_START_FIT)
            ack, problem = self._run_param_command(
                command, timeout_s=timeout_s, attempts=attempts)
            if ack is None:
                # ``_run_param_command`` folds both faults into one string, so
                # the rejection -- the one case with a culprit the device named
                # -- is recognised by its own prefix and passed through as-is.
                if problem.startswith("device rejected"):
                    return PressureFitCommandOutcome(ok=False, message=problem)
                return PressureFitCommandOutcome(
                    ok=False,
                    message=(f"no answer after {max(1, int(attempts))} attempts; "
                             f"the firmware may not implement subcommand "
                             f"{PARAM_SUBCMD_START_FIT:#04x} yet"))
            return PressureFitCommandOutcome(ok=True, ack=ack)

    def _run(self) -> None:
        try:
            import serial
        except ImportError as exc:
            self._set_status("error", "pyserial is not installed")
            return

        parser = UsbCdcFrameParser(link_kind=self._link_kind)
        serial_handle = None
        last_valid_frame_s = time.monotonic()
        next_recovery_s = 0.0
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
                        port, baudrate=115200, timeout=0.2,
                        write_timeout=SERIAL_WRITE_TIMEOUT_S)
                    parser.reset()
                    self._matrix_assembler.reset()
                    self._drop_counter.reset()
                    opened_s = time.monotonic()
                    last_valid_frame_s = opened_s
                    with self._state_lock:
                        self.active_port = str(port)
                        self.connected = True
                        self._serial_handle = serial_handle
                        self._mag_cache.reset()
                        # Same reason as the magnetometer cache above: a reopen
                        # may be a different glove, and set numbers and write
                        # indices would look identical, so stale parameters
                        # would be hard to notice.
                        self._fit_cache.reset()
                        # And the version, for the same reason: a reopen may be
                        # a different glove, and the version gates every
                        # optional command (0x50 probe, 0x51 parameters).
                        # Re-arm the announcement so a waiter gets the new
                        # device's version rather than the old one.
                        self.firmware_version = None
                        self._version_event.clear()
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
                # Block for one byte when idle, then immediately drain whatever
                # the driver already has.  The frame parser accepts partial
                # packets, so this preserves throughput while removing the
                # host-side batching delay.  (Reading with ``read(4096)``
                # instead waits for a nearly full 4 KiB block on Windows, which
                # on a wired link batching 6-8 IMU samples per read is what
                # used to turn an 80 Hz device stream into about 10 Hz of
                # solved poses.)
                #
                # What arrives is still a *batch*, not a single frame: a
                # Bluetooth dongle hands over one batch per RF event, so a
                # read here can carry several frames that the device produced
                # tens of milliseconds apart.  Trimming that batch to keep only
                # its newest frame discards device frames the link delivered
                # perfectly well, which is a host-side loss, not a link loss --
                # ``link_drops`` below is what tells the two apart.
                data = _read_available(serial_handle)
            except (OSError, serial.SerialException) as exc:
                self._set_status("disconnected", f"serial read failed: {exc}")
                with self._state_lock:
                    self.connected = False
                    self._serial_handle = None
                self._fail_pending_pings()
                self._fail_pending_param()
                try:
                    serial_handle.close()
                except Exception:
                    pass
                serial_handle = None
                self.stop_event.wait(0.2)
                continue

            # Time-driven, so it still fires on a link that is idle right now.
            self._maybe_probe_latency()

            wire_frames = parser.feed(data) if data else []
            if wire_frames:
                last_valid_frame_s = time.monotonic()
            for wire_frame in wire_frames:
                host_timestamp_us = time.time_ns() // 1_000
                with self._state_lock:
                    self.firmware_version = wire_frame.version
                    self._version_event.set()
                # Every frame type draws from one device-wide counter, so the
                # gap is charged against all of them, not just the IMU ones.
                self._drop_counter.observe(
                    wire_frame.sequence,
                    host_timestamp_us,
                    sequence_mask(
                        wire_frame.header_size, wire_frame.version),
                )
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
                        # The 0x03 stream is three tiers on one type on a
                        # Bluetooth link: a full picture, and the sparse/bitmap
                        # patches that carry only what moved since the last one.
                        # The assembler holds the running picture and hands back
                        # a complete one for every tier, so what reaches the
                        # queue is always what the device's scan actually read --
                        # never a patch wearing the shape of a picture.  It
                        # returns ``None`` for a patch that arrived before the
                        # first keyframe and for a payload that fits no tier,
                        # and both are dropped here rather than published as a
                        # partial picture.
                        #
                        # On a wired link there is no patch to hold: the
                        # assembler was built with this transport's link kind, so
                        # every frame here is the whole picture (v1.2.16 4.4) and
                        # the parser has already refused anything that is not
                        # one.
                        matrix = self._matrix_assembler.update(
                            wire_frame.payload,
                            sequence=wire_frame.sequence,
                            scan_time_us=wire_frame.timestamp_us,
                        )
                        if matrix is not None:
                            _put_latest(self.tactile_queue, TactileFrame(
                                sequence=int(matrix.sequence),
                                timestamp_us=int(matrix.scan_time_us),
                                samples=matrix.samples,
                                processed=False,
                                # The first hop the latency probe measures is
                                # the wait *here*, so the stamp is taken as
                                # close to the queue as it can be -- before the
                                # frame is in it, not after.  ``0.0`` when the
                                # probe is off, which ``record`` skips.
                                host_enqueue_us=_latency_stamp(),
                            ))
                    elif wire_frame.message_type == USB_TYPE_DELAY_RESPONSE:
                        self._resolve_ping(wire_frame.payload)
                    elif wire_frame.message_type == USB_TYPE_MAG_TELEMETRY:
                        self._update_mag_telemetry(
                            wire_frame, host_timestamp_us)
                    elif wire_frame.message_type == USB_TYPE_PRESSURE_FIT_PARAMS:
                        self._update_pressure_fit(
                            wire_frame, host_timestamp_us)
                    elif wire_frame.message_type == USB_TYPE_PARAM_ACK:
                        self._resolve_param_ack(wire_frame.payload)
                    elif wire_frame.message_type == USB_TYPE_STATUS:
                        status = wire_frame.payload.decode(
                            "utf-8", errors="replace").strip()
                        if status:
                            self._set_status(f"device: {status}", None)
                except ValueError as exc:
                    self._set_status("error", f"invalid USB payload: {exc}")
            self._sync_link_errors(parser)

            # Recover only the damaged Bluetooth side.  Closing this handle
            # has no effect on the other hand because every hand owns a separate
            # transport (and, in the bimanual viewer, a separate process).
            now_s = time.monotonic()
            recovery_reason = None
            if self._is_bluetooth and now_s >= next_recovery_s:
                if now_s - last_valid_frame_s >= BT_VALID_FRAME_SILENCE_S:
                    recovery_reason = (
                        f"no valid frame for {now_s - last_valid_frame_s:.1f}s")

            if recovery_reason is not None:
                self._recovery_count += 1
                self._sync_link_errors(parser)
                self._set_status(
                    "recovering",
                    f"Bluetooth auto-recovery #{self._recovery_count}: "
                    f"{recovery_reason}")
                # Do not send AT+REBOOT automatically.  On some BP101Y units
                # that command removes the USB device without re-enumerating it
                # until a physical unplug/replug.  Reopen only this COM handle;
                # the explicit toolbar action remains available when a person
                # deliberately wants a hardware reboot.
                with self._state_lock:
                    self.connected = False
                    self._serial_handle = None
                self._fail_pending_pings()
                self._fail_pending_param()
                try:
                    serial_handle.close()
                except Exception:
                    pass
                serial_handle = None
                parser.reset()
                next_recovery_s = now_s + BT_RECOVERY_COOLDOWN_S
                self.stop_event.wait(BT_REOPEN_DELAY_S)
                continue

        if serial_handle is not None:
            try:
                serial_handle.close()
            except Exception:
                pass
        with self._state_lock:
            self.connected = False
            self._serial_handle = None
        self._fail_pending_pings()
        self._fail_pending_param()


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

    def wait_for_version(self, timeout_s: float = 3.0) -> str | None:
        """Block until the device announces its firmware version.

        Use this before deciding anything from :attr:`firmware_version`: right
        after :meth:`start` the reader thread has not parsed a frame yet, so
        the attribute is still ``None`` on a healthy link.
        """

        return self._transport.wait_for_version(timeout_s)

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

    @property
    def mag_telemetry(self) -> MagTelemetrySnapshot | None:
        """Latest independently cached v1.2.13 magnetic telemetry."""

        return self._transport.mag_telemetry

    @property
    def link_errors(self) -> dict[str, int]:
        """Cumulative frame-parser error tally for this link.

        ``crc_errors`` counts frames whose CRC rejected the candidate layout,
        ``length_errors`` frames whose declared length was impossible, and
        ``discarded_bytes`` bytes dropped while hunting for the next magic.
        ``recoveries`` counts automatic Bluetooth COM reopens triggered by two
        seconds without a valid frame.
        Cumulative since the transport started, so a caller wanting a rate keeps
        the previous triple and subtracts.  A steady stream at zero is normal;
        a rising ``discarded_bytes`` means the parser, not the transport, is
        eating the wire.
        """

        return self._transport.link_errors

    @property
    def link_drops(self) -> dict[str, int]:
        """Cumulative sequence-gap loss for this link.

        ``frames`` counts every frame the host received -- all five frame types,
        since they share one device-wide counter -- and ``dropped`` the frames
        the device sent that never reached the host, so
        ``dropped / (frames + dropped)`` is the share of the device's own stream
        lost in the link.  Cumulative since the transport started; a caller
        wanting a rate keeps the previous pair and subtracts.

        This is the counterweight to :attr:`link_errors`: those count bytes the
        parser rejected, this counts frames the link never delivered.  Neither
        rises when a consumer deliberately discards frames it received, which is
        why the two together say where frames are going.
        """

        return self._transport.link_drops

    def ping(self, timeout_s: float = 1.0) -> UsbLinkLatency | None:
        """Measure the host<->device round trip (firmware v1.2.11+).

        The same call works on the wired USB CDC link and on a Bluetooth dongle
        port, since a dongle relays the SPP stream byte for byte.  Returns
        ``None`` when the link is down, when the device has not announced
        v1.2.11+, or when no answer arrives in time.
        """

        self.start()
        return self._transport.ping(timeout_s)

    @property
    def supports_pressure_fit(self) -> bool:
        """Whether the connected device speaks the v1.2.16 parameter commands.

        ``False`` before the first frame announces a version, and permanently
        ``False`` on older firmware.  The parameter block itself is opaque to
        the transport: current firmware accepts at most 780 bytes and the
        transport only moves the bytes and checks their CRC-32.
        """

        version = self._transport.firmware_version
        return version is not None and supports_pressure_fit_params(version)

    @property
    def pressure_fit_snapshots(self) -> dict[int, PressureFitSnapshot]:
        """Latest cached 0x06 per set index, as the reader thread stored them."""

        return self._transport.pressure_fit_snapshots

    def pressure_fit_snapshot(self, set_index: int) -> PressureFitSnapshot | None:
        """Latest cached 0x06 for one set index, or ``None`` if never seen.

        This reads the cache only -- it does not ask the device.  Use
        :meth:`read_pressure_fit` for that.
        """

        return self._transport.pressure_fit_snapshot(set_index)

    def read_raw_param_block(
            self, *, timeout_s: float = 5.0,
    ) -> RawParamReadOutcome:
        """Read the current firmware's complete opaque parameter block."""

        self.start()
        return self._transport.read_raw_param_block(timeout_s=timeout_s)

    def write_raw_param_block(
            self, data: bytes, *, timeout_s: float = PARAM_ACK_TIMEOUT_S,
    ) -> RawParamWriteOutcome:
        """Write one complete raw parameter block using ``0x51/0x01``."""

        self.start()
        return self._transport.write_raw_param_block(
            data, timeout_s=timeout_s)

    def read_pressure_fit(
            self, set_index: int, *, timeout_s: float = 2.0,
    ) -> PressureFitReadOutcome:
        """Ask the device for one stored set and return the outcome.

        ``outcome.ok`` is ``False`` for a link failure, firmware that predates
        v1.2.16, a device rejection and a silent read alike; ``outcome.message``
        says which, and ``outcome.ack`` carries the device's CRC32.
        """

        self.start()
        return self._transport.read_pressure_fit_set(
            set_index, timeout_s=timeout_s)

    def query_pressure_fit(
            self, *, timeout_s: float = 5.0,
    ) -> PressureFitCommandOutcome:
        """Read every stored set: the outcome carries parameters and missing.

        A single-connection command, so it holds the parameter command channel
        for its whole duration; a concurrent :meth:`write_pressure_fit` on the
        same stream serialises behind it rather than interleaving.
        """

        self.start()
        return self._transport.query_pressure_fit(timeout_s=timeout_s)

    def write_pressure_fit(
            self, set_index: int, blob: bytes, *,
            timeout_s: float = PARAM_ACK_TIMEOUT_S,
            retries: int = PARAM_SHARD_RETRIES,
            progress=None,
    ) -> PressureFitWriteOutcome:
        """Write one opaque 1024-byte set and verify it with a read-back.

        ``blob`` is the k+b image; the host checks its CRC-32 but not its
        meaning.  Returns the outcome instead of raising, so a link failure
        mid-write is reported rather than thrown.
        """

        self.start()
        return self._transport.write_pressure_fit_set(
            set_index, blob, timeout_s=timeout_s, retries=retries,
            progress=progress)

    def clear_pressure_fit(
            self, *, timeout_s: float = 2.0) -> PressureFitCommandOutcome:
        """Erase every stored parameter set.  Check ``outcome.ok``."""

        self.start()
        return self._transport.clear_pressure_fit(timeout_s=timeout_s)

    def start_pressure_fit(
            self, *, timeout_s: float = PARAM_ACK_TIMEOUT_S,
            attempts: int = PARAM_COMMAND_ATTEMPTS,
    ) -> PressureFitCommandOutcome:
        """Tell the device to fit for itself (0x06); read nothing back.

        ``outcome.ack`` is the whole answer.  Nothing is cached and nothing is
        returned in ``params``: a send that collected the 0x06 frames it
        happened to see would be doing a read-back on the send path, which is
        the opposite of what this command is for.  Read sets with
        :meth:`query_pressure_fit`, which is where the write-back lives.
        """

        self.start()
        return self._transport.start_pressure_fit(
            timeout_s=timeout_s, attempts=attempts)


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


__all__ = [
    "MagTelemetrySnapshot", "PressureFitCommandOutcome", "PressureFitReadOutcome",
    "PressureFitSnapshot", "PressureFitWriteOutcome",
    "RawParamReadOutcome", "RawParamWriteOutcome",
    "RawImuStream", "SensorStream", "TactileStream",
    "write_pressure_fit_shards",
]
