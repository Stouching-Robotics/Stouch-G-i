"""Plaintext single-glove orchestration over public SDK interfaces.

This module owns threading, synchronization, and display smoothing.
The protected hand solver remains reachable only through ``HandSolver``.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import logging
import math
import copy
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Full, Queue
import sys
import threading
import time
from typing import Iterator
from uuid import uuid4

import numpy as np

from common.errors import StreamClosedError, StreamTimeoutError
from common.frame_pacing import BURST_SPAN_US, IMU_POSE_HOLD_S, recent_batch
from sdk import get_version as sdk_version
from glove_io.streams import RawImuStream, TactileStream
from sdk.solver import open_pose_processor
from sdk.mag import from_snapshot, mag_available, MagCalibrationDropMonitor
from sdk.pressure_fit import PARAM_VERSION_WAIT_S
from sdk.transport import LinkSpec, check_frame_side, make_raw_stream, resolve_link
from sdk.types import HandFrame
from sdk.force import fit_finger_force_matrix
from sdk.force_codec import decode_force_parameters
from common.types import (
    LINK_AUTO, DeviceHealth, GloveConfig, GloveInfo, RawImuFrame,
    TactileFrame)


def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

SDK_ROOT = (Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False)
            else _sdk_root(Path(__file__).resolve().parent))
DEFAULT_REGISTRY = SDK_ROOT / "config" / "glove_devices.json"

# How long a solve loop waits for the next frame before giving control back.
# A Bluetooth dongle hands its frames over one RF event at a time, so the window
# has to cover a dongle's own batch latency -- the same span
# :data:`common.frame_pacing.BURST_SPAN_US` uses, and the boundary the loop
# treats as a timestamp jump rather than an advance.
_FRAME_WAIT_S = BURST_SPAN_US / 1_000_000.0

# How many assembled frames :meth:`GloveEngine.read` may hold for a caller that
# is not keeping up.
#
# ``read`` is FIFO and a full queue evicts its oldest frame, so this depth *is*
# the latency of a saturated link.  At 128 frames of 60 Hz that was 2.13 s, and
# a consumer below 60 Hz -- a 50 Hz render loop is enough -- settled with the
# queue permanently full and every ``read`` returning a two-second-old pose.
# Measured against a real glove: frame age climbed 468 ms -> 1415 ms -> 2132 ms
# as the queue filled 56 -> 113 -> 128, then held there; a consumer that kept up
# saw 9.6 ms median.  Nothing was wrong with the link or the solve -- the whole
# two seconds was this queue.
#
# The depth has to stay well inside ``common.frame_pacing.BURST_SPAN_US`` (250 ms
# = 15 frames), the window the rest of the system already uses to decide what is
# "current".  Deeper than that and the SDK hands back exactly the history every
# consumer's staleness guard discards -- which is what the GUI viewers were each
# working around with their own drain loops.  8 frames is 133 ms at 60 Hz: room
# for a rendering hiccup and half the window, and the guard window still fits a
# Bluetooth dongle's burst on top of it.
#
# Dropping here costs nothing that is wanted: the frames evicted are ones the
# caller had already failed to keep up with, and recording does not go through
# this path at all -- :meth:`GloveEngine._imu_loop` appends to the recorder
# before the queue, so every frame the device sent is still written.
DEFAULT_QUEUE_FRAMES = 8


class _JointSmoother:
    def __init__(self, tau_s: float):
        self.tau_s = max(0.0, float(tau_s))
        self.value: np.ndarray | None = None
        self.timestamp_s: float | None = None

    def update(self, joints: np.ndarray, timestamp_s: float) -> np.ndarray:
        value = np.asarray(joints, dtype=np.float32).reshape(21, 3)
        if self.value is None or self.timestamp_s is None:
            self.value = value.copy()
        else:
            dt = max(float(timestamp_s) - self.timestamp_s, 1e-6)
            alpha = (1.0 if self.tau_s == 0.0
                     else 1.0 - math.exp(-dt / self.tau_s))
            self.value += np.float32(alpha) * (value - self.value)
        self.timestamp_s = float(timestamp_s)
        return self.value.copy()


def _windowed_fps(entries: deque[tuple[float, int]],
                  count: int, now: float) -> float:
    """Record one arrival in ``entries`` and return the one-second rate.

    A sliding window over host-side arrival *counts*, matching what the GUI used
    to compute for itself: a Bluetooth dongle delivers a whole batch per RF
    event, so the device clock advances in steps a rate cannot be read off,
    while arrival counts stay honest.
    """

    entries.append((now, count))
    while len(entries) > 2 and now - entries[0][0] > 1.0:
        entries.popleft()
    span = entries[-1][0] - entries[0][0]
    if len(entries) < 2 or span <= 0.0:
        return 0.0
    return (entries[-1][1] - entries[0][1]) / span


@dataclass(frozen=True)
class ForceCalibrationResult:
    ok: bool
    calibration: dict | None = None
    data: bytes | None = None
    error: str = ""
    status: str = "unread"


class _PoseQueue(Queue[HandFrame | None]):
    ended = False
    error: Exception | None = None

    def finish(self, error: Exception | None = None) -> None:
        """Wake readers once. The engine lock serializes queue termination."""
        if self.ended:
            return
        self.ended = True
        self.error = error
        while True:
            try:
                self.get_nowait()
            except Empty:
                break
        self.put_nowait(None)


class GloveEngine:
    def __init__(self, config: GloveConfig,
                 queue_size: int = DEFAULT_QUEUE_FRAMES):
        self.config = config
        self.solver = open_pose_processor(config.side, config.calibration)
        self._resolved_port: str | None = None
        self.serial_number: str = str(config.serial_number or "")
        self._link: LinkSpec | None = None
        self._stream: RawImuStream | None = None
        self._queue = _PoseQueue(maxsize=max(2, int(queue_size)))
        self._lifecycle_lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._latest: HandFrame | None = None
        self._latest_tactile_raw: np.ndarray | None = None
        self._tactile_connection_generation = None
        self._tactile_connection_id = None
        # Bumped once per cached pressure scan -- one per poll, the newest the
        # transport had queued (see :meth:`_tactile_scan_frames`) -- and copied
        # onto each assembled HandFrame as ``tactile_fresh``.  Attitude and
        # pressure are separate device streams at the same nominal rate, so the
        # cached matrix is re-attached to several attitude frames running; a
        # caller that filters over time has to be able to see the difference
        # between a new scan and the same one again.  Counting cache replacements
        # rather than frames received keeps that answer the same: scans a later
        # one of the same poll overwrote were never observable.
        self._tactile_generation = 0
        self._tactile_emitted = 0
        self._lock = threading.RLock()
        self._missing = tuple(f"IMU-{index}" for index in range(16))
        self._frame_times: deque[float] = deque(maxlen=120)
        # (host_monotonic_s, cumulative_frames_received) over a one-second
        # window, which is what :attr:`device_fps` divides.  Counting arrivals
        # rather than reading device timestamps keeps it honest across a
        # Bluetooth batch handover, where the device clock jumps by a batch.
        self._arrivals: deque[tuple[float, int]] = deque()
        self._arrival_count = 0
        # The same window over pressure scans instead of attitude frames, which
        # is what :attr:`HandFrame.tactile_fps` reports.  Counted where a scan is
        # cached rather than read off the device clock, for the reason above: a
        # Bluetooth batch hands over several scans per RF event.
        self._tactile_arrivals: deque[tuple[float, int]] = deque()
        self._tactile_arrival_count = 0
        self._tactile_fps = 0.0
        self._smoother = _JointSmoother(config.smoothing_ms / 1000.0)
        self._started = False
        self._last_error: str | None = None
        self._last_tactile_error: str | None = None
        self._force_revision = 0
        self._force_session_id = uuid4().hex
        self._force_connection_key = None
        self._force_connection_id = None
        self._force_baseline = None
        self._force_baseline_id = None
        self._force_baseline_revision = 0
        self._force_generation = None
        self._force_state = {"source": "unavailable", "status": "unread", "error": "", "calibration": None}

    def start(self) -> "GloveEngine":
        with self._lifecycle_lock:
            if self._started:
                return self
            if self._stream is not None:
                self._stop_stream()
            try:
                self._start_stream()
            except BaseException:
                try:
                    self._stop_stream()
                except Exception:
                    logging.getLogger(__name__).exception("Glove startup rollback failed")
                raise
            return self

    def _start_stream(self) -> None:
        with self._lock:
            self._queue = _PoseQueue(maxsize=self._queue.maxsize)
            self._force_session_id = uuid4().hex
            self._force_connection_key = None
            self._force_connection_id = None
            self._clear_force_baseline()
            self._invalidate_force("unread")
            # A new transport can reuse generation 1; no old scan belongs to it.
            self._latest_tactile_raw = None
            self._tactile_connection_generation = None
            self._tactile_connection_id = None
            self._tactile_generation = 0
            self._tactile_emitted = 0
            self._tactile_arrivals.clear()
            self._tactile_arrival_count = 0
            self._tactile_fps = 0.0
            self._last_error = None
            self._last_tactile_error = None
        self._link = resolve_link(self.config)
        self._resolved_port = self._link.port
        self.serial_number = self._link.serial_number
        self._stream = make_raw_stream(self._link)
        self._mag_drop_monitor = MagCalibrationDropMonitor()
        self._stop.clear()
        self._stream.start()
        workers = [
            threading.Thread(target=self._imu_loop, name=f"sdk-{self.config.side}-imu",
                             daemon=True),
        ]
        if self.config.include_tactile:
            workers.append(threading.Thread(
                target=self._tactile_loop,
                name=f"sdk-{self.config.side}-tactile", daemon=True))
        with self._lock:
            self._threads = workers
            for thread in workers:
                thread.start()
            self._started = True

    def _invalidate_force(self, status="unread", error=""):
        with self._lock:
            self._force_revision += 1
            self._force_generation = None
            self._force_state = {"source": "unavailable", "status": status, "error": error, "calibration": None}

    def _clear_force_baseline(self):
        """Clear local baseline output state; caller holds _lock."""
        self._force_baseline = None
        self._force_baseline_id = None
        self._force_baseline_revision += 1

    def _sync_force_connection(self):
        """Track live identity even while coefficients are absent or reading."""
        stream = self._stream
        transport = None if stream is None else stream._transport
        key = ((transport, transport.connection_generation)
               if self._started and not self._stop.is_set()
               and transport is not None and transport.connected else None)
        previous = self._force_connection_key
        if (key is None and previous is None) or (
                key is not None and previous is not None
                and key[0] is previous[0] and key[1] == previous[1]):
            return
        self._force_connection_key = key
        self._force_connection_id = (None if key is None else
                                     f"{self._force_session_id}:{uuid4().hex}:{key[1]}")
        self._clear_force_baseline()
        if previous is not None:
            self._invalidate_force("disconnected", "connection or parameters changed during read")

    def _force_snapshot(self):
        with self._lock:
            self._sync_force_connection()
            state = copy.deepcopy(self._force_state)
            state.update(connection_id=self._force_connection_id,
                         baseline_id=self._force_baseline_id,
                         baseline_ready=self._force_baseline is not None)
            return state

    @property
    def force_calibration_status(self):
        return self._force_snapshot()

    def set_force_baseline(self, baseline, *, connection_id=None, baseline_id=None) -> str | None:
        """Apply a measured local baseline to this live connection, without I/O."""
        with self._lock:
            self._sync_force_connection()
            if self._force_connection_id is None:
                raise StreamClosedError("force baseline requires a started connected engine")
            if not self.config.include_tactile:
                raise ValueError("force baseline requires tactile input")
            if connection_id is not None and connection_id != self._force_connection_id:
                raise ValueError("force baseline connection_id is stale")
            if baseline is None:
                self._clear_force_baseline()
                return None
            measured = np.asarray(baseline, dtype=np.float64)
            if (measured.shape != (16, 16) or not np.all(np.isfinite(measured))
                    or np.any(measured < 0) or np.any(measured > 4095)):
                raise ValueError("baseline must be a finite 16x16 ADC matrix in 0..4095")
            if baseline_id is not None and not isinstance(baseline_id, str):
                raise ValueError("baseline_id must be a string")
            self._force_baseline = measured.astype(np.float32, copy=True)
            self._force_baseline_id = uuid4().hex if baseline_id is None else baseline_id
            self._force_baseline_revision += 1
            return self._force_baseline_id

    def read_force_calibration(self, *, timeout_s=5.0):
        with self._lock:
            self._sync_force_connection()
            stream = self._stream
            if not self._started or stream is None or self._stop.is_set():
                return ForceCalibrationResult(False, error="engine is not started", status="disconnected")
            if not self.config.include_tactile:
                self._invalidate_force("no_tactile", "tactile is disabled")
                return ForceCalibrationResult(False, error="tactile is disabled", status="no_tactile")
            transport = stream._transport
            if not transport.connected:
                self._invalidate_force("disconnected", "transport is disconnected")
                return ForceCalibrationResult(False, error="transport is disconnected", status="disconnected")
            generation = transport.connection_generation
            connection_id = self._force_connection_id
            self._invalidate_force("reading")
            revision = self._force_revision
        try:
            raw = transport.read_force_raw(timeout_s=max(0.0, float(timeout_s)), generation=generation)
            data = raw.data
            if not raw.ok:
                result = ForceCalibrationResult(False, data=data, error=raw.message, status="timeout" if "answer" in raw.message or "sent no" in raw.message else "invalid")
            elif not data or all(b == 0xff for b in data):
                result = ForceCalibrationResult(False, data=data, error="parameter block is empty", status="empty")
            else:
                calibration = decode_force_parameters(data, expected_side=self.config.side)
                result = ForceCalibrationResult(True, calibration=calibration, data=data, status="ready")
        except Exception as exc:
            result = ForceCalibrationResult(False, data=locals().get("data"), error=str(exc), status="invalid")
        with self._lock:
            self._sync_force_connection()
            if (revision != self._force_revision or not self._started or self._stream is not stream
                    or connection_id != self._force_connection_id or not transport.connected
                    or transport.connection_generation != generation):
                error = "connection or parameters changed during read"
                # A completed old request must not overwrite a newer read or close.
                if revision == self._force_revision:
                    self._invalidate_force("disconnected", error)
                return ForceCalibrationResult(False, data=result.data, error=error, status="disconnected")
            self._force_generation = generation
            self._force_state = {"source": "device" if result.ok else "unavailable", "status": result.status, "error": result.error, "calibration": copy.deepcopy(result.calibration)}
        return result

    def _put_latest(self, frame: HandFrame) -> None:
        while True:
            try:
                self._queue.put_nowait(frame)
                return
            except Full:
                try:
                    self._queue.get_nowait()
                except Empty:
                    return

    def _note_arrival(self, now: float) -> float:
        """Record one received attitude frame and return its arrival rate."""

        self._arrival_count += 1
        return _windowed_fps(self._arrivals, self._arrival_count, now)

    def _note_tactile_scan(self, now: float) -> float:
        """Record one cached pressure scan and return its arrival rate.

        The twin of :meth:`_note_arrival` on the pressure stream.  The panel's
        own refresh rate is a separate number from the attitude rate, and a
        caller that shows both must not let one stand in for the other: a hand
        whose IMU settles would otherwise take its pressure readout with it.
        """

        self._tactile_arrival_count += 1
        return _windowed_fps(
            self._tactile_arrivals, self._tactile_arrival_count, now)

    @property
    def _firmware_version(self) -> str | None:
        """The glove's own version, learned when its 0x01 status frame lands.

        Read through rather than cached: 0x01 arrives once, shortly after the
        port opens, so a value latched in :meth:`start` would be permanently
        ``None``.
        """

        return None if self._stream is None else self._stream.firmware_version

    def _latency_ms(self) -> float | None:
        """Host<->device round trip, or ``None`` when the link cannot measure it."""

        latency = None if self._stream is None else self._stream.latency
        return None if latency is None else latency.rtt_ms

    def _take_tactile_freshness(self) -> bool:
        """Was a new pressure scan cached since the last frame was assembled?

        Answers once per call and remembers that it did, so consecutive frames
        built while one scan is still the newest report ``False`` after the
        first.  Callers filtering pressure over time gate on this: re-feeding a
        repeat shortens the filter's time constant and finishes its warm-up
        early.  Takes ``_lock`` itself, which is reentrant, so it can be called
        from inside a block that already holds it.
        """

        with self._lock:
            fresh = self._tactile_generation != self._tactile_emitted
            self._tactile_emitted = self._tactile_generation
            return fresh

    def _imu_batch_frames(self, timeout: float) -> Iterator[RawImuFrame]:
        """Yield every frame worth solving, one poll batch at a time.

        The transport keeps a **window of the frames it received** and drops the
        oldest once that window is full (``buffer_size``, 256 frames by default),
        so the queue holds the newest frames -- but ``read`` hands them over one
        at a time starting from the oldest end.  A solver even slightly slower
        than the device therefore never catches up: it settles at a permanent lag
        equal to the whole window, which at 60 Hz is 4.3 s.  Every link
        diagnostic stays clean the entire time, because nothing is wrong with the
        link -- which is exactly why this reads as "seconds of delay on the
        screen while the HUD says 1 ms".

        Draining to the newest frame alone fixes that lag and breaks something
        else: a Bluetooth dongle hands over one *batch* per RF event, so keeping
        only the newest makes the solve rate equal the *batch* rate -- a silent
        integer-fold loss of real samples, which reads exactly like a slow glove.
        :func:`common.frame_pacing.recent_batch` is the policy that survives both
        shapes: keep the whole batch, trim only what is far enough behind the
        newest to be history rather than a burst.  Every frame yielded here is
        solved, so a burst costs its own duration once instead of costing the
        stream its rate forever.

        The blocking ``read`` comes first and the non-blocking ``poll`` second on
        purpose: ``poll`` never blocks, so a bare poll loop would spin at 100% of
        a core while the glove is quiet.
        """

        assert self._stream is not None
        while not self._stop.is_set():
            try:
                first = self._stream.read(timeout=timeout)
            except StreamTimeoutError:
                continue
            yield from recent_batch([first, *self._stream.poll()])

    def _imu_loop(self) -> None:
        try:
            assert self._stream is not None and self._link is not None
            generation = None
            for imu_frame in self._imu_batch_frames(_FRAME_WAIT_S):
                packet_generation = getattr(imu_frame, "_connection_generation", None)
                with self._lock, self._transport_state_lock():
                    if self._stop.is_set():
                        break
                    if not self._generation_is_current(packet_generation):
                        continue
                    firmware = getattr(imu_frame, "_wire_firmware_version", self._firmware_version)
                    check_frame_side(self.config.side, self._link, firmware)
                if packet_generation != generation:
                    generation = packet_generation
                    self._smoother = _JointSmoother(self.config.smoothing_ms / 1000.0)
                    self._arrivals.clear()
                    self._arrival_count = 0
                    self._mag_drop_monitor = MagCalibrationDropMonitor()
                keypoints = self.solver.process(imu_frame)
                timestamp_s = keypoints.timestamp_us / 1_000_000.0
                smoothed = self._smoother.update(
                    keypoints.joints_m, timestamp_s)
                now = time.monotonic()
                device_fps = self._note_arrival(now)
                with self._lock, self._transport_state_lock():
                    if self._stop.is_set() or not self._generation_is_current(packet_generation):
                        continue
                    force_state = self._force_snapshot()
                    current_generation = self._stream._transport.connection_generation
                    tactile_raw = (None if self._latest_tactile_raw is None
                                   or not self._stream.connected
                                   or self._tactile_connection_generation != current_generation
                                   or self._tactile_connection_id != force_state["connection_id"]
                                   else self._latest_tactile_raw.copy())
                    tactile_fresh = self._take_tactile_freshness()
                    tactile_fps = self._tactile_fps
                    force_revision = self._force_revision
                    baseline_revision = self._force_baseline_revision
                    baseline = (None if self._force_baseline is None
                                else self._force_baseline.copy())
                force = None
                force_status = force_state["status"]
                force_error = force_state["error"]
                if force_state["calibration"] is not None:
                    if force_state["calibration"]["model"] == "delta_adc" and baseline is None:
                        force_status = "no_baseline"
                        force_error = "measured force baseline is unavailable"
                    elif tactile_raw is None:
                        force_status = "no_tactile"
                    else:
                        force = fit_finger_force_matrix(
                            tactile_raw, self.config.side, force_state["calibration"], baseline=baseline)
                frame = HandFrame(
                    force_n=force,
                    force_source=force_state["source"],
                    force_status=force_status,
                    force_error=force_error,
                    force_fresh=bool(force is not None and tactile_fresh),
                    side=self.config.side,
                    sequence=keypoints.sequence,
                    timestamp_us=keypoints.timestamp_us,
                    imu_xyzw=keypoints.imu_xyzw,
                    imu_valid=keypoints.valid_mask,
                    sensor_age_s=keypoints.sensor_age_s,
                    joints_raw_m=keypoints.joints_m,
                    joint_xyzw=keypoints.joint_xyzw,
                    joints_smoothed_m=smoothed,
                    tactile_raw=tactile_raw,
                    tactile_fresh=tactile_fresh,
                    status=keypoints.status,
                    raw_imu_xyzw=imu_frame.quaternions_xyzw,
                    raw_imu_present=imu_frame.present_mask,
                    raw_imu_valid=imu_frame.valid_mask,
                    raw_device_timestamp_us=imu_frame.device_timestamp_us,
                    firmware_version=firmware,
                    link_kind=self._link.kind,
                    mag=from_snapshot(self._stream.mag_telemetry),
                    device_fps=device_fps,
                    tactile_fps=tactile_fps,
                    link_drops=dict(self._stream.link_drops),
                    link_errors=dict(self._stream.link_errors),
                    latency_ms=self._latency_ms(),
                )
                object.__setattr__(frame, "_force_revision", force_revision)
                object.__setattr__(frame, "_force_connection_id", force_state["connection_id"])
                object.__setattr__(frame, "_force_baseline_id", force_state["baseline_id"])
                object.__setattr__(frame, "_force_baseline_revision", baseline_revision)
                object.__setattr__(frame, "_connection_generation", packet_generation)
                with self._lock, self._transport_state_lock():
                    if self._stop.is_set() or not self._generation_is_current(packet_generation):
                        continue
                    if getattr(self._latest, "_connection_generation", None) != packet_generation:
                        self._frame_times.clear()
                    missing_mask = ~(keypoints.sensor_age_s <= IMU_POSE_HOLD_S + 1e-9)
                    self._missing = tuple(
                        f"IMU-{index}" for index in np.flatnonzero(missing_mask))
                    self._latest = frame
                    self._frame_times.append(now)
                    dropped = self._mag_drop_monitor.update(frame.mag)
                with self._lock, self._transport_state_lock():
                    if not self._pose_is_current(frame):
                        continue
                    self._put_latest(frame)
                if dropped:
                    channels = ', '.join(f'IMU-{channel}: 3->{level}' for channel, level in dropped)
                    print(f'[WARNING][MAG] {self.config.side} {self.serial_number} '
                          f'磁力计校准状态下降 / calibration lost '
                          f'(physical channels, 0-based): {channels}. '
                          '请检查磁环境并缓慢画圆重新校准。', flush=True)
        except Exception as exc:
            with self._lock:
                if not self._stop.is_set():
                    self._fail_pose_stream(exc)

    def _fail_pose_stream(self, exc: Exception) -> None:
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._stop.set()
        self._latest = None
        self._queue.finish(exc)

    def _transport_state_lock(self):
        transport = None if self._stream is None else self._stream._transport
        return getattr(transport, "_state_lock", nullcontext())

    def _generation_is_current(self, generation: int | None) -> bool:
        if generation is None:
            return True  # Synthetic pose inputs carry no transport metadata.
        transport = None if self._stream is None else self._stream._transport
        return (transport is not None and transport.connected
                and transport.connection_generation == generation)

    def _pose_is_current(self, frame: HandFrame | None) -> bool:
        return (frame is not None and not self._stop.is_set()
                and self._generation_is_current(getattr(frame, "_connection_generation", None)))

    def _tactile_scan_frames(
            self, stream: TactileStream,
            timeout: float) -> Iterator[TactileFrame]:
        """Yield one scan per poll: the newest the transport had queued.

        Deliberately **not** :func:`common.frame_pacing.recent_batch`, which the
        IMU loop uses, because the two caches want different things from a batch.
        Every raw IMU frame gets solved, so a trimmed batch would silently lose
        samples; a pressure scan only ever replaces ``_latest_tactile_raw``, so
        the newest scan is already the whole of what a batch is worth and
        trimming the rest would arrive at that same single frame.  The drain
        still matters: the transport buffers 2048 scans, so reading them one at a
        time from the oldest end would publish a matrix up to 34 s behind the
        glove.

        (``recent_batch`` would not work here in any case: it keys on
        ``device_timestamp_us``, and ``TactileFrame`` carries ``timestamp_us``.
        A batch of one never reaches that code, which is what kept the mismatch
        hidden.)
        """

        while not self._stop.is_set():
            try:
                first = stream.read(timeout=timeout)
            except StreamTimeoutError:
                continue
            batch = [first, *stream.poll()]
            yield batch[-1]

    def _tactile_loop(self) -> None:
        """Cache the newest raw pressure frame for the next HandFrame.

        Raw only: the SDK hands over the 12-bit ADC matrix and stops there.
        Filtering, gating and force fitting are the caller's -- the SDK's job
        ends at delivering the channel faithfully, which is also why nothing
        here downsamples or interpolates.
        """

        try:
            assert self._stream is not None
            owner = self._stream
            stream = owner.tactile_stream()
            for tactile_frame in self._tactile_scan_frames(
                    stream, _FRAME_WAIT_S):
                if self._stop.is_set():
                    break
                with self._lock:
                    state = self._force_snapshot()
                    if self._stream is not owner:
                        break
                    generation = getattr(tactile_frame, "_connection_generation", None)
                    if state["connection_id"] is None or generation != owner._transport.connection_generation:
                        continue
                    self._latest_tactile_raw = tactile_frame.samples.copy()
                    self._tactile_connection_generation = generation
                    self._tactile_connection_id = state["connection_id"]
                    self._tactile_generation += 1
                    self._tactile_fps = self._note_tactile_scan(
                        time.monotonic())
        except Exception as exc:
            with self._lock:
                self._last_tactile_error = f"{type(exc).__name__}: {exc}"

    def read(self, timeout: float | None = None) -> HandFrame:
        return self._read_frame(self._read_queue(), timeout)

    def _read_queue(self) -> _PoseQueue:
        with self._lifecycle_lock:
            self.start()
            return self._queue

    def _check_read_queue(self, queue: _PoseQueue) -> None:
        with self._lock:
            if queue.error is not None:
                raise StreamClosedError(
                    f"{self.config.side} glove stream stopped: "
                    f"{type(queue.error).__name__}: {queue.error}") from queue.error
            if queue.ended or not self._started or queue is not self._queue:
                raise StreamClosedError(f"{self.config.side} glove was closed")

    def _read_frame(self, queue: _PoseQueue, timeout: float | None) -> HandFrame:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                frame = queue.get(timeout=remaining)
            except Empty as exc:
                self._check_read_queue(queue)
                raise StreamTimeoutError(
                    f"Timed out waiting for {self.config.side} glove frame") from exc
            with self._lock, self._transport_state_lock():
                if frame is None:
                    # Pass the shutdown marker to the next blocked reader.
                    queue.put_nowait(None)
                self._check_read_queue(queue)
                if self._pose_is_current(frame):
                    return self._current_force_frame(frame)

    def _current_force_frame(self, frame):
        if frame is None:
            return None
        with self._lock:
            state = self._force_snapshot()
            connection_id = getattr(frame, "_force_connection_id", None)
            if (getattr(frame, "_force_revision", None) != self._force_revision
                    or connection_id != state["connection_id"]
                    or getattr(frame, "_force_baseline_revision", None) != self._force_baseline_revision):
                source, status, error = state["source"], state["status"], state["error"]
                if connection_id is None or connection_id != state["connection_id"]:
                    source, status, error = "unavailable", "disconnected", "frame belongs to a previous connection"
                elif (state["calibration"] is not None
                      and state["calibration"]["model"] == "delta_adc"
                      and not state["baseline_ready"]):
                    status, error = "no_baseline", "measured force baseline is unavailable"
                filtered = replace(frame, force_n=None, force_source=source,
                                   force_status=status, force_error=error, force_fresh=False)
                # Preserve origin: replacing dataclass fields drops internal metadata.
                for name in ("_force_revision", "_force_connection_id", "_force_baseline_id",
                             "_force_baseline_revision", "_connection_generation"):
                    object.__setattr__(filtered, name, getattr(frame, name, None))
                return filtered
            return frame

    def latest(self) -> HandFrame | None:
        with self._lock, self._transport_state_lock():
            if self._pose_is_current(self._latest):
                return self._current_force_frame(self._latest)
            return None

    def _param_stream(self) -> RawImuStream:
        """The running stream, opening the glove first if it has not been."""

        if self._stream is None:
            self.start()
        assert self._stream is not None
        return self._stream

    def wait_for_firmware_version(
            self, timeout_s: float = PARAM_VERSION_WAIT_S) -> str | None:
        """Block until the device's 0x01 status frame names a firmware, or give up.

        The version is what every pressure-fit entry point gates on, and it
        arrives with the first frame the reader parses -- seconds later on a
        port opened before the glove was powered up.  Returns ``None`` on
        timeout, which is not the same as an unsupported firmware.
        """

        return self._param_stream().wait_for_version(timeout_s)

    def health(self) -> DeviceHealth:
        with self._lock, self._transport_state_lock():
            current = self._pose_is_current(self._latest)
            times = list(self._frame_times) if current else []
            missing = self._missing if current else tuple(f"IMU-{index}" for index in range(16))
            errors = [value for value in (
                self._last_error, self._last_tactile_error) if value]
            connected = (self._started and not self._stop.is_set()
                         and self._stream is not None and self._stream.connected)
        fps = 0.0
        if len(times) >= 2:
            fps = (len(times) - 1) / max(times[-1] - times[0], 1e-9)
        return DeviceHealth(
            connected=connected,
            ready_imu_count=16 - len(missing),
            missing_imus=missing,
            stream_fps=fps,
            message=self._health_message(),
            last_error="; ".join(errors) if errors else None,
        )

    @property
    def port(self) -> str | None:
        return getattr(self._stream, "port", None) or self._resolved_port

    def info(self) -> GloveInfo:
        """What is running this glove, and where it is attached.

        Answers with whatever is known now: ``firmware_version`` stays ``None``
        until the device's 0x01 status frame has been parsed, which is a few
        hundred milliseconds after :meth:`start`.
        """

        firmware = self._firmware_version
        return GloveInfo(
            side=self.config.side,
            sdk_version=sdk_version(),
            firmware_version=firmware,
            link_kind=(self._link.kind if self._link is not None
                       else LINK_AUTO),
            port=self.port or "",
            serial_number=self.serial_number,
            mag_available=mag_available(firmware),
            backend=self.solver.backend,
        )

    @property
    def mesh_vertices_m(self) -> np.ndarray | None:
        """The newest solved MANO mesh, wrist-centred, ``(778, 3)`` float32."""

        with self._lock, self._transport_state_lock():
            if self._started and self._pose_is_current(self._latest):
                return self.solver.mesh_vertices_m
            return None

    def _health_message(self) -> str:
        """Say which link is carrying this hand, and what would block the other."""

        if self._link is None:
            return f"{self.config.side} glove not started"
        port = self.port
        link = replace(self._link, port=port) if port and port != self._link.port else self._link
        base = f"{self.config.side} glove on {link.label}"
        if self._link.note:
            base = f"{base} -- {self._link.note}"
        return base

    def close(self) -> None:
        with self._lifecycle_lock:
            self._stop_stream()

    def _stop_stream(self) -> None:
        with self._lock:
            self._queue.finish()
            self._started = False
            self._stop.set()
            self._latest = None
            self._missing = tuple(f"IMU-{index}" for index in range(16))
            self._frame_times.clear()
            self._force_connection_key = None
            self._force_connection_id = None
            self._clear_force_baseline()
            self._invalidate_force("disconnected", "engine closed")
        # The raw reads can auto-start their transport. Let the workers leave
        # their bounded reads before stopping that transport.
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        if self._stream is not None:
            self._stream.stop()
        if any(thread.is_alive() for thread in self._threads):
            raise StreamClosedError(f"{self.config.side} glove workers did not stop")
        self._threads.clear()
        self._stream = None
        self._arrivals.clear()
        self._arrival_count = 0
        self._smoother = _JointSmoother(self.config.smoothing_ms / 1000.0)
        self.solver.reset_stream_state()

    @property
    def backend(self) -> dict:
        return self.solver.backend

    def neutral_joints(self) -> np.ndarray:
        return self.solver.neutral_joints()


class Glove:
    __slots__ = ("_impl",)

    def __init__(self, config: GloveConfig):
        self._impl = GloveEngine(config)

    @classmethod
    def _wrap(cls, engine: GloveEngine) -> "Glove":
        """Adopt an engine that is already running.

        :class:`~sdk.bimanual.BimanualGlove` builds its two hands itself, so it
        cannot construct them the usual way -- and without this its callers had
        no public route to per-hand state like ``mesh_vertices_m``, which left
        them reaching through ``BimanualGlove._impl`` for it.  The engine must
        already be started; this only attaches a facade to it.
        """

        glove = cls.__new__(cls)
        glove._impl = engine
        return glove

    def start(self) -> "Glove":
        self._impl.start()
        return self

    def read(self, timeout: float | None = None) -> HandFrame:
        return self._impl.read(timeout)

    def latest(self) -> HandFrame | None:
        return self._impl.latest()

    def health(self) -> DeviceHealth:
        return self._impl.health()

    def info(self) -> GloveInfo:
        """Which firmware, which link, which port -- and the SDK's own version."""

        return self._impl.info()

    @property
    def side(self) -> str:
        return self.config.side

    @property
    def config(self) -> GloveConfig:
        return self._impl.config

    @property
    def port(self) -> str | None:
        return self._impl.port

    @property
    def serial_number(self) -> str:
        return self._impl.serial_number

    @property
    def mesh_vertices_m(self) -> np.ndarray | None:
        """The newest solved MANO mesh, for a renderer painting at its own rate."""

        return self._impl.mesh_vertices_m

    @property
    def backend(self) -> dict:
        return self._impl.backend

    def neutral_joints(self):
        return self._impl.neutral_joints()

    def set_root_limits_enabled(self, enabled: bool) -> None:
        """Enable four-finger twist/spread and flexion limits (default True)."""
        self._impl.solver.set_root_limits_enabled(enabled)

    def set_root_limits_blending_enabled(self, enabled: bool) -> None:
        """Relax joint limits linearly toward opposition (default True)."""
        self._impl.solver.set_root_limits_blending_enabled(enabled)

    def set_fist_correction_enabled(self, enabled: bool) -> None:
        """Enable four-finger fist correction; the thumb keeps its independent motion."""
        self._impl.solver.set_fist_correction_enabled(enabled)

    def set_contact_controls(self, enabled: bool, blend: float = 1.0) -> None:
        """Apply attraction controls to subsequent solved frames."""
        self._impl.solver.set_contact_controls(enabled, blend)

    def wait_for_firmware_version(
            self, timeout_s: float = PARAM_VERSION_WAIT_S) -> str | None:
        """Block until the device names its firmware, or give up (``None``)."""

        return self._impl.wait_for_firmware_version(timeout_s)

    @property
    def force_calibration_status(self):
        return self._impl.force_calibration_status

    def read_force_calibration(self, *, timeout_s=5.0):
        return self._impl.read_force_calibration(timeout_s=timeout_s)

    def set_force_baseline(self, baseline, *, connection_id=None, baseline_id=None) -> str | None:
        return self._impl.set_force_baseline(
            baseline, connection_id=connection_id, baseline_id=baseline_id)

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "Glove":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
