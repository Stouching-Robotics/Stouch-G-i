"""Plaintext single-glove orchestration over public SDK interfaces.

This module owns threading, synchronization, display smoothing, and recording.
The protected hand solver remains reachable only through ``HandSolver``.
"""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time

import numpy as np

from glove_io.single_recording import SingleRecordingSession
from glove_io.tactile_processing import TactilePreprocessor
from common.errors import (
    DeviceNotFoundError, RecordingError,
    StreamClosedError, StreamTimeoutError)
from runtime.interfaces.device import DeviceManager
from runtime.interfaces.sensors import RawImuStream
from runtime.interfaces.solver import HandSolver
from common.types import (
    DeviceHealth, GloveConfig, HandFrame,
    RecordingResult, RecordingStatus)


def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

SDK_ROOT = _sdk_root(Path(__file__).resolve().parent)
DEFAULT_REGISTRY = SDK_ROOT / "config" / "glove_devices.json"


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


def _resolve_port(config: GloveConfig) -> tuple[str, str]:
    manager = DeviceManager(config.registry or DEFAULT_REGISTRY)
    if config.serial_number:
        wanted = str(config.serial_number).upper()
        matches = [
            device for device in manager.list_devices()
            if device.serial_number.upper() == wanted
        ]
        if len(matches) != 1:
            raise DeviceNotFoundError(
                f"Expected one STM32 glove with serial {wanted}, found {len(matches)}")
        return matches[0].device, wanted
    return manager.resolve_port(config.side)


def _usb_ids(calibration: Path) -> tuple[int, int]:
    """Read non-secret transport identifiers from a calibration file."""

    try:
        payload = json.loads(calibration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0x0483, 0x5740
    return int(payload.get("usb_vid", 0x0483)), int(
        payload.get("usb_pid", 0x5740))


class GloveEngine:
    def __init__(self, config: GloveConfig, queue_size: int = 128):
        self.config = config
        self.solver = HandSolver(config.side, config.calibration)
        self.port: str | None = None
        self.serial_number: str = str(config.serial_number or "")
        self._stream: RawImuStream | None = None
        self._queue: Queue[HandFrame] = Queue(maxsize=max(2, int(queue_size)))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._latest: HandFrame | None = None
        self._latest_tactile: np.ndarray | None = None
        self._latest_tactile_raw: np.ndarray | None = None
        self._lock = threading.RLock()
        self._missing = tuple(f"IMU-{index}" for index in range(16))
        self._frame_times: deque[float] = deque(maxlen=120)
        self._smoother = _JointSmoother(config.smoothing_ms / 1000.0)
        self._tactile_pre = TactilePreprocessor(
            base_gate=0.0,
            dynamic_noise_ratio=0.0,
            temporal_smooth=0.15,
            spatial_filter=False,
            calibration_frames=100,
            bypass_gates=False,
        )
        self._started = False
        self._last_error: str | None = None
        self._last_tactile_error: str | None = None
        self._recording: SingleRecordingSession | None = None
        self._recording_lock = threading.RLock()

    def start(self) -> "GloveEngine":
        if self._started:
            return self
        self.port, self.serial_number = _resolve_port(self.config)
        usb_vid, usb_pid = _usb_ids(Path(self.config.calibration))
        self._stream = RawImuStream(
            serial_port=self.port,
            usb_vid=usb_vid,
            usb_pid=usb_pid,
        )
        self._started = True
        self._last_error = None
        self._last_tactile_error = None
        self._stop.clear()
        self._stream.start()
        self._threads = [
            threading.Thread(target=self._imu_loop, name=f"sdk-{self.config.side}-imu",
                             daemon=True),
        ]
        if self.config.include_tactile:
            self._threads.append(threading.Thread(
                target=self._tactile_loop,
                name=f"sdk-{self.config.side}-tactile", daemon=True))
        for thread in self._threads:
            thread.start()
        return self

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

    def _imu_loop(self) -> None:
        try:
            assert self._stream is not None
            for imu_frame in self._stream.frames():
                if self._stop.is_set():
                    break
                keypoints = self.solver.process(imu_frame)
                timestamp_s = keypoints.timestamp_us / 1_000_000.0
                smoothed = self._smoother.update(
                    keypoints.joints_m, timestamp_s)
                with self._lock:
                    tactile = (None if self._latest_tactile is None
                               else self._latest_tactile.copy())
                    tactile_raw = (None if self._latest_tactile_raw is None
                                   else self._latest_tactile_raw.copy())
                frame = HandFrame(
                    side=self.config.side,
                    sequence=keypoints.sequence,
                    timestamp_us=keypoints.timestamp_us,
                    imu_xyzw=keypoints.imu_xyzw,
                    imu_valid=keypoints.valid_mask,
                    sensor_age_s=keypoints.sensor_age_s,
                    joints_raw_m=keypoints.joints_m,
                    joints_smoothed_m=smoothed,
                    tactile=tactile,
                    status=keypoints.status,
                    raw_imu_xyzw=imu_frame.quaternions_xyzw,
                    raw_imu_present=imu_frame.present_mask,
                    raw_imu_valid=imu_frame.valid_mask,
                    raw_device_timestamp_us=imu_frame.device_timestamp_us,
                    tactile_raw=tactile_raw,
                )
                now = time.monotonic()
                with self._lock:
                    missing_mask = ~np.isfinite(keypoints.sensor_age_s)
                    self._missing = tuple(
                        f"IMU-{index}" for index in np.flatnonzero(missing_mask))
                    self._latest = frame
                    self._frame_times.append(now)
                self._append_recording(frame)
                self._put_latest(frame)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._stop.set()

    def _tactile_loop(self) -> None:
        try:
            assert self._stream is not None
            for tactile_frame in self._stream.tactile_stream().frames():
                if self._stop.is_set():
                    break
                processed, _ = self._tactile_pre.process(tactile_frame.samples)
                with self._lock:
                    self._latest_tactile_raw = tactile_frame.samples.copy()
                    if processed is not None:
                        self._latest_tactile = np.asarray(
                            processed, dtype=np.float32).reshape(16, 16).copy()
        except Exception as exc:
            with self._lock:
                self._last_tactile_error = f"{type(exc).__name__}: {exc}"

    def read(self, timeout: float | None = None) -> HandFrame:
        if not self._started:
            self.start()
        try:
            return self._queue.get(timeout=timeout)
        except Empty as exc:
            with self._lock:
                last_error = self._last_error
            if last_error is not None:
                raise StreamClosedError(
                    f"{self.config.side} glove stream stopped: {last_error}") from exc
            raise StreamTimeoutError(
                f"Timed out waiting for {self.config.side} glove frame") from exc

    def latest(self) -> HandFrame | None:
        with self._lock:
            return self._latest

    def _append_recording(self, frame: HandFrame) -> None:
        with self._recording_lock:
            if self._recording is not None:
                self._recording.add(frame)

    def start_recording(self, name: str | None = None) -> RecordingStatus:
        if not self._started:
            self.start()
        with self._recording_lock:
            if self._recording is None:
                self._recording = SingleRecordingSession(
                    self.config, self.serial_number)
            try:
                return self._recording.start(name)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RecordingError(str(exc)) from exc

    def stop_recording(self) -> RecordingResult:
        with self._recording_lock:
            if self._recording is None:
                raise RecordingError("recording has not been started")
            try:
                return self._recording.stop()
            except (OSError, RuntimeError, ValueError) as exc:
                raise RecordingError(str(exc)) from exc

    def recording_status(self) -> RecordingStatus:
        with self._recording_lock:
            if self._recording is None:
                return RecordingStatus("idle", 0, 0, None)
            return self._recording.status()

    def health(self) -> DeviceHealth:
        with self._lock:
            times = list(self._frame_times)
            missing = self._missing
            errors = [value for value in (
                self._last_error, self._last_tactile_error) if value]
        fps = 0.0
        if len(times) >= 2:
            fps = (len(times) - 1) / max(times[-1] - times[0], 1e-9)
        return DeviceHealth(
            connected=(self._started and not self._stop.is_set()
                       and self._stream is not None
                       and self._stream.connected),
            ready_imu_count=16 - len(missing),
            missing_imus=missing,
            stream_fps=fps,
            message=f"{self.config.side} glove on {self.port}",
            last_error="; ".join(errors) if errors else None,
        )

    def close(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        with self._recording_lock:
            if (self._recording is not None
                    and self._recording.state == "recording"):
                self._recording.stop()
        self._started = False

    @property
    def backend(self) -> dict:
        return self.solver.backend

    def neutral_joints(self) -> np.ndarray:
        return self.solver.neutral_joints()
