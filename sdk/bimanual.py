"""Plaintext bimanual synchronization and recording orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from sdk.glove import SDK_ROOT, Glove, GloveEngine
from common.errors import RecordingError, StreamTimeoutError
from common.types import (
    BimanualConfig, BimanualFrame, BimanualHealth, HandFrame,
    RecordingResult, RecordingStatus)
from sdk.io.recording import BimanualRecordingSession, SkeletonRenderer
from sdk.io.session import (
    load_session_metadata, session_metadata_path)


class BimanualEngine:
    def __init__(self, config: BimanualConfig):
        self.config = config
        self.left = GloveEngine(config.left)
        self.right = GloveEngine(config.right)
        self._sequence = 0
        self._started = False
        self._recording: BimanualRecordingSession | None = None
        self._recording_lock = threading.RLock()
        self._projector = SkeletonRenderer()
        self._last_sync_error_ms: float | None = None

    def start(self) -> "BimanualEngine":
        if not self._started:
            self.left.start()
            self.right.start()
            self._started = True
        return self

    @staticmethod
    def _read_fresh(engine: GloveEngine, deadline: float | None) -> "HandFrame":
        """One hand's next frame, waiting no longer than ``deadline``."""

        remaining = (None if deadline is None
                     else max(0.0, deadline - time.monotonic()))
        return engine.read(remaining)

    def read(self, timeout: float | None = None) -> BimanualFrame:
        """The next synchronized pair, or raise if nothing new arrived.

        The two hands are independent links, so one of them can go quiet while
        the other keeps delivering.  Reading them in turn and letting whichever
        timeout raise used to throw away the frame the healthy hand had already
        produced, which froze *both* hands on screen until the caller's own
        staleness check took over.  A hand that produced nothing therefore
        contributes the newest frame it did deliver, and the pair survives.

        The exception is still raised when *neither* hand has anything new, so
        a caller polling at its own pace keeps seeing "nothing yet" rather than
        a stream of repeats -- an idle glove must not look busy.  It is raised
        as well when the silent hand has never delivered anything at all, since
        half a pair is not a frame.  A repeated ``sequence`` marks the hand
        that was filled in, and its ``sensor_age_s`` and health show how far
        behind it is, so nothing is hidden.
        """

        if not self._started:
            self.start()
        # One budget for the pair, not one per hand: charging each hand the
        # full timeout would make a silent link cost twice what the caller
        # asked to wait.
        deadline = None if timeout is None else time.monotonic() + timeout
        fresh = []
        for engine in (self.left, self.right):
            try:
                fresh.append(self._read_fresh(engine, deadline))
            except StreamTimeoutError:
                fresh.append(None)
        left, right = fresh
        if left is None and right is None:
            raise StreamTimeoutError("neither hand delivered a frame")
        if left is None:
            left = self.left.latest()
        if right is None:
            right = self.right.latest()
        if left is None or right is None:
            silent = self.config.left.side if left is None else self.config.right.side
            raise StreamTimeoutError(f"{silent} hand has delivered no frame yet")
        timestamp_us = max(left.timestamp_us, right.timestamp_us)
        frame = BimanualFrame(
            sequence=self._sequence,
            timestamp_us=timestamp_us,
            left=left,
            right=right,
            synchronization_error_ms=(
                abs(left.timestamp_us - right.timestamp_us) / 1000.0),
        )
        self._last_sync_error_ms = frame.synchronization_error_ms
        self._sequence += 1
        self._append_recording(frame)
        return frame

    def _recording_output(self, name: str | None) -> Path:
        root = (Path(self.config.output_root)
                if self.config.output_root is not None
                else SDK_ROOT / "data/keypoints_21_bimanual")
        session_name = str(name or datetime.now().strftime("%Y%m%d_%H%M%S"))
        if "/" in session_name or "\\" in session_name:
            raise RecordingError("recording name must not contain a path")
        return root / session_name / "chunk-000.parquet"

    def start_recording(self, name: str | None = None) -> RecordingStatus:
        if not self._started:
            self.start()
        with self._recording_lock:
            if self._recording is None:
                output = self._recording_output(name)
                self._recording = BimanualRecordingSession(
                    output,
                    serials={
                        "left": self.left.serial_number,
                        "right": self.right.serial_number,
                    },
                    metadata={
                        "mode": "bimanual",
                        "sides": ["left", "right"],
                        "sample_fps": float(min(
                            self.config.left.sample_rate_hz,
                            self.config.right.sample_rate_hz)),
                        "calibration_files": {
                            "left": str(Path(self.config.left.calibration).resolve()),
                            "right": str(Path(self.config.right.calibration).resolve()),
                        },
                        "display": {
                            "recorded_source": "raw",
                            "separation_m": float(
                                self.config.display_separation_m),
                        },
                    },
                )
            try:
                started = self._recording.start(datetime.now().timestamp())
            except (OSError, RuntimeError, ValueError) as exc:
                raise RecordingError(str(exc)) from exc
            if not started:
                raise RecordingError(
                    f"cannot start recording from state {self._recording.state}")
            return self.recording_status()

    def _append_recording(self, frame: BimanualFrame) -> None:
        with self._recording_lock:
            if self._recording is None or self._recording.state != "recording":
                return
            raw = np.stack([
                frame.left.joints_raw_m, frame.right.joints_raw_m])
            smoothed = np.stack([
                frame.left.joints_smoothed_m, frame.right.joints_smoothed_m])
            hands2d = np.stack([
                self._projector.project(smoothed[index])[0]
                for index in range(2)
            ])
            result = {
                "hands2d": hands2d,
                "hands3d": raw,
                "smoothed": smoothed,
                "labels": ["Left", "Right"],
                "presents": [True, True],
                "propagated": [False, False],
                "stage2": [
                    frame.left.status is not None,
                    frame.right.status is not None,
                ],
                "reprojection_error": [float("nan"), float("nan")],
            }
            runtimes = {
                "left": SimpleNamespace(
                    latest_imu=frame.left.imu_xyzw,
                    # No processed pressure map from the SDK -- see
                    # common.types.HandFrame.tactile_raw.
                    latest_tactile=None,
                    latest_raw_imu=frame.left.raw_imu_xyzw,
                    latest_raw_present=frame.left.raw_imu_present,
                    latest_raw_valid=frame.left.raw_imu_valid,
                    latest_raw_device_timestamp_us=(
                        frame.left.raw_device_timestamp_us),
                    latest_tactile_raw=frame.left.tactile_raw),
                "right": SimpleNamespace(
                    latest_imu=frame.right.imu_xyzw,
                    latest_tactile=None,
                    latest_raw_imu=frame.right.raw_imu_xyzw,
                    latest_raw_present=frame.right.raw_imu_present,
                    latest_raw_valid=frame.right.raw_imu_valid,
                    latest_raw_device_timestamp_us=(
                        frame.right.raw_device_timestamp_us),
                    latest_tactile_raw=frame.right.tactile_raw),
            }
            self._recording.add(
                frame.timestamp_us / 1_000_000.0, result, runtimes)

    def stop_recording(self) -> RecordingResult:
        with self._recording_lock:
            if self._recording is None:
                raise RecordingError("recording has not been started")
            try:
                path = self._recording.stop_and_save()
            except (OSError, RuntimeError, ValueError) as exc:
                raise RecordingError(str(exc)) from exc
            metadata = load_session_metadata(path) if path is not None else {}
            recording = metadata.get("recording", {})
            return RecordingResult(
                parquet_path=path,
                session_path=(session_metadata_path(path) if path is not None else None),
                take_index=int(recording.get(
                    "take_index", self._recording.take_index)),
                frame_count=int(recording.get("frame_count", 0)),
                duration_s=float(recording.get("duration_s", 0.0)),
                sample_fps=float(metadata.get("sample_fps", 0.0)),
            )

    def recording_status(self) -> RecordingStatus:
        with self._recording_lock:
            if self._recording is None:
                return RecordingStatus("idle", 0, 0, None)
            return RecordingStatus(
                state=self._recording.state,
                take_index=self._recording.take_index,
                frame_count=self._recording.frame_count,
                output_path=self._recording.output_path,
            )

    def health(self) -> BimanualHealth:
        return BimanualHealth(
            left=self.left.health(),
            right=self.right.health(),
            synchronization_error_ms=self._last_sync_error_ms,
        )

    def close(self) -> None:
        with self._recording_lock:
            if self._recording is not None and self._recording.state == "recording":
                self._recording.stop_and_save()
        self.left.close()
        self.right.close()
        self._started = False


class BimanualGlove:
    __slots__ = ("_impl",)

    def __init__(self, config: BimanualConfig):
        self._impl = BimanualEngine(config)

    @property
    def left(self) -> Glove:
        """The left hand's own facade -- for state a frame does not carry.

        ``BimanualFrame`` carries one complete ``HandFrame`` per hand, which is
        everything a frame reader needs.  Two things live outside it: the newest
        solved mesh, and the solver backend.  Both are properties of one hand's
        engine, and the returned facade is the same public surface a
        single-hand caller already gets, so nobody has to reach into ``_impl``.
        """

        return Glove._wrap(self._impl.left)

    @property
    def right(self) -> Glove:
        """The right hand's own facade.  See :attr:`left`."""

        return Glove._wrap(self._impl.right)

    def start(self) -> "BimanualGlove":
        self._impl.start()
        return self

    def read(self, timeout: float | None = None) -> BimanualFrame:
        return self._impl.read(timeout)

    def health(self) -> BimanualHealth:
        return self._impl.health()

    def start_recording(self, name: str | None = None) -> RecordingStatus:
        return self._impl.start_recording(name)

    def stop_recording(self) -> RecordingResult:
        return self._impl.stop_recording()

    @property
    def recording_status(self) -> RecordingStatus:
        return self._impl.recording_status()

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "BimanualGlove":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
