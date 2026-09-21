"""Plaintext bimanual synchronization and recording orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np

from sdk.glove import GloveEngine
from common.errors import RecordingError
from common.types import (
    BimanualConfig, BimanualFrame, BimanualHealth,
    RecordingResult, RecordingStatus)
from glove_io.recording import BimanualRecordingSession, SkeletonRenderer
from glove_io.session import (
    load_session_metadata, session_metadata_path)


def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)


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

    def read(self, timeout: float | None = None) -> BimanualFrame:
        if not self._started:
            self.start()
        left = self.left.read(timeout)
        right = self.right.read(timeout)
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
                else PROJECT_ROOT / "data/keypoints_21_bimanual")
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
                    latest_tactile=frame.left.tactile,
                    latest_raw_imu=frame.left.raw_imu_xyzw,
                    latest_raw_present=frame.left.raw_imu_present,
                    latest_raw_valid=frame.left.raw_imu_valid,
                    latest_raw_device_timestamp_us=(
                        frame.left.raw_device_timestamp_us),
                    latest_tactile_raw=frame.left.tactile_raw),
                "right": SimpleNamespace(
                    latest_imu=frame.right.imu_xyzw,
                    latest_tactile=frame.right.tactile,
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
