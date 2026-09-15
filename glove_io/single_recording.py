"""Plaintext repeated single-glove Parquet recording implementation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from glove_io.recording import (
    KeypointParquetRecorderWithIMU, SkeletonRenderer, make_hand_result)
from glove_io.session import (
    load_session_metadata, session_metadata_path,
    update_session_metadata, write_session_metadata)
from common.types import (
    GloveConfig, HandFrame, RecordingResult, RecordingStatus)


def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

SDK_ROOT = _sdk_root(Path(__file__).resolve().parent)


class SingleRecordingSession:
    def __init__(self, config: GloveConfig, serial_number: str = ""):
        self.config = config
        default_root = SDK_ROOT / (
            "data/keypoints_21_left"
            if config.side == "left" else "data/keypoints_21")
        self.output_root = Path(config.recording_root or default_root)
        self.serial_number = str(serial_number or "")
        self.projector = SkeletonRenderer()
        self.base_output_path: Path | None = None
        self.output_path: Path | None = None
        self.recorder: KeypointParquetRecorderWithIMU | None = None
        self.state = "idle"
        self.take_index = 0
        self.started_s: float | None = None
        self.metadata_written = False

    @property
    def frame_count(self) -> int:
        return len(self.recorder.rows) if self.recorder is not None else 0

    def _new_base_path(self, name: str | None) -> Path:
        session_name = str(name or datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not session_name or "/" in session_name or "\\" in session_name:
            raise ValueError("recording name must not contain a path")
        return self.output_root / session_name / "chunk-000.parquet"

    def _allocate_output_path(self) -> Path:
        assert self.base_output_path is not None
        base = self.base_output_path
        if (self.take_index == 1 and not base.exists()
                and not session_metadata_path(base).exists()):
            return base
        root = base.parent.parent
        stem = base.parent.name
        ordinal = max(2, self.take_index)
        while True:
            candidate = root / f"{stem}_take_{ordinal:03d}" / base.name
            if (not candidate.exists()
                    and not session_metadata_path(candidate).exists()):
                return candidate
            ordinal += 1

    def start(self, name: str | None = None) -> RecordingStatus:
        if self.state not in {"idle", "saved"}:
            raise RuntimeError(f"cannot start recording from state {self.state}")
        if self.base_output_path is None or name is not None:
            self.base_output_path = self._new_base_path(name)
            if name is not None and self.state == "saved":
                self.take_index = 0
        self.take_index += 1
        self.output_path = self._allocate_output_path()
        self.recorder = KeypointParquetRecorderWithIMU(
            self.output_path,
            checkpoint_frames=600,
            side=self.config.side,
            usb_serial=self.serial_number,
        )
        self.started_s = None
        self.metadata_written = False
        self.state = "recording"
        return self.status()

    def _metadata(self) -> dict:
        return {
            "mode": "single",
            "side": self.config.side,
            "sides": [self.config.side],
            "sample_fps": float(self.config.sample_rate_hz),
            "devices": {
                self.config.side: {"usb_serial": self.serial_number},
            },
            "calibration_files": {
                self.config.side: str(Path(
                    self.config.calibration).resolve()),
            },
            "recording": {"take_index": self.take_index},
        }

    def add(self, frame: HandFrame) -> bool:
        if self.state != "recording" or self.recorder is None:
            return False
        assert self.output_path is not None
        frame_s = float(frame.timestamp_us) / 1_000_000.0
        if self.started_s is None:
            self.started_s = frame_s
        if not self.metadata_written:
            write_session_metadata(self.output_path, self._metadata())
            self.metadata_written = True
        timestamp_s = max(0.0, frame_s - self.started_s)
        uv, _ = self.projector.project(frame.joints_smoothed_m)
        result = make_hand_result(
            frame.side,
            frame.joints_raw_m,
            frame.joints_smoothed_m,
            uv,
            imu_ready=bool(frame.imu_valid.all()),
        )
        self.recorder.add(
            self.frame_count,
            timestamp_s,
            result,
            imu_quats=frame.imu_xyzw,
            tactile=frame.tactile,
            raw_imu_quats=frame.raw_imu_xyzw,
            raw_present_mask=frame.raw_imu_present,
            raw_valid_mask=frame.raw_imu_valid,
            raw_device_timestamp_us=frame.raw_device_timestamp_us,
            tactile_raw=frame.tactile_raw,
        )
        return True

    def stop(self) -> RecordingResult:
        if self.state != "recording" or self.recorder is None:
            raise RuntimeError("recording has not been started")
        path = self.recorder.save()
        if path is not None:
            rows = self.recorder.rows
            duration_s = (
                float(rows[-1]["timestamp"]) - float(rows[0]["timestamp"])
                if len(rows) >= 2 else 0.0)
            sample_fps = (
                (len(rows) - 1) / duration_s
                if len(rows) >= 2 and duration_s > 0.0
                else float(self.config.sample_rate_hz))
            update_session_metadata(path, {
                "sample_fps": sample_fps,
                "recording": {
                    "frame_count": len(rows),
                    "duration_s": duration_s,
                },
            })
            metadata = load_session_metadata(path)
        else:
            metadata = {}
        self.state = "saved"
        recording = metadata.get("recording", {})
        return RecordingResult(
            parquet_path=path,
            session_path=(session_metadata_path(path)
                          if path is not None else None),
            take_index=self.take_index,
            frame_count=int(recording.get("frame_count", 0)),
            duration_s=float(recording.get("duration_s", 0.0)),
            sample_fps=float(metadata.get(
                "sample_fps", self.config.sample_rate_hz)),
        )

    def status(self) -> RecordingStatus:
        return RecordingStatus(
            state=self.state,
            take_index=self.take_index,
            frame_count=self.frame_count,
            output_path=self.output_path,
        )
