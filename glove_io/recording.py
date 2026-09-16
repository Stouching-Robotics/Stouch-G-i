"""Plaintext projection and Parquet writers used by capture applications."""

from __future__ import annotations

import copy
import math
import os
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from glove_io.session import (
    session_metadata_path,
    update_session_metadata,
    write_session_metadata,
)

try:
    import polars as pl
except ImportError:  # pragma: no cover - recording is optional
    pl = None


class JointPositionSmoother:
    """Display-only exponential smoother for 21 Cartesian keypoints."""

    def __init__(self, tau_s: float = 0.035):
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


class SkeletonRenderer:
    """Fixed virtual camera used only for the recorded 2-D projection."""

    def __init__(self, size: int = 720, pixels_per_metre: float = 2850.0):
        self.size = int(size)
        self.pixels_per_metre = float(pixels_per_metre)
        self.view_rotation = R.from_euler(
            "xyz", [58.0, 0.0, -42.0], degrees=True
        ).as_matrix().astype(np.float32)

    def project(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(joints, dtype=np.float32).reshape(21, 3)
        camera_points = points @ self.view_rotation.T
        uv = np.empty((21, 2), dtype=np.float32)
        uv[:, 0] = self.size * 0.5 + camera_points[:, 0] * self.pixels_per_metre
        uv[:, 1] = self.size * 0.60 - camera_points[:, 1] * self.pixels_per_metre
        return uv, camera_points


def make_hand_result(
    side: str,
    joints: np.ndarray,
    smoothed: np.ndarray,
    virtual_uv: np.ndarray,
    imu_ready: bool = True,
) -> dict:
    normalized = str(getattr(side, "value", side)).strip().lower()
    if normalized not in ("left", "right"):
        raise ValueError(f"side must be left or right, got {side!r}")
    slot = 0 if normalized == "left" else 1
    hands2d = np.zeros((2, 21, 2), dtype=np.float32)
    hands2d[slot] = np.asarray(virtual_uv, dtype=np.float32).reshape(21, 2)
    hands3d = np.full((2, 21, 3), np.nan, dtype=np.float32)
    hands3d[slot] = np.asarray(joints, dtype=np.float32).reshape(21, 3)
    smoothed3d = np.full((2, 21, 3), np.nan, dtype=np.float32)
    smoothed3d[slot] = np.asarray(smoothed, dtype=np.float32).reshape(21, 3)
    labels = ["", ""]
    labels[slot] = normalized.title()
    presents = [False, False]
    presents[slot] = True
    stage2 = [False, False]
    stage2[slot] = bool(imu_ready)
    return {
        "hands2d": hands2d,
        "hands3d": hands3d,
        "smoothed": smoothed3d,
        "labels": labels,
        "presents": presents,
        "propagated": [False, False],
        "stage2": stage2,
        "reprojection_error": [float("nan"), float("nan")],
    }


class KeypointParquetRecorder:
    """Reference-compatible keypoint writer with atomic checkpoints."""

    def __init__(self, output_path, episode_index: int = 0,
                 task_index: int = 0, checkpoint_frames: int = 600):
        if pl is None:
            raise RuntimeError("recording requires polars")
        self.output_path = Path(output_path)
        self.episode_index = int(episode_index)
        self.task_index = int(task_index)
        self.checkpoint_frames = max(0, int(checkpoint_frames))
        self.rows: list[dict] = []
        self._save_lock = threading.Lock()
        self._finalized = False

    def add(self, frame_index: int, timestamp_s: float, result: dict) -> None:
        keypoints2d = np.asarray(result["hands2d"], np.float32).reshape(2, 21, 2)
        hand3d = np.asarray(result["hands3d"], np.float32).reshape(2, 21, 3)
        smoothed = np.asarray(result["smoothed"], np.float32).reshape(2, 21, 3)
        labels = [str(value or "") for value in result["labels"]]
        presents = [bool(value) for value in result["presents"]]
        self.rows.append({
            "episode_index": self.episode_index,
            "frame_index": int(frame_index),
            "timestamp": np.float32(timestamp_s),
            "task_index": self.task_index,
            "observation.keypoints.stereo_left": keypoints2d.reshape(-1).tolist(),
            "observation.keypoints.stereo_right": np.zeros(84, np.float32).tolist(),
            "observation.keypoints.hand_3d": hand3d.reshape(-1).tolist(),
            "observation.keypoints.reprojection_error": np.asarray(
                result["reprojection_error"], np.float32).reshape(2).tolist(),
            "observation.keypoints.hand_0_present": presents[0],
            "observation.keypoints.hand_1_present": presents[1],
            "observation.keypoints.hand_0_label": labels[0],
            "observation.keypoints.hand_1_label": labels[1],
            "observation.keypoints.stage2": [bool(v) for v in result["stage2"]],
            "observation.keypoints.propagated": [
                bool(v) for v in result["propagated"]],
            "action": [0.0],
            "observation.keypoints.hand_3d_smoothed": smoothed.reshape(-1).tolist(),
        })
        if self.checkpoint_frames and len(self.rows) % self.checkpoint_frames == 0:
            self._schedule_save(list(self.rows))

    def _series(self):
        values = lambda name: [row[name] for row in self.rows]
        return [
            pl.Series("episode_index", values("episode_index"), dtype=pl.Int64),
            pl.Series("frame_index", values("frame_index"), dtype=pl.Int64),
            pl.Series("timestamp", values("timestamp"), dtype=pl.Float32),
            pl.Series("task_index", values("task_index"), dtype=pl.Int64),
            pl.Series("observation.keypoints.stereo_left",
                      values("observation.keypoints.stereo_left"),
                      dtype=pl.Array(pl.Float32, 84)),
            pl.Series("observation.keypoints.stereo_right",
                      values("observation.keypoints.stereo_right"),
                      dtype=pl.Array(pl.Float32, 84)),
            pl.Series("observation.keypoints.hand_3d",
                      values("observation.keypoints.hand_3d"),
                      dtype=pl.Array(pl.Float32, 126)),
            pl.Series("observation.keypoints.reprojection_error",
                      values("observation.keypoints.reprojection_error"),
                      dtype=pl.Array(pl.Float32, 2)),
            pl.Series("observation.keypoints.hand_0_present",
                      values("observation.keypoints.hand_0_present"),
                      dtype=pl.Boolean),
            pl.Series("observation.keypoints.hand_1_present",
                      values("observation.keypoints.hand_1_present"),
                      dtype=pl.Boolean),
            pl.Series("observation.keypoints.hand_0_label",
                      values("observation.keypoints.hand_0_label"), dtype=pl.String),
            pl.Series("observation.keypoints.hand_1_label",
                      values("observation.keypoints.hand_1_label"), dtype=pl.String),
            pl.Series("observation.keypoints.stage2",
                      values("observation.keypoints.stage2"),
                      dtype=pl.Array(pl.Boolean, 2)),
            pl.Series("observation.keypoints.propagated",
                      values("observation.keypoints.propagated"),
                      dtype=pl.Array(pl.Boolean, 2)),
            pl.Series("action", values("action"), dtype=pl.Array(pl.Float32, 1)),
            pl.Series("observation.keypoints.hand_3d_smoothed",
                      values("observation.keypoints.hand_3d_smoothed"),
                      dtype=pl.Array(pl.Float32, 126)),
        ]

    def save(self) -> Path | None:
        """Synchronously write all current rows (the authoritative final save).

        Checkpoint writes run on a background thread and are best-effort; the
        final save marks them stale so a slow checkpoint can never overwrite
        this file with an older snapshot.
        """
        with self._save_lock:
            self._finalized = True
            return self._write_parquet()

    def _write_parquet(self) -> Path | None:
        if not self.rows:
            return None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        pl.DataFrame(self._series()).write_parquet(
            temporary, compression="zstd", statistics=True)
        os.replace(temporary, self.output_path)
        return self.output_path

    def _schedule_save(self, rows: list[dict]) -> None:
        """Write a complete-row snapshot on a daemon thread so checkpointing
        never blocks the recording cadence. ``rows`` must be captured at a
        point where every row is fully populated; it is not mutated after."""
        threading.Thread(
            target=self._save_checkpoint, args=(rows,), daemon=True).start()

    def _save_checkpoint(self, rows: list[dict]) -> None:
        try:
            with self._save_lock:
                if self._finalized:
                    return
                clone = copy.copy(self)
                clone.rows = list(rows)
                clone._write_parquet()
        except BaseException:
            # Checkpoint writes are best-effort crash protection; a failure
            # must never propagate into the recording path.
            pass


class KeypointParquetRecorderWithIMU(KeypointParquetRecorder):
    TACTILE_DIM = 256

    def __init__(self, output_path, episode_index: int = 0,
                 task_index: int = 0, checkpoint_frames: int = 600,
                 side: str = "right", usb_serial: str = ""):
        super().__init__(output_path, episode_index, task_index, 0)
        self.side = str(side).lower()
        if self.side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.IMU_COL = ("observation.imu.quaternion" if self.side == "right"
                        else "observation.imu.left.quaternion")
        self.TACTILE_COL = f"observation.tactile.{self.side}_glove"
        self.RAW_IMU_COL = (
            f"observation.imu.{self.side}.raw_physical_quaternion")
        self.RAW_PRESENT_COL = (
            f"observation.imu.{self.side}.raw_present_mask")
        self.RAW_VALID_COL = f"observation.imu.{self.side}.raw_valid_mask"
        self.RAW_TIMESTAMP_COL = (
            f"observation.imu.{self.side}.device_timestamp_us")
        self.RAW_TACTILE_COL = (
            f"observation.tactile.{self.side}_glove_raw")
        self.SERIAL_COL = f"observation.device.{self.side}.serial"
        self.usb_serial = str(usb_serial or "")
        self._imu_checkpoint_frames = max(0, int(checkpoint_frames))

    def add(self, frame_index, timestamp_s, result, imu_quats=None,
            tactile=None, *, raw_imu_quats=None, raw_present_mask=None,
            raw_valid_mask=None, raw_device_timestamp_us=None,
            tactile_raw=None) -> None:
        super().add(frame_index, timestamp_s, result)
        row = self.rows[-1]
        row[self.IMU_COL] = (np.asarray(imu_quats, np.float32).reshape(64).tolist()
                             if imu_quats is not None
                             else np.full(64, np.nan, np.float32).tolist())
        row[self.TACTILE_COL] = (
            np.asarray(tactile, np.float32).reshape(256).tolist()
            if tactile is not None
            else np.full(256, np.nan, np.float32).tolist())
        row[self.SERIAL_COL] = self.usb_serial
        row[self.RAW_IMU_COL] = (
            np.asarray(raw_imu_quats, np.float32).reshape(64).tolist()
            if raw_imu_quats is not None
            else np.full(64, np.nan, np.float32).tolist())
        row[self.RAW_PRESENT_COL] = (
            np.asarray(raw_present_mask, dtype=bool).reshape(16).tolist()
            if raw_present_mask is not None else [False] * 16)
        row[self.RAW_VALID_COL] = (
            np.asarray(raw_valid_mask, dtype=bool).reshape(16).tolist()
            if raw_valid_mask is not None else [False] * 16)
        row[self.RAW_TIMESTAMP_COL] = (
            int(raw_device_timestamp_us)
            if raw_device_timestamp_us is not None else -1)
        row[self.RAW_TACTILE_COL] = (
            np.asarray(tactile_raw, np.float32).reshape(256).tolist()
            if tactile_raw is not None
            else np.full(256, np.nan, np.float32).tolist())
        if (self._imu_checkpoint_frames
                and len(self.rows) % self._imu_checkpoint_frames == 0):
            self._schedule_save(list(self.rows))

    def _series(self):
        series = super()._series()
        series.extend([
            pl.Series(self.IMU_COL, [row[self.IMU_COL] for row in self.rows],
                      dtype=pl.Array(pl.Float32, 64)),
            pl.Series(self.TACTILE_COL,
                      [row[self.TACTILE_COL] for row in self.rows],
                      dtype=pl.Array(pl.Float32, 256)),
            pl.Series(self.SERIAL_COL,
                      [row[self.SERIAL_COL] for row in self.rows], dtype=pl.String),
            pl.Series(self.RAW_IMU_COL,
                      [row[self.RAW_IMU_COL] for row in self.rows],
                      dtype=pl.Array(pl.Float32, 64)),
            pl.Series(self.RAW_PRESENT_COL,
                      [row[self.RAW_PRESENT_COL] for row in self.rows],
                      dtype=pl.Array(pl.Boolean, 16)),
            pl.Series(self.RAW_VALID_COL,
                      [row[self.RAW_VALID_COL] for row in self.rows],
                      dtype=pl.Array(pl.Boolean, 16)),
            pl.Series(self.RAW_TIMESTAMP_COL,
                      [row[self.RAW_TIMESTAMP_COL] for row in self.rows],
                      dtype=pl.Int64),
            pl.Series(self.RAW_TACTILE_COL,
                      [row[self.RAW_TACTILE_COL] for row in self.rows],
                      dtype=pl.Array(pl.Float32, 256)),
        ])
        return series


class BimanualRecorder(KeypointParquetRecorder):
    EXTRA_DIMS = {
        "observation.imu.left.quaternion": 64,
        "observation.imu.right.quaternion": 64,
        "observation.tactile.left_glove": 256,
        "observation.tactile.right_glove": 256,
        "observation.imu.left.raw_physical_quaternion": 64,
        "observation.imu.right.raw_physical_quaternion": 64,
        "observation.tactile.left_glove_raw": 256,
        "observation.tactile.right_glove_raw": 256,
    }
    BOOL_DIMS = {
        "observation.imu.left.raw_present_mask": 16,
        "observation.imu.right.raw_present_mask": 16,
        "observation.imu.left.raw_valid_mask": 16,
        "observation.imu.right.raw_valid_mask": 16,
    }
    # Device-side IMU arrival rate at sample time, next to the solved keypoints
    # it produced.  Rows are written on the recorder's own wall-clock cadence,
    # so without this a file cannot distinguish "the glove sent 50 Hz" from
    # "the glove sent 100 Hz and the host solved 50 of it".
    RATE_COLS = (
        "observation.imu.left.device_fps",
        "observation.imu.right.device_fps",
    )

    def __init__(self, output_path, episode_index=0, task_index=0,
                 checkpoint_frames=600, serials=None):
        super().__init__(output_path, episode_index, task_index, 0)
        self._checkpoint = max(0, int(checkpoint_frames))
        self.serials = dict(serials or {})

    def add_both(self, frame_index, timestamp_s, result, runtimes) -> None:
        super().add(frame_index, timestamp_s, result)
        row = self.rows[-1]
        for side in ("left", "right"):
            runtime = runtimes[side]
            imu_key = f"observation.imu.{side}.quaternion"
            tactile_key = f"observation.tactile.{side}_glove"
            row[imu_key] = (
                np.asarray(runtime.latest_imu, np.float32).reshape(64).tolist()
                if runtime.latest_imu is not None
                else np.full(64, np.nan, np.float32).tolist())
            row[tactile_key] = (
                np.asarray(runtime.latest_tactile, np.float32).reshape(256).tolist()
                if runtime.latest_tactile is not None
                else np.full(256, np.nan, np.float32).tolist())
            raw_imu = getattr(runtime, "latest_raw_imu", None)
            raw_present = getattr(runtime, "latest_raw_present", None)
            raw_valid = getattr(runtime, "latest_raw_valid", None)
            raw_timestamp = getattr(
                runtime, "latest_raw_device_timestamp_us", None)
            tactile_raw = getattr(runtime, "latest_tactile_raw", None)
            row[f"observation.imu.{side}.raw_physical_quaternion"] = (
                np.asarray(raw_imu, np.float32).reshape(64).tolist()
                if raw_imu is not None
                else np.full(64, np.nan, np.float32).tolist())
            row[f"observation.imu.{side}.raw_present_mask"] = (
                np.asarray(raw_present, dtype=bool).reshape(16).tolist()
                if raw_present is not None else [False] * 16)
            row[f"observation.imu.{side}.raw_valid_mask"] = (
                np.asarray(raw_valid, dtype=bool).reshape(16).tolist()
                if raw_valid is not None else [False] * 16)
            row[f"observation.imu.{side}.device_timestamp_us"] = (
                int(raw_timestamp) if raw_timestamp is not None else -1)
            row[f"observation.tactile.{side}_glove_raw"] = (
                np.asarray(tactile_raw, np.float32).reshape(256).tolist()
                if tactile_raw is not None
                else np.full(256, np.nan, np.float32).tolist())
            row[f"observation.device.{side}.serial"] = self.serials.get(side, "")
            row[f"observation.imu.{side}.device_fps"] = float(
                getattr(runtime, "device_fps", 0.0) or 0.0)
        if self._checkpoint and len(self.rows) % self._checkpoint == 0:
            self._schedule_save(list(self.rows))

    def _series(self):
        series = super()._series()
        for name, dimension in self.EXTRA_DIMS.items():
            series.append(pl.Series(
                name, [row[name] for row in self.rows],
                dtype=pl.Array(pl.Float32, dimension)))
        for name, dimension in self.BOOL_DIMS.items():
            series.append(pl.Series(
                name, [row[name] for row in self.rows],
                dtype=pl.Array(pl.Boolean, dimension)))
        for side in ("left", "right"):
            name = f"observation.imu.{side}.device_timestamp_us"
            series.append(pl.Series(
                name, [row[name] for row in self.rows], dtype=pl.Int64))
        for side in ("left", "right"):
            name = f"observation.device.{side}.serial"
            series.append(pl.Series(
                name, [row[name] for row in self.rows], dtype=pl.String))
        for name in self.RATE_COLS:
            series.append(pl.Series(
                name, [row[name] for row in self.rows], dtype=pl.Float32))
        return series


class BimanualRecordingSession:
    """Repeatable recording lifecycle shared by live UI and public runtime."""

    def __init__(self, output_path, episode_index=0, task_index=0,
                 checkpoint_frames=600, serials=None, metadata=None,
                 enabled=True):
        self.base_output_path = Path(output_path)
        self.output_path = self.base_output_path
        self.episode_index = int(episode_index)
        self.task_index = int(task_index)
        self.checkpoint_frames = int(checkpoint_frames)
        self.serials = dict(serials or {})
        self.metadata = dict(metadata or {})
        self.enabled = bool(enabled)
        self.state = "idle" if self.enabled else "disabled"
        self.recorder: BimanualRecorder | None = None
        self.started_s: float | None = None
        self.metadata_written = False
        self.saved_path: Path | None = None
        self.saved_paths: list[Path] = []
        self.take_index = 0

    @property
    def frame_count(self) -> int:
        return len(self.recorder.rows) if self.recorder is not None else 0

    def _allocate_output_path(self, take_index: int) -> Path:
        base = self.base_output_path
        if (take_index == 1 and not base.exists()
                and not session_metadata_path(base).exists()):
            return base
        if base.name == "chunk-000.parquet":
            root, session_stem = base.parent.parent, base.parent.name
        else:
            root, session_stem = base.parent, base.stem
        ordinal = max(2, int(take_index))
        while True:
            candidate = root / f"{session_stem}_take_{ordinal:03d}" / base.name
            if (not candidate.exists()
                    and not session_metadata_path(candidate).exists()):
                return candidate
            ordinal += 1

    def start(self, now_s: float) -> bool:
        if not self.enabled or self.state not in {"idle", "saved"}:
            return False
        self.take_index += 1
        self.output_path = self._allocate_output_path(self.take_index)
        self.recorder = BimanualRecorder(
            self.output_path, self.episode_index, self.task_index,
            self.checkpoint_frames, serials=self.serials)
        self.started_s = float(now_s)
        self.metadata_written = False
        self.saved_path = None
        self.state = "recording"
        return True

    def add(self, now_s: float, result: dict, runtimes) -> bool:
        if self.state != "recording" or self.recorder is None:
            return False
        if not self.metadata_written:
            metadata = copy.deepcopy(self.metadata)
            metadata["recording"] = {"take_index": self.take_index}
            write_session_metadata(self.output_path, metadata)
            self.metadata_written = True
        if self.frame_count == 0:
            self.started_s = float(now_s)
        timestamp_s = max(0.0, float(now_s) - float(self.started_s))
        self.recorder.add_both(self.frame_count, timestamp_s, result, runtimes)
        return True

    def stop_and_save(self, view_metadata: dict | None = None) -> Path | None:
        if self.state != "recording" or self.recorder is None:
            return self.saved_path
        saved = self.recorder.save()
        self.saved_path = saved
        if saved is not None:
            self.saved_paths.append(saved)
            duration_s = (
                float(self.recorder.rows[-1]["timestamp"])
                - float(self.recorder.rows[0]["timestamp"])
                if self.recorder.rows else 0.0)
            measured_fps = (
                (len(self.recorder.rows) - 1) / duration_s
                if len(self.recorder.rows) >= 2 and duration_s > 0.0
                else float(self.metadata.get("sample_fps", 0.0)))
            updates = {
                "sample_fps": measured_fps,
                "recording": {
                    "frame_count": len(self.recorder.rows),
                    "duration_s": duration_s,
                },
            }
            if view_metadata is not None:
                updates["view"] = view_metadata
            update_session_metadata(self.output_path, updates)
        self.state = "saved"
        return saved


__all__ = [
    "BimanualRecorder",
    "BimanualRecordingSession",
    "JointPositionSmoother",
    "KeypointParquetRecorder",
    "KeypointParquetRecorderWithIMU",
    "SkeletonRenderer",
    "make_hand_result",
]
