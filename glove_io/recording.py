"""Plaintext projection and Parquet writers used by capture applications."""

from __future__ import annotations

import copy
from contextlib import contextmanager, suppress
import math
import os
import tempfile
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


def _vec(value, width: int, dtype=np.float32) -> np.ndarray:
    """One row's wide vector, as its own contiguous 1-D array.

    Rows keep numpy arrays rather than Python lists.  A single bimanual row
    carries ~1600 numbers, which as Python floats is ~45 KB per row (a
    ten-minute take reached ~500 MB) and, worse, had to be walked element by
    element to build the Parquet columns -- that loop is what froze the GUI
    while recording (see ``_series``).  ``copy=True`` because callers pass
    views of buffers they reuse for the next frame.
    """
    return np.array(value, dtype=dtype, copy=True).reshape(width)


@contextmanager
def _atomic_parquet_path(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=".parquet-", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        os.close(fd)
        try:
            yield temporary
        except pl.exceptions.PolarsError as exc:
            raise OSError(f"Cannot write recording to {destination}: {exc}") from exc
        os.replace(temporary, destination)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


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
    """Write bounded batches and publish a complete take on save."""

    _MAX_BATCH_FRAMES = 600

    def __init__(self, output_path, episode_index: int = 0,
                 task_index: int = 0, checkpoint_frames: int = 600):
        if pl is None:
            raise RuntimeError("recording requires polars")
        self.output_path = Path(output_path)
        self.episode_index = int(episode_index)
        self.task_index = int(task_index)
        self.checkpoint_frames = max(0, int(checkpoint_frames))
        self._batch_size = min(
            self.checkpoint_frames or self._MAX_BATCH_FRAMES,
            self._MAX_BATCH_FRAMES)
        self._lock = threading.Lock()
        self._batch_condition = threading.Condition()
        self._buffer: list[dict] = []
        self._pending_rows: list[dict] | None = None
        self._writing = False
        self._staging_dir: Path | None = None
        self._part_count = 0
        self._has_prefix = False
        self._saved_frame_count = 0
        self._frame_count = 0
        self._first_timestamp_s = 0.0
        self._last_timestamp_s = 0.0

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def duration_s(self) -> float:
        with self._lock:
            return self._last_timestamp_s - self._first_timestamp_s

    def add(self, frame_index: int, timestamp_s: float, result: dict) -> None:
        self._append_row(self._make_row(frame_index, timestamp_s, result))

    def _make_row(self, frame_index: int, timestamp_s: float, result: dict) -> dict:
        keypoints2d = np.asarray(result["hands2d"], np.float32).reshape(2, 21, 2)
        hand3d = np.asarray(result["hands3d"], np.float32).reshape(2, 21, 3)
        smoothed = np.asarray(result["smoothed"], np.float32).reshape(2, 21, 3)
        labels = [str(value or "") for value in result["labels"]]
        presents = [bool(value) for value in result["presents"]]
        return {
            "episode_index": self.episode_index,
            "frame_index": int(frame_index),
            "timestamp": np.float32(timestamp_s),
            "task_index": self.task_index,
            "observation.keypoints.stereo_left": _vec(keypoints2d, 84),
            "observation.keypoints.stereo_right": np.zeros(84, np.float32),
            "observation.keypoints.hand_3d": _vec(hand3d, 126),
            "observation.keypoints.reprojection_error": _vec(
                result["reprojection_error"], 2),
            "observation.keypoints.hand_0_present": presents[0],
            "observation.keypoints.hand_1_present": presents[1],
            "observation.keypoints.hand_0_label": labels[0],
            "observation.keypoints.hand_1_label": labels[1],
            "observation.keypoints.stage2": _vec(result["stage2"], 2, dtype=bool),
            "observation.keypoints.propagated": _vec(
                result["propagated"], 2, dtype=bool),
            "action": np.zeros(1, np.float32),
            "observation.keypoints.hand_3d_smoothed": _vec(smoothed, 126),
        }

    @staticmethod
    def _column(rows: list[dict], name, dtype, width: int):
        """One ``Array`` column, stacked in C rather than walked in Python.

        ``np.stack`` + polars' 2-D numpy path is ~17x faster than handing
        polars a list of per-row lists, and -- the part that matters while a
        take is running -- it does not spend the whole conversion holding the
        GIL, so the GUI thread keeps its timeslice.
        """
        return pl.Series(name, np.stack([row[name] for row in rows]),
                         dtype=pl.Array(dtype, width))

    def _series(self, rows: list[dict]):
        def values(name):
            return [row[name] for row in rows]

        return [
            pl.Series("episode_index", values("episode_index"), dtype=pl.Int64),
            pl.Series("frame_index", values("frame_index"), dtype=pl.Int64),
            pl.Series("timestamp", values("timestamp"), dtype=pl.Float32),
            pl.Series("task_index", values("task_index"), dtype=pl.Int64),
            self._column(rows, "observation.keypoints.stereo_left", pl.Float32, 84),
            self._column(rows, "observation.keypoints.stereo_right", pl.Float32, 84),
            self._column(rows, "observation.keypoints.hand_3d", pl.Float32, 126),
            self._column(
                rows, "observation.keypoints.reprojection_error", pl.Float32, 2),
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
            self._column(rows, "observation.keypoints.stage2", pl.Boolean, 2),
            self._column(rows, "observation.keypoints.propagated", pl.Boolean, 2),
            self._column(rows, "action", pl.Float32, 1),
            self._column(
                rows, "observation.keypoints.hand_3d_smoothed", pl.Float32, 126),
        ]

    def _append_row(self, row: dict) -> None:
        timestamp_s = float(row["timestamp"])
        with self._lock, self._batch_condition:
            if len(self._buffer) == self._batch_size:
                # A slow writer frees space when it takes the full buffer.
                # A failed writer leaves both batches in place for retry.
                self._batch_condition.wait_for(
                    lambda: len(self._buffer) < self._batch_size or not self._writing)
                if len(self._buffer) == self._batch_size:
                    self._retry_pending()
                    self._schedule_batch()

            self._buffer.append(row)
            if self._frame_count == 0:
                self._first_timestamp_s = timestamp_s
            self._last_timestamp_s = timestamp_s
            self._frame_count += 1

            if len(self._buffer) == self._batch_size and self._pending_rows is None:
                self._schedule_batch()

    def _take_buffer(self) -> list[dict]:
        rows = self._buffer
        self._pending_rows, self._buffer = rows, []
        return rows

    def _schedule_batch(self) -> None:
        self._take_buffer()
        self._writing = True
        try:
            threading.Thread(
                target=self._write_pending, name="parquet-recording", daemon=True,
            ).start()
        except Exception:
            # Keep the batch for a synchronous retry if the worker cannot start.
            self._writing = False

    def _write_pending(self) -> None:
        with self._batch_condition:
            rows = self._pending_rows
        assert rows is not None
        while True:
            written = False
            try:
                self._commit_batch(rows)
                written = True
            except BaseException:
                # Retry without retaining a traceback and its column buffers.
                pass

            with self._batch_condition:
                if written:
                    self._pending_rows = None
                if written and len(self._buffer) == self._batch_size:
                    rows = self._take_buffer()
                else:
                    self._writing = False
                self._batch_condition.notify_all()
                if not self._writing:
                    return

    def _retry_pending(self) -> None:
        if self._pending_rows is not None:
            self._commit_batch(self._pending_rows)
            self._pending_rows = None

    def _part_path(self, index: int) -> Path:
        assert self._staging_dir is not None
        # Keep unfinished batches out of the replay launcher's *.parquet list.
        return self._staging_dir / f"part-{index:08d}.parquet.part"

    def _commit_batch(self, rows: list[dict], *, final: bool = False) -> None:
        if not self._has_prefix and not self._part_count and (
                self.checkpoint_frames or final):
            self._write_parquet(rows, self.output_path)
            self._has_prefix = True
            return

        if self._staging_dir is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._staging_dir = Path(tempfile.mkdtemp(
                prefix=f".{self.output_path.name}.", suffix=".parts",
                dir=self.output_path.parent.resolve()))
        self._write_parquet(rows, self._part_path(self._part_count))
        self._part_count += 1

    def _write_parquet(self, rows: list[dict], destination: Path) -> None:
        with _atomic_parquet_path(destination) as temporary:
            pl.DataFrame(self._series(rows)).write_parquet(
                temporary, compression="zstd", statistics=True)

    def _publish_parts(self, parts: list[Path]) -> None:
        sources = [self.output_path, *parts] if self._has_prefix else parts
        if len(sources) == 1:
            os.replace(sources[0], self.output_path)
            return

        with _atomic_parquet_path(self.output_path) as temporary:
            pl.scan_parquet(
                sources, parallel="none", low_memory=True, glob=False,
                hive_partitioning=False,
            ).sink_parquet(
                temporary, engine="streaming", maintain_order=True,
                row_group_size=self._MAX_BATCH_FRAMES, compression="zstd", statistics=True)

    def save(self) -> Path | None:
        """Publish all accepted frames, retaining unfinished work on failure."""
        # Hold the public lock while waiting so save has a stable frame set.
        with self._lock, self._batch_condition:
            if self._frame_count == 0:
                return None
            if self._frame_count == self._saved_frame_count:
                return self.output_path

            self._batch_condition.wait_for(lambda: not self._writing)
            self._retry_pending()
            if self._buffer:
                self._commit_batch(self._buffer, final=True)
                self._buffer = []
            parts = [self._part_path(index) for index in range(self._part_count)]
            if parts:
                self._publish_parts(parts)

            # The published file is now the sole prefix for any future add.
            # Forget staging before cleanup so a cleanup failure cannot
            # cause already published parts to be appended again.
            self._has_prefix = True
            self._saved_frame_count = self._frame_count
            staging_dir, self._staging_dir = self._staging_dir, None
            self._part_count = 0
            for part in parts:
                with suppress(OSError):
                    part.unlink(missing_ok=True)
            if staging_dir is not None:
                with suppress(OSError):
                    staging_dir.rmdir()
            return self.output_path


class KeypointParquetRecorderWithIMU(KeypointParquetRecorder):
    TACTILE_DIM = 256

    def __init__(self, output_path, episode_index: int = 0,
                 task_index: int = 0, checkpoint_frames: int = 600,
                 side: str = "right", usb_serial: str = ""):
        super().__init__(output_path, episode_index, task_index, checkpoint_frames)
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

    def add(self, frame_index, timestamp_s, result, imu_quats=None,
            tactile=None, *, raw_imu_quats=None, raw_present_mask=None,
            raw_valid_mask=None, raw_device_timestamp_us=None,
            tactile_raw=None) -> None:
        row = self._make_row(frame_index, timestamp_s, result)
        row[self.IMU_COL] = (_vec(imu_quats, 64) if imu_quats is not None
                             else np.full(64, np.nan, np.float32))
        row[self.TACTILE_COL] = (_vec(tactile, 256) if tactile is not None
                                 else np.full(256, np.nan, np.float32))
        row[self.SERIAL_COL] = self.usb_serial
        row[self.RAW_IMU_COL] = (_vec(raw_imu_quats, 64)
                                 if raw_imu_quats is not None
                                 else np.full(64, np.nan, np.float32))
        row[self.RAW_PRESENT_COL] = (_vec(raw_present_mask, 16, dtype=bool)
                                     if raw_present_mask is not None
                                     else np.zeros(16, bool))
        row[self.RAW_VALID_COL] = (_vec(raw_valid_mask, 16, dtype=bool)
                                   if raw_valid_mask is not None
                                   else np.zeros(16, bool))
        row[self.RAW_TIMESTAMP_COL] = (
            int(raw_device_timestamp_us)
            if raw_device_timestamp_us is not None else -1)
        row[self.RAW_TACTILE_COL] = (_vec(tactile_raw, 256)
                                     if tactile_raw is not None
                                     else np.full(256, np.nan, np.float32))
        self._append_row(row)

    def _series(self, rows: list[dict]):
        series = super()._series(rows)
        series.extend([
            self._column(rows, self.IMU_COL, pl.Float32, 64),
            self._column(rows, self.TACTILE_COL, pl.Float32, 256),
            pl.Series(self.SERIAL_COL,
                      [row[self.SERIAL_COL] for row in rows], dtype=pl.String),
            self._column(rows, self.RAW_IMU_COL, pl.Float32, 64),
            self._column(rows, self.RAW_PRESENT_COL, pl.Boolean, 16),
            self._column(rows, self.RAW_VALID_COL, pl.Boolean, 16),
            pl.Series(self.RAW_TIMESTAMP_COL,
                      [row[self.RAW_TIMESTAMP_COL] for row in rows],
                      dtype=pl.Int64),
            self._column(rows, self.RAW_TACTILE_COL, pl.Float32, 256),
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
        super().__init__(output_path, episode_index, task_index, checkpoint_frames)
        self.serials = dict(serials or {})

    def add_both(self, frame_index, timestamp_s, result, runtimes) -> None:
        self._append_row(self._make_row(frame_index, timestamp_s, result, runtimes))

    def _make_row(self, frame_index, timestamp_s, result, runtimes=None) -> dict:
        row = super()._make_row(frame_index, timestamp_s, result)
        for side in ("left", "right"):
            runtime = runtimes[side] if runtimes is not None else None
            imu = runtime.latest_imu if runtimes is not None else None
            tactile = runtime.latest_tactile if runtimes is not None else None
            imu_key = f"observation.imu.{side}.quaternion"
            tactile_key = f"observation.tactile.{side}_glove"
            row[imu_key] = (_vec(imu, 64) if imu is not None
                            else np.full(64, np.nan, np.float32))
            row[tactile_key] = (_vec(tactile, 256) if tactile is not None
                                else np.full(256, np.nan, np.float32))
            raw_imu = getattr(runtime, "latest_raw_imu", None)
            raw_present = getattr(runtime, "latest_raw_present", None)
            raw_valid = getattr(runtime, "latest_raw_valid", None)
            raw_timestamp = getattr(
                runtime, "latest_raw_device_timestamp_us", None)
            tactile_raw = getattr(runtime, "latest_tactile_raw", None)
            row[f"observation.imu.{side}.raw_physical_quaternion"] = (
                _vec(raw_imu, 64) if raw_imu is not None
                else np.full(64, np.nan, np.float32))
            row[f"observation.imu.{side}.raw_present_mask"] = (
                _vec(raw_present, 16, dtype=bool) if raw_present is not None
                else np.zeros(16, bool))
            row[f"observation.imu.{side}.raw_valid_mask"] = (
                _vec(raw_valid, 16, dtype=bool) if raw_valid is not None
                else np.zeros(16, bool))
            row[f"observation.imu.{side}.device_timestamp_us"] = (
                int(raw_timestamp) if raw_timestamp is not None else -1)
            row[f"observation.tactile.{side}_glove_raw"] = (
                _vec(tactile_raw, 256) if tactile_raw is not None
                else np.full(256, np.nan, np.float32))
            row[f"observation.device.{side}.serial"] = self.serials.get(side, "")
            row[f"observation.imu.{side}.device_fps"] = float(
                getattr(runtime, "device_fps", 0.0) or 0.0)
        return row

    def _series(self, rows: list[dict]):
        series = super()._series(rows)
        for name, dimension in self.EXTRA_DIMS.items():
            series.append(self._column(rows, name, pl.Float32, dimension))
        for name, dimension in self.BOOL_DIMS.items():
            series.append(self._column(rows, name, pl.Boolean, dimension))
        for side in ("left", "right"):
            name = f"observation.imu.{side}.device_timestamp_us"
            series.append(pl.Series(
                name, [row[name] for row in rows], dtype=pl.Int64))
        for side in ("left", "right"):
            name = f"observation.device.{side}.serial"
            series.append(pl.Series(
                name, [row[name] for row in rows], dtype=pl.String))
        for name in self.RATE_COLS:
            series.append(pl.Series(
                name, [row[name] for row in rows], dtype=pl.Float32))
        return series


class BimanualRecordingSession:
    """Repeatable recording lifecycle shared by the live UI and the SDK facade."""

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
        self._stop_view_metadata: dict | None = None
        self.take_index = 0

    @property
    def frame_count(self) -> int:
        return self.recorder.frame_count if self.recorder is not None else 0

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
        if view_metadata is not None:
            self._stop_view_metadata = copy.deepcopy(view_metadata)
        saved = self.recorder.save()
        if saved is not None:
            frame_count = self.recorder.frame_count
            duration_s = self.recorder.duration_s
            measured_fps = (
                (frame_count - 1) / duration_s
                if frame_count >= 2 and duration_s > 0.0
                else float(self.metadata.get("sample_fps", 0.0)))
            updates = {
                "sample_fps": measured_fps,
                "recording": {
                    "frame_count": frame_count,
                    "duration_s": duration_s,
                },
            }
            if self._stop_view_metadata is not None:
                updates["view"] = self._stop_view_metadata
            update_session_metadata(self.output_path, updates)
            self.saved_paths.append(saved)
        self._stop_view_metadata = None
        self.saved_path = saved
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
