"""Plaintext Parquet inspection and MP4 replay adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from gui.rendering import replay as replay_application
from glove_io.session import (
    load_session_metadata, session_metadata_path)
from common.errors import ReplayError
from common.types import RecordingInfo, ReplayResult


class ReplayEngine:
    def inspect(self, input_path: Path | str) -> RecordingInfo:
        path = Path(input_path).resolve()
        if not path.is_file():
            raise ReplayError(f"recording does not exist: {path}")
        try:
            frame = pl.read_parquet(path)
            metadata = load_session_metadata(path)
        except Exception as exc:
            raise ReplayError(str(exc)) from exc
        columns = set(frame.columns)
        tactile_columns = [
            name for name in columns
            if name in {
                "observation.tactile.left_glove",
                "observation.tactile.right_glove",
            }]
        raw_tactile_columns = [
            name for name in columns if name in {
                "observation.tactile.left_glove_raw",
                "observation.tactile.right_glove_raw",
            }]
        raw_imu_columns = [
            name for name in columns
            if name.endswith(".raw_physical_quaternion")]
        tactile_valid_frames = 0
        tactile_nonzero_samples = 0
        for column in tactile_columns:
            values = np.asarray(
                frame[column].to_numpy(), dtype=np.float32).reshape(
                    frame.height, -1)
            tactile_valid_frames += int(
                np.isfinite(values).any(axis=1).sum())
            tactile_nonzero_samples += int(
                np.count_nonzero(np.nan_to_num(values)))
        sides = []
        if "observation.keypoints.hand_0_present" in columns:
            if bool(frame["observation.keypoints.hand_0_present"].any()):
                sides.append("left")
        if "observation.keypoints.hand_1_present" in columns:
            if bool(frame["observation.keypoints.hand_1_present"].any()):
                sides.append("right")
        if not sides:
            values = np.asarray(
                frame["observation.keypoints.hand_3d"].to_list(),
                dtype=np.float32).reshape(-1, 2, 21, 3)
            if np.isfinite(values[:, 0]).any():
                sides.append("left")
            if np.isfinite(values[:, 1]).any():
                sides.append("right")
        session = session_metadata_path(path)
        return RecordingInfo(
            parquet_path=path,
            session_path=session if session.is_file() else None,
            mode=str(metadata.get(
                "mode", "bimanual" if len(sides) == 2 else "single")),
            sides=tuple(sides),
            frame_count=frame.height,
            sample_fps=float(metadata.get("sample_fps", 30.0)),
            has_imu=any(name.startswith("observation.imu.")
                        for name in columns),
            has_tactile=bool(tactile_columns),
            tactile_valid_frames=tactile_valid_frames,
            tactile_nonzero_samples=tactile_nonzero_samples,
            has_raw_imu=bool(raw_imu_columns),
            has_raw_tactile=bool(raw_tactile_columns),
        )

    def render_mp4(
        self,
        input_path: Path | str,
        output_path: Path | str | None = None,
        *,
        fps: float | None = None,
        smooth: bool = False,
        view: str | None = None,
        include_tactile: bool = True,
        tactile_threshold: float = 0.0,
        tactile_scale: float = 0.6,
        overwrite: bool = False,
    ) -> ReplayResult:
        info = self.inspect(input_path)
        output = (Path(output_path).resolve() if output_path is not None
                  else info.parquet_path.parent / "preview.mp4")
        if output.exists() and not overwrite:
            raise ReplayError(f"replay output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        argv = [str(info.parquet_path), "--out", str(output)]
        if fps is not None:
            if float(fps) <= 0.0:
                raise ValueError("fps must be positive")
            argv.extend(["--fps", str(float(fps))])
        if smooth:
            argv.append("--smooth")
        if not include_tactile:
            argv.append("--no-tactile")
        if tactile_threshold:
            argv.extend(["--tactile-threshold", str(float(tactile_threshold))])
        if float(tactile_scale) <= 0.0:
            raise ValueError("tactile_scale must be positive")
        argv.extend(["--tactile-scale", str(float(tactile_scale))])
        if view is not None:
            if view not in {"static", "rotating"}:
                raise ValueError("view must be static or rotating")
            argv.extend(["--view", view])
        try:
            result = replay_application.main(argv)
        except SystemExit as exc:
            raise ReplayError(str(exc)) from exc
        except Exception as exc:
            raise ReplayError(str(exc)) from exc
        if result not in (None, 0) or not output.is_file():
            raise ReplayError(f"replay did not create output: {output}")
        return ReplayResult(
            input_path=info.parquet_path,
            output_path=output,
            frame_count=info.frame_count,
            sample_fps=float(fps if fps is not None else info.sample_fps),
        )
