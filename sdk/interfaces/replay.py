"""Public recording inspection and MP4 replay interface."""

from __future__ import annotations

from pathlib import Path

from glove_io.replay import ReplayEngine as _Engine
from common.types import RecordingInfo, ReplayResult


class RecordingReplay:
    __slots__ = ("_impl",)

    def __init__(self):
        self._impl = _Engine()

    def inspect(self, input_path: Path | str) -> RecordingInfo:
        return self._impl.inspect(input_path)

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
        return self._impl.render_mp4(
            input_path,
            output_path,
            fps=fps,
            smooth=smooth,
            view=view,
            include_tactile=include_tactile,
            tactile_threshold=tactile_threshold,
            tactile_scale=tactile_scale,
            overwrite=overwrite,
        )
