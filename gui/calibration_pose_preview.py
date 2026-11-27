"""Static wearing instructions and neutral Friston orientation geometry."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
import sys

import cv2
import numpy as np

from gui.friston_display import skeleton_for_side


BUNDLE_ROOT = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
PREVIEW_WIDTH = 660
PREVIEW_HEIGHT = 300
_REFERENCE_DIR = BUNDLE_ROOT / "assets" / "calibration_previews" / "v2_0_8"


class CalibrationPosePreview:
    """Read the public calibration guide images by side and root-checklist index."""

    def __init__(self):
        self._reference_dir = _REFERENCE_DIR
        self._image_cache: dict[tuple[str, int], np.ndarray] = {}
        self._bimanual_image_cache: dict[int, np.ndarray] = {}

    @cached_property
    def _neutral(self) -> dict[str, np.ndarray]:
        return {
            side: skeleton.keypoints_m(skeleton.rest_local_xyzw)
            for side, skeleton in self._skeletons.items()
        }

    @cached_property
    def _neutral_mesh(self) -> dict[str, np.ndarray]:
        return {
            side: skeleton.mesh_vertices_m(skeleton.rest_local_xyzw)
            for side, skeleton in self._skeletons.items()
        }

    @cached_property
    def _mesh_faces(self) -> dict[str, np.ndarray]:
        return {side: skeleton.faces for side, skeleton in self._skeletons.items()}

    @cached_property
    def _skeletons(self):
        return {side: skeleton_for_side(side) for side in ("left", "right")}

    def _reference_path(
            self, kind: str, step_index: int, side: str | None = None) -> Path:
        if kind == "single":
            if side not in ("left", "right"):
                raise ValueError(f"side must be left or right, got {side!r}")
            name = f"single_{side}_step_{step_index:02d}.png"
        elif kind == "dual":
            name = f"dual_step_{step_index:02d}.png"
        else:
            raise ValueError(f"unknown reference kind: {kind!r}")
        return self._reference_dir / name

    def _load_reference_image(
            self, kind: str, step_index: int, side: str | None = None
    ) -> np.ndarray:
        path = self._reference_path(kind, step_index, side)
        if not path.is_file():
            raise FileNotFoundError(f"Missing calibration preview image: {path}")
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                f"cannot decode frozen calibration reference: {path}")
        expected = (PREVIEW_HEIGHT, PREVIEW_WIDTH, 3)
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(
                f"invalid frozen calibration reference {path}: "
                f"expected uint8 {expected}, got {image.dtype} {image.shape}")
        return np.ascontiguousarray(image)

    def render_bgr(self, side: str, step_index: int) -> np.ndarray:
        side = str(side).lower()
        if side not in ("left", "right"):
            raise ValueError(f"side must be left or right, got {side!r}")
        index = max(0, min(int(step_index), 14))
        key = (side, index)
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached.copy()
        image = self._load_reference_image("single", index, side)
        self._image_cache[key] = image
        return image.copy()

    def render_bimanual_bgr(self, step_index: int) -> np.ndarray:
        """Return a frozen left/right full-hand reference panel."""
        index = max(0, min(int(step_index), 14))
        cached = self._bimanual_image_cache.get(index)
        if cached is not None:
            return cached.copy()
        canvas = self._load_reference_image("dual", index)
        self._bimanual_image_cache[index] = canvas
        return canvas.copy()
