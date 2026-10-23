#!/usr/bin/env python3
"""STM32 glove -> 21 solved joints -> live 3D view, through the client SDK.

Everything arrives from the one SDK entry point: ``sdk.open_glove`` ->
``Glove.read`` -> :class:`sdk.HandFrame`, which hands over the 21 solved
keypoints (metres), finger force (newtons) and the link diagnostics -- and
nothing raw.  The display is the spherical-coordinate perspective camera from
``render_21_points.py``: fingers colored by BGR, brightness by depth, z=0
ground grid + RGB axes, with keyboard live view rotation and saving (view
parameters stored in ``view_config_3d.json``, so config.json and the lite
script's view_config.json are not touched).

Data source: whichever link carries this hand -- ``open_glove`` is asked for no
particular one, so the wired USB cable and the Bluetooth dongle are equally
valid and are told apart inside :mod:`sdk.transport`.

View interaction: the camera orbits the origin while the ground grid, RGB axes
and the hand rotate with the scene:
    on-screen direction pad or middle-drag: yaw/elevation orbit
    drag the on-screen outer ring: roll (about the view axis)
    wheel: zoom dist | left-drag: pan
    e/c: roll (about the view axis) | a/d/w/s: fine-tune yaw/elev
    r: reset view | v: save view | q/ESC: quit and save
Tactile overlay: fixed 2D panel in the lower-left corner, fed straight from the
frame's newton force matrix; it does not rotate with the view and the [ / ] keys
resize it.
On exit (including Ctrl+C) the view is saved.

Dependencies match the lite version: numpy scipy opencv-python PySide6.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys
import time
import types
import warnings

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

if getattr(sys, "frozen", False):
    BUNDLE_ROOT = Path(sys._MEIPASS)
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    BUNDLE_ROOT = PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from gui import APP_VERSION  # noqa: E402
from gui.calibration_selector import select_calibration_files  # noqa: E402
from gui.qt_viewer import (  # noqa: E402
    QtCanvas, ensure_qapp, pump_events)
from gui.rendering.tactile import render_hand  # noqa: E402
from common.frame_pacing import BURST_SPAN_US  # noqa: E402
from common.i18n import L, is_en, set_lang  # noqa: E402
from common.types import LINK_USB  # noqa: E402
from sdk import (  # noqa: E402
    CalibrationError, HandFrame, StreamTimeoutError, get_version, open_glove)
# The device manager is not on the client SDK's public surface; resolving an
# explicit ``--serial-port`` reaches it through :mod:`sdk.device`.
from sdk.device import DeviceManager  # noqa: E402

DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "glove_devices.json"

# View parameters are stored here (display only; calibration is unaffected).
# Kept separate from the lite script's view_config.json: spherical
# (yaw/elev/dist) and orthogonal (euler/ppm) are two distinct parameter spaces
# and would overwrite each other if shared.
VIEW3D_CONFIG_PATH = PROJECT_ROOT / "config" / "view_config_3d.json"

# --------------------------------------------------------------------------
# Rendering (inlined from render_21_points.py; numpy+cv2 only, no D435/stereo_s80m deps)
# --------------------------------------------------------------------------
N_HANDS = 2
# W/H are the *design baseline*, not the live canvas size.  The live viewer
# rasterises at the window's own pixel size (see gui/qt_viewer.py's
# canvas_pixel_size_for), so every absolute pixel value in the drawing code --
# font scales, thicknesses, card sizes, margins -- stays quoted at this
# baseline and does not scale with the canvas.  Only edge- and centre-relative
# anchors follow the live size.
#
# They must stay module constants: calibration_pose_preview builds Camera()
# directly at ten sites and relies on this default, and the offline replay
# writer has its own separate pair.
W, H = 1280, 720
FOV_DEG = 50.0                                   # vertical field of view
# How far the fixed 2D tactile overlay may be scaled: the keyboard steps and
# the drag-to-resize handles both stay inside this range.
TACTILE_SCALE_MIN, TACTILE_SCALE_MAX = 0.3, 2.0

# 21-point skeleton bones + groups: 0 thumb 1 index 2 middle 3 ring 4 pinky 5 metacarpal
BONES = [
    (0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),                # thumb
    (0, 5, 1), (5, 6, 1), (6, 7, 1), (7, 8, 1),                # index
    (0, 9, 2), (9, 10, 2), (10, 11, 2), (11, 12, 2),           # middle
    (0, 13, 3), (13, 14, 3), (14, 15, 3), (15, 16, 3),         # ring
    (0, 17, 4), (17, 18, 4), (18, 19, 4), (19, 20, 4),         # pinky
    (5, 9, 5), (9, 13, 5), (13, 17, 5),                        # metacarpal
]
# finger group colors (BGR)
FINGER_BGR = [
    (60, 80, 255),     # 0 thumb  red
    (60, 255, 255),    # 1 index  yellow
    (60, 220, 60),     # 2 middle green
    (255, 210, 60),    # 3 ring   cyan
    (255, 120, 255),   # 4 pinky  purple
    (210, 175, 150),   # 5 metacarpal light steel blue (readable on dark blue background)
]
HAND_MESH_BGR = (215, 200, 185)


# --------------------------------------------------------------------------
# Text rendering: cv2 Hershey fonts are ASCII-only, so CJK (Chinese) strings
# are drawn through Qt (QPainter + system font fallback).  Qt is already a
# hard dependency of the live viewer and renders CJK without extra packages.
# --------------------------------------------------------------------------


def _qt_text_patch(text: str, font_px: int, color, thickness: int):
    """Render `text` with Qt into an (RGBA uint8 HxWx4 patch, ascent) pair."""
    from PySide6.QtCore import QRect, Qt as _Qt
    from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter
    from PySide6.QtWidgets import QApplication
    font = QFont(QApplication.font())
    font.setPixelSize(int(font_px))
    if int(thickness) >= 2:
        font.setBold(True)
    metrics = QFontMetrics(font)
    w = metrics.horizontalAdvance(text)
    if w <= 0:
        return None
    ascent = metrics.ascent()
    h = metrics.height()
    if h <= 0:
        return None
    qimg = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    qimg.fill(_Qt.transparent)
    painter = QPainter(qimg)
    painter.setFont(font)
    painter.setPen(QColor(int(color[2]), int(color[1]), int(color[0])))
    painter.drawText(QRect(0, 0, w, h), _Qt.AlignLeft | _Qt.AlignTop, text)
    painter.end()
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    buf = qimg.constBits()
    rgba = np.ndarray(
        shape=(h, w, 4), dtype=np.uint8, buffer=buf,
        strides=(qimg.bytesPerLine(), 4, 1)).copy()
    return rgba, ascent


def _put_text_cjk(img: np.ndarray, text: str, org, scale: float, color,
                  thickness: int) -> None:
    """Draw a CJK string via Qt and blend it onto the BGR canvas at a cv2-style baseline origin."""
    patch = _qt_text_patch(
        text, max(9, int(round(scale * 28.0))), color, thickness)
    if patch is None:
        return
    rgba, ascent = patch
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)[..., ::-1]  # RGBA -> BGR
    h, w = rgba.shape[:2]
    # cv2 org is the baseline; Qt draws with the baseline `ascent` px from the top.
    x, y = int(org[0]), int(org[1]) - ascent
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    region = img[y0:y1, x0:x1].astype(np.float32)
    px0, py0 = x0 - x, y0 - y
    sub_alpha = alpha[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]
    sub_rgb = rgb[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]
    img[y0:y1, x0:x1] = (region * (1.0 - sub_alpha) + sub_rgb * sub_alpha).astype(np.uint8)


def put_text(img: np.ndarray, text: str, org, scale: float, color,
             thickness: int, line_type: int = cv2.LINE_AA) -> None:
    """Draw text on a BGR canvas; routes CJK through Qt (cv2 Hershey is ASCII-only)."""
    if not text:
        return
    if text.isascii():
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, line_type)
    else:
        _put_text_cjk(img, text, org, scale, color, thickness)


def text_size(text: str, scale: float, thickness: int) -> tuple[int, int]:
    """(width, height) matching put_text's output, for centering text."""
    if not text:
        return 0, 0
    if text.isascii():
        return cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    from PySide6.QtGui import QFont, QFontMetrics
    from PySide6.QtWidgets import QApplication
    font = QFont(QApplication.font())
    font.setPixelSize(max(9, int(round(scale * 28.0))))
    if int(thickness) >= 2:
        font.setBold(True)
    metrics = QFontMetrics(font)
    return metrics.horizontalAdvance(text), metrics.height()


class Camera:
    """Spherical-coordinate camera looking at the origin, perspective projection.

    pan_px is the image-space offset from drag-panning.  The three rotations
    are: yaw about the world vertical axis, elev about the horizontal axis,
    and roll about the view (fwd) axis.

    size is the canvas this camera projects into.  It defaults to the design
    baseline so offscreen callers (calibration_pose_preview, the replay
    writer) are unaffected; the live viewer passes its current canvas.
    """

    def __init__(self, yaw_deg, elev_deg, dist, roll_deg=0.0, pan_px=(0.0, 0.0),
                 size: tuple[int, int] = (W, H)):
        self._w, self._h = int(size[0]), int(size[1])
        yaw, elev = np.deg2rad(yaw_deg), np.deg2rad(elev_deg)
        roll = np.deg2rad(roll_deg)
        cp = np.array([dist * np.cos(elev) * np.sin(yaw),
                       dist * np.sin(elev),
                       dist * np.cos(elev) * np.cos(yaw)])
        self.pos = cp
        self.fwd = -cp / np.linalg.norm(cp)
        self.right = np.cross(self.fwd, [0, 1, 0])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        # Roll about the view axis: rotate the image-plane basis by roll
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        r0, u0 = self.right, self.up
        self.right = r0 * cos_r + u0 * sin_r
        self.up = -r0 * sin_r + u0 * cos_r
        # Focal length in pixels: scaling it with the canvas height is what
        # makes the 3D content keep the same *fraction* of the frame at any
        # window size, and therefore what makes the view resolution-independent.
        self.f = (self._h / 2) / np.tan(np.deg2rad(FOV_DEG) / 2)
        self.pan_px = [float(pan_px[0]), float(pan_px[1])]

    def project(self, pts3d):
        """(N,3) -> (N,2) image coordinates + (N,) depth (positive in front of camera)."""
        v = pts3d - self.pos
        z = v @ self.fwd
        x = v @ self.right
        y = v @ self.up
        u = self.f * x / z + self._w / 2
        vv = self._h / 2 - self.f * y / z
        # pan_px stays in raw canvas pixels and is deliberately NOT normalised
        # by the canvas size: both the centre above and the offset here are in
        # the same space, so resizing the window leaves the scene exactly where
        # it looks like it is instead of drifting outward as the window grows.
        # The cost is cosmetic -- a pan saved at one window size is a different
        # fraction of the frame at another -- and pan is a manual affordance
        # that defaults to (0, 0).
        u += self.pan_px[0]
        vv += self.pan_px[1]
        return np.stack([u, vv], -1), z


def hand_display_basis(joints: np.ndarray) -> np.ndarray:
    """Return [thumb-side, forward, normal] basis from wrist/MCP geometry."""
    hand = np.asarray(joints, dtype=float).reshape(21, 3)
    wrist = hand[0]
    forward = hand[[5, 9, 13, 17]].mean(axis=0) - wrist
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    thumb_side = hand[5] - hand[17]  # pinky MCP -> index MCP (thumb side)
    thumb_side -= forward * float(thumb_side @ forward)
    thumb_side /= max(float(np.linalg.norm(thumb_side)), 1e-9)
    normal = np.cross(thumb_side, forward)
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    return np.stack([thumb_side, forward, normal], axis=1)


def align_left_to_right_reference(
        left_reference: np.ndarray, right_reference: np.ndarray) -> R:
    """Align left display to the tuned right pose, with thumb sides mirrored.

    Right-hand geometry and camera are untouched.  The desired left forward
    axis equals the right forward axis, while its thumb-side axis is reversed;
    this puts both thumbs inside and both pinkies outside in an egocentric view.
    Because left/right anatomy has opposite handedness, this also means the
    side-aware palm-facing directions match: -left local normal == right local
    normal.  No extra 180-degree display roll is needed.
    """
    left_basis = hand_display_basis(left_reference)
    right_basis = hand_display_basis(right_reference)
    target = right_basis.copy()
    target[:, 0] *= -1.0
    target[:, 2] = np.cross(target[:, 0], target[:, 1])
    target[:, 2] /= max(float(np.linalg.norm(target[:, 2])), 1e-9)
    matrix = target @ left_basis.T
    if np.linalg.det(matrix) < 0.0:
        raise ValueError("left-to-right display alignment produced a reflection")
    return R.from_matrix(matrix)


def apply_hand_display_rotation(joints: np.ndarray, rotation: R) -> np.ndarray:
    hand = np.asarray(joints, dtype=np.float32).reshape(21, 3)
    wrist = hand[0].copy()
    return (rotation.apply(hand - wrist) + wrist).astype(np.float32)


def place_hands_at_wrist_anchors(
        hands: np.ndarray, horizontal_axis: np.ndarray | None = None,
        separation_m: float = 0.30,
        anchor_axis: np.ndarray | None = None) -> np.ndarray:
    """Place left/right 21-point sets at fixed palm-relative wrist anchors.

    Wrist placement is deliberately independent of the camera.  The left-to-
    right wrist axis is the opposite of the displayed right hand's thumb-side
    axis, so both thumbs point inward.  That lateral axis is perpendicular to
    both the common finger-forward and side-aware palm-facing directions: the
    wrists therefore have equal height in the shared palm coordinate system.
    ``horizontal_axis`` is retained only for compatibility and is ignored.
    ``anchor_axis`` may provide a fixed session lateral axis; otherwise the
    current displayed right-hand lateral axis is used.

    Only one translation per hand is applied, so every within-hand distance
    and the original computed pose are preserved.
    """
    slots = np.asarray(hands, dtype=np.float32).reshape(2, 21, 3).copy()
    del horizontal_axis
    axis = (np.asarray(anchor_axis, dtype=float).reshape(3).copy()
            if anchor_axis is not None
            else -hand_display_basis(slots[1])[:, 0].copy())
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("right-hand lateral axis is degenerate")
    axis /= norm
    anchors = np.stack([
        -axis * float(separation_m) * 0.5,
        axis * float(separation_m) * 0.5,
    ]).astype(np.float32)
    for slot in range(2):
        slots[slot] += anchors[slot] - slots[slot, 0]
    return slots


def _bg(w: int = W, h: int = H) -> np.ndarray:
    # Dark navy-blue vertical gradient in the style of modelling software.
    # Still a single vectorized row op, so the palette change costs no frame time.
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)
    top = np.array([54, 40, 30], np.float32)      # BGR: upper edge
    bottom = np.array([26, 17, 13], np.float32)   # BGR: lower edge
    grad = (top[None, :] + (bottom - top)[None, :] * t[:, None]).astype(np.uint8)
    img = np.empty((h, w, 3), np.uint8)
    img[...] = grad[:, None, :]
    return img


def _draw_hand(img, cam, kpts, hand_idx) -> None:
    pts, z = cam.project(kpts)                       # (21,2) + (21,)
    vis = np.isfinite(kpts).all(axis=1) & (z > 0)
    # Depth -> brightness factor (near bright, far dim); fingers slightly brighter than metacarpal
    zmin, zmax = 0.05, 0.6
    lum = np.clip((zmax - z) / (zmax - zmin), 0.35, 1.0)
    for a, b, g in BONES:
        if vis[a] and vis[b]:
            k = 0.5 * (lum[a] + lum[b])
            col = tuple(int(c * k) for c in FINGER_BGR[g])
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     col, 3, cv2.LINE_AA)
    for p, ok, g in zip(pts, vis, range(21)):
        if ok:
            cv2.circle(img, tuple(p.astype(int)), 2, (245, 248, 255), -1, cv2.LINE_AA)


class _PicklePlaceholder:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


def _load_mano_faces(side: str) -> np.ndarray | None:
    side = str(side).lower()
    cached = getattr(_load_mano_faces, "_cache", {})
    if side in cached:
        return cached[side]
    faces = None
    neutral_mesh = BUNDLE_ROOT / "mano_neutral_meshes.npz"
    if neutral_mesh.is_file():
        try:
            with np.load(neutral_mesh, allow_pickle=False) as archive:
                faces = np.asarray(archive[f"{side}_faces"], dtype=np.int32)
        except (OSError, KeyError, ValueError):
            faces = None
    if faces is None:
        model_path = BUNDLE_ROOT / "assets/hand/models" / f"MANO_{side.upper()}.pkl"
        if model_path.is_file():
            old_modules = {
                name: sys.modules.get(name)
                for name in ("chumpy", "chumpy.reordering", "chumpy.ch")
            }
            try:
                sys.modules["chumpy"] = types.ModuleType("chumpy")
                sys.modules["chumpy.reordering"] = types.ModuleType("chumpy.reordering")
                sys.modules["chumpy.ch"] = types.ModuleType("chumpy.ch")
                sys.modules["chumpy.reordering"].Select = _PicklePlaceholder
                sys.modules["chumpy.ch"].Ch = _PicklePlaceholder
                with model_path.open("rb") as handle:
                    payload = pickle.load(handle, encoding="latin1")
                faces = np.asarray(payload["f"], dtype=np.int32)
            except Exception as exc:
                print(f"[View warning] Cannot load MANO faces: {exc}", file=sys.stderr)
                faces = None
            finally:
                for name, module in old_modules.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    cached = dict(cached)
    cached[side] = faces
    _load_mano_faces._cache = cached
    return faces


def _draw_hand_mesh(img, cam, vertices, faces) -> bool:
    if vertices is None or faces is None:
        return False
    verts = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if verts.shape != (778, 3) or not np.isfinite(verts).all():
        return False
    pts, z = cam.project(verts)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(z) & (z > 0.0)
    if not finite.any():
        return False
    camera_pts = np.stack([
        (verts - cam.pos) @ cam.right,
        (verts - cam.pos) @ cam.up,
        z,
    ], axis=1)
    face_depth = z[faces].mean(axis=1)
    face_ok = finite[faces].all(axis=1)
    face_camera_pts = camera_pts[faces]
    face_normals = np.cross(
        face_camera_pts[:, 1] - face_camera_pts[:, 0],
        face_camera_pts[:, 2] - face_camera_pts[:, 0],
    )
    face_normal_lengths = np.linalg.norm(face_normals, axis=1)
    face_shades = (
        0.72
        + 0.28 * np.abs(face_normals[:, 2])
        / np.maximum(face_normal_lengths, 1e-9)
    )
    face_colors = (
        face_shades[:, None] * np.asarray(HAND_MESH_BGR, np.float64)[None, :]
    ).astype(np.int32)
    projected_faces = np.round(pts[faces]).astype(np.int32)
    # Precompute colours once per frame instead of tuple()/int() per face.
    color_list = [tuple(int(c) for c in rgb) for rgb in face_colors]
    # Use LINE_8, not LINE_AA.  The MANO mesh is 1538 small triangles, so the
    # anti-aliased fill re-samples every edge pixel ~16x and turns ~3000 fills
    # per frame (two hands) into a 5-10x slowdown, dropping the 60 Hz preview
    # to ~15-33 FPS.  Plain fill keeps the mesh at the display rate with no
    # visible loss on a hand that is re-rendered every frame.
    for face_index in np.argsort(face_depth)[::-1]:
        if not face_ok[face_index]:
            continue
        cv2.fillConvexPoly(
            img, projected_faces[face_index], color_list[face_index], cv2.LINE_8)
    return True


def render_frame_live(
        cam, kpts_slots, hud_lines, title, grid_r=None,
        grid_basis: np.ndarray | None = None,
        mesh_slots: list[np.ndarray | None] | None = None,
        mesh_faces: np.ndarray | None = None,
        display_mode: str = "bones",
        size: tuple[int, int] = (W, H)) -> np.ndarray:
    """Render one live frame: background + ground grid + left/right hand slots + HUD.

    grid_r: grid/axes radius in metres.  When None it is auto-fitted to the
    current frame's hand scale (for offline batch use); live mode must pass a
    fixed value (scaled with camera dist), otherwise the grid zooms along with
    hand motion and looks like "auto-zoom".

    size: the canvas to rasterise into, which must be the size the camera was
    built for.  Defaults to the design baseline so offscreen callers are
    unaffected.  Only the background takes it: the title and HUD keep their
    absolute pixel size and offsets, because they are UI chrome rather than
    content -- everything else in the interface (the orbit pad, the baseline
    banner, the tactile cards) is fixed-size too, and on a high-DPI display a
    size that grew with the canvas would render far too large.
    """
    w, h = int(size[0]), int(size[1])
    img = _bg(w, h)
    # The RGB world axes (and the hidden ground lattice) are not drawn: the
    # live background is plain, only the hands and HUD are rendered.
    display_mode = display_mode if display_mode in ("bones", "hand") else "bones"
    for slot in range(N_HANDS):
        mesh_drawn = False
        if display_mode == "hand" and mesh_slots is not None:
            mesh_drawn = _draw_hand_mesh(img, cam, mesh_slots[slot], mesh_faces)
        if display_mode == "bones" or not mesh_drawn:
            _draw_hand(img, cam, kpts_slots[slot], slot)
    if title:
        put_text(img, title, (20, 34), 0.9, (255, 246, 240), 2)
    # Without an in-canvas title the HUD starts where the title used to be,
    # so the remaining lines fill that space instead of leaving a gap.
    top = 66 if title else 34
    for line in hud_lines:
        put_text(img, line, (20, top), 0.55, (225, 205, 185), 1)
        top += 24
    return img


# --------------------------------------------------------------------------
# View state: spherical camera (yaw/elev/roll + zoom dist + pan pan_px),
# saveable/loadable/resettable.  The camera orbits the origin while the ground
# grid, RGB axes and hand rotate with the scene.
# --------------------------------------------------------------------------
class LiveViewState:
    DEFAULTS = {"yaw": 180.0, "elev": 35.0, "roll": 0.0, "dist": 0.5,
                "pan_px": [0.0, 0.0]}
    DIST_MIN, DIST_MAX = 0.1, 5.0
    _FLOAT_KEYS = ("yaw", "elev", "roll", "dist")

    def __init__(self, path: str | os.PathLike = VIEW3D_CONFIG_PATH):
        self.path = Path(path)
        self.reset()
        self.load()

    def reset(self):
        for key, value in self.DEFAULTS.items():
            setattr(self, key, list(value) if isinstance(value, list) else float(value))

    def load(self) -> "LiveViewState":
        if not self.path.exists():
            return self
        try:
            import json
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for key in self._FLOAT_KEYS:
                if (isinstance(data.get(key), (int, float))
                        and np.isfinite(float(data[key]))):
                    value = float(data[key])
                    if key == "dist":
                        value = float(np.clip(value, self.DIST_MIN, self.DIST_MAX))
                    setattr(self, key, value)
            pan = data.get("pan_px")
            if (isinstance(pan, list) and len(pan) == 2
                    and all(np.isfinite(float(v)) for v in pan)):
                self.pan_px = [float(pan[0]), float(pan[1])]
        except (ValueError, OSError) as exc:
            print(f"[View warning] Cannot parse view configuration; using defaults: {exc}",
                  file=sys.stderr)
        return self

    def save(self) -> Path:
        import json
        data = {key: float(getattr(self, key)) for key in self._FLOAT_KEYS}
        data["pan_px"] = [float(self.pan_px[0]), float(self.pan_px[1])]
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.path


class Live3DViewer:
    """Live viewer with a spherical camera (grid + axes + hand rotate together).

    Middle-drag: 360° free rotation (horizontal -> yaw, vertical -> elev)
    wheel: zoom dist | left-drag: pan
    e/c: roll (about the view axis) | a/d/w/s: fine-tune yaw/elev
    r: reset view | v: save view | q/ESC: quit and save
    With --view rotating, yaw auto-rotates at --yaw-rate (can be combined with
    middle-drag rotation).  The view affects display only.
    """

    # Right-drag rotation sensitivity: dragging the full canvas width ~= 360°
    # yaw.  `_refresh_canvas_size` shadows this per instance once the real
    # canvas size is known; the class value is the design-baseline default.
    ORBIT_DEG_PER_PX = 360.0 / W
    MODEL_CONTROL_SENSITIVITIES = (
        ("fine", 3.0, 0.75),
        ("standard", 5.0, 1.25),
        ("fast", 8.0, 2.0),
    )

    def __init__(self, view_state: LiveViewState, initial_slots: np.ndarray,
                 title: str, rotating: bool = False, yaw_rate: float = 60.0,
                 hand_side: str = "right",
                 display_rotations: list[R | None] | None = None,
                 show_title_in_canvas: bool = True,
                 display_mode: str = "hand",
                 initial_mesh_vertices: np.ndarray | None = None,
                 redraw_interval_s: float = 0.0,
                 show_baseline: bool = False,
                 show_surface_color: bool = False,
                 tactile_panel_modes: list[tuple[str, str]] | None = None,
                 tactile_panel_mode: str | None = None,
                 show_tactile_toggle: bool = True,
                 show_orientation_button: bool = False,
                 show_selector_button: bool = False,
                 show_dongle_reboot_button: bool = False,
                 show_pressure_fit_buttons: bool = False):
        self.view = view_state
        # The reset target is the state with which this particular window was
        # opened (including values loaded from its view JSON), not hard-coded
        # factory defaults.
        self._initial_view = {
            "yaw": float(view_state.yaw),
            "elev": float(view_state.elev),
            "roll": float(view_state.roll),
            "dist": float(view_state.dist),
            "pan_px": [float(view_state.pan_px[0]), float(view_state.pan_px[1])],
        }
        # The on-screen pad transforms only the hand geometry.  Camera/grid/
        # world axes remain unchanged.  All UI increments use the fixed axes
        # actually drawn by this viewer.
        self._model_control_rotation = R.identity()
        self._model_control_sensitivity_index = 1
        self.title = f"{title} V{APP_VERSION}"
        self._show_title_in_canvas = bool(show_title_in_canvas)
        self.rotating = bool(rotating)
        self.yaw_rate = float(yaw_rate)
        self.hand_side = str(hand_side).strip().lower()
        if self.hand_side not in ("left", "right"):
            raise ValueError("hand_side must be left or right")
        self.hand_slot = 0 if self.hand_side == "left" else 1
        self.display_mode = display_mode if display_mode in ("bones", "hand") else "bones"
        self._mesh_faces = _load_mano_faces(self.hand_side)
        self.exit_requested = False
        # Set by the "back to calibration selector" button; the owning live
        # session polls this and returns to the calibration-file picker.
        self.request_selector = False
        self.fps = 0.0
        # The two device streams' own arrival rates, both carried on the frame
        # and both a separate number from the solve rate above: attitude and
        # pressure arrive independently, so one cannot stand in for the other.
        self.device_fps = 0.0
        self.tactile_fps = 0.0
        self.frame_count = 0
        self.missing_labels: list[str] = []
        self.safety_intervened = False
        self.active_contact: str | None = None
        self._kpts_slots = np.asarray(initial_slots, np.float32).reshape(2, 21, 3)
        self._mesh_slots: list[np.ndarray | None] = [None, None]
        if initial_mesh_vertices is not None:
            mesh = np.asarray(initial_mesh_vertices, dtype=np.float32)
            if mesh.shape == (778, 3) and np.isfinite(mesh).all():
                self._mesh_slots[self.hand_slot] = mesh
        self._display_rotations = list(display_rotations or [None, None])
        if len(self._display_rotations) != N_HANDS:
            raise ValueError("display_rotations must contain left/right entries")
        self._redraw_requested = True
        # Minimum seconds between Qt canvas repaints; 0 means redraw as soon as
        # new data arrives (the caller usually already gates updates to the
        # display FPS, so this is an optional extra throttle).
        self._redraw_interval_s = max(0.0, float(redraw_interval_s))
        self._last_redraw_s: float | None = None
        self._drag_start: tuple[int, int] | None = None
        self._pan_at_drag: list[float] = [0.0, 0.0]
        self._orbit_start: tuple[int, int] | None = None
        self._yaw_at_orbit: float = 0.0
        self._elev_at_orbit: float = 0.0
        self._last_tick_s: float | None = None
        # Tactile overlay: 16x16 finger-force matrix in newtons, fixed 2D panel in the lower-left corner (does not rotate with the view)
        self.tactile_frame: np.ndarray | None = None
        self.tactile_scale = 0.6
        # Live canvas size.  Seeded with the design baseline so the first _draw
        # below is exactly what it always was; _refresh_canvas_size() then
        # takes over from the widget's real geometry.
        self.canvas_w, self.canvas_h = W, H

        ensure_qapp()
        # Window title bar carries host + SDK versions (e.g.
        # "Stouch Glove V1.2.0 (SDK v1.2.0)"); the in-canvas HUD title is
        # drawn only when show_title_in_canvas.
        self._qcanvas = QtCanvas(
            self, f"{self.title} (SDK v{get_version()})", W, H,
            display_modes=self._display_mode_options(),
            display_mode=self.display_mode,
            show_baseline=show_baseline,
            show_surface_color=show_surface_color,
            tactile_panel_modes=tactile_panel_modes,
            tactile_panel_mode=tactile_panel_mode,
            show_tactile_toggle=show_tactile_toggle,
            show_orientation_button=show_orientation_button,
            show_selector_button=show_selector_button,
            show_dongle_reboot_button=show_dongle_reboot_button,
            show_pressure_fit_buttons=show_pressure_fit_buttons)
        self._refresh_canvas_size()
        self._canvas = self._draw()
        self._qcanvas.set_canvas(self._canvas)

    def _refresh_canvas_size(self) -> bool:
        """Adopt the window's current pixel size as the render canvas size.

        Polled from tick() rather than pushed from Qt's resize event: the size
        is a cached getter so this is free at the tick rate, and it also covers
        the first frame and a devicePixelRatio change, neither of which is
        guaranteed to arrive as a resize of this widget.

        Returns True when the size changed, which means a redraw is owed --
        without that, an idle glove stream would leave the window showing the
        previous frame stretched by Qt's fit-to-window blit.
        """
        size = self._qcanvas.canvas_pixel_size()
        if size == (self.canvas_w, self.canvas_h):
            return False
        self.canvas_w, self.canvas_h = size
        # Rotation sensitivity is per canvas pixel, matching the coordinates
        # _on_mouse receives (QtCanvas maps pointer positions into canvas
        # space), so a full-width drag stays a full 360 degree turn.
        self.ORBIT_DEG_PER_PX = 360.0 / max(1, self.canvas_w)
        self._redraw_requested = True
        return True

    def _camera(self) -> Camera:
        cam = Camera(self.view.yaw, self.view.elev, self.view.dist,
                     roll_deg=self.view.roll,
                     size=(self.canvas_w, self.canvas_h))
        cam.pan_px = list(self.view.pan_px)
        return cam

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xFFFF
            if delta & 0x8000:              # high bit set -> scroll down; reinterpret as signed negative
                delta -= 0x10000
            if delta == 0:
                return
            factor = 1.15 ** (delta / 120.0)
            self.view.dist = float(np.clip(
                self.view.dist / factor,
                LiveViewState.DIST_MIN, LiveViewState.DIST_MAX))
            self._redraw_requested = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self._pan_at_drag = list(self.view.pan_px)
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle button down: record the origin, enter 360° free rotation
            self._orbit_start = (x, y)
            self._yaw_at_orbit = self.view.yaw
            self._elev_at_orbit = self.view.elev
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._orbit_start is not None and (flags & cv2.EVENT_FLAG_MBUTTON):
                # Middle-drag: horizontal -> 360° yaw orbit, vertical -> elev pitch (clamped to ±89° to avoid flipping)
                dx = float(x - self._orbit_start[0])
                dy = float(y - self._orbit_start[1])
                self.view.yaw = (self._yaw_at_orbit
                                 + dx * self.ORBIT_DEG_PER_PX) % 360.0
                self.view.elev = float(np.clip(
                    self._elev_at_orbit - dy * self.ORBIT_DEG_PER_PX,
                    -89.0, 89.0))
                self._redraw_requested = True
            elif self._drag_start is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
                self.view.pan_px[0] = self._pan_at_drag[0] + float(x - self._drag_start[0])
                self.view.pan_px[1] = self._pan_at_drag[1] + float(y - self._drag_start[1])
                self._redraw_requested = True
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag_start = None
        elif event == cv2.EVENT_MBUTTONUP:
            self._orbit_start = None

    def _view_control_step(self, direction: str) -> None:
        """Rotate only the hand model; repeated presses are driven by Qt."""
        # The rendered coordinate convention is X=red, Y=green, Z=blue.
        # Match the controls to the visible axes: left/right -> green Y,
        # up/down -> red X.  The ring uses blue Z below.
        increments = {
            "left": (1, -1.0),
            "right": (1, 1.0),
            "up": (0, -1.0),
            "down": (0, 1.0),
        }
        item = increments.get(direction)
        if item is not None:
            axis_index, sign = item
            _, step_deg, _ = self.MODEL_CONTROL_SENSITIVITIES[
                self._model_control_sensitivity_index]
            self._apply_model_control_rotation(
                self._model_control_axis(axis_index), sign * step_deg)

    def _view_control_roll(self, delta_deg: float) -> None:
        """Rotate only the hand model around the fixed rendered Z axis."""
        _, _, drag_multiplier = self.MODEL_CONTROL_SENSITIVITIES[
            self._model_control_sensitivity_index]
        self._apply_model_control_rotation(
            self._model_control_axis(2), float(delta_deg) * drag_multiplier)

    def _model_control_axis(self, index: int) -> np.ndarray:
        """Resolve X/Y/Z against the axes actually drawn in this viewer."""
        basis = self._grid_basis()
        if basis is None:
            basis = np.eye(3, dtype=float)
        axis = np.asarray(basis, dtype=float).reshape(3, 3)[:, int(index)]
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        return axis

    def _apply_model_control_rotation(
            self, axis: np.ndarray, angle_deg: float) -> None:
        delta = R.from_rotvec(
            np.asarray(axis, dtype=float) * np.deg2rad(float(angle_deg)))
        # Pre-composition keeps every increment on the fixed, rendered axis.
        self._model_control_rotation = delta * self._model_control_rotation
        self._redraw_requested = True

    def _view_control_reset(self) -> None:
        """Return to exactly the camera/model state captured at window open."""
        for key in ("yaw", "elev", "roll", "dist"):
            setattr(self.view, key, float(self._initial_view[key]))
        self.view.pan_px = list(self._initial_view["pan_px"])
        self._model_control_rotation = R.identity()
        self._model_control_sensitivity_index = 1
        self._redraw_requested = True

    def _view_control_cycle_sensitivity(self) -> None:
        self._model_control_sensitivity_index = (
            self._model_control_sensitivity_index + 1
        ) % len(self.MODEL_CONTROL_SENSITIVITIES)
        self._redraw_requested = True

    def _view_control_sensitivity_label(self) -> str:
        name = self.MODEL_CONTROL_SENSITIVITIES[
            self._model_control_sensitivity_index][0]
        labels = {
            "fine": L("灵敏度：精细", "Sensitivity: Fine"),
            "standard": L("灵敏度：标准", "Sensitivity: Standard"),
            "fast": L("灵敏度：快速", "Sensitivity: Fast"),
        }
        return labels[name]

    def _view_control_angles(self) -> tuple[float, float, float]:
        """Return display yaw/pitch/roll for the hand-only transform."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yaw, pitch, roll = self._model_control_rotation.as_euler(
                "yxz", degrees=True)
        return float(yaw), float(pitch), float(roll)

    def _key_command(self, key: int):
        # Camera orbits the origin; grid/axes/hand rotate with the scene
        if key in (ord("a"), ord("d")):
            self.view.yaw += 3.0 if key == ord("a") else -3.0
            self._redraw_requested = True
        elif key in (ord("w"), ord("s")):
            self.view.elev += 3.0 if key == ord("w") else -3.0
            self._redraw_requested = True
        elif key in (ord("e"), ord("c")):
            self.view.roll = (self.view.roll + (3.0 if key == ord("e") else -3.0)) % 360.0
            self._redraw_requested = True
        elif key == ord("["):
            # Shrink the tactile overlay (fixed 2D, does not affect the 3D view)
            self.tactile_scale = max(
                TACTILE_SCALE_MIN, round(self.tactile_scale - 0.1, 1))
            self._redraw_requested = True
        elif key == ord("]"):
            self.tactile_scale = min(
                TACTILE_SCALE_MAX, round(self.tactile_scale + 0.1, 1))
            self._redraw_requested = True
        elif key == ord("r"):
            self._view_control_reset()
        elif key == ord("v"):
            saved = self.view.save()
            print(f"View saved: {saved}")

    def _hud_lines(self) -> list[str]:
        view_label = L("视角", "View")
        lines = [
            L(f"GUI / 解算: {self.fps:5.1f} FPS",
              f"GUI / solve: {self.fps:5.1f} FPS"),
            L(f"IMU {self.device_fps:5.1f} Hz   压力 {self.tactile_fps:5.1f} Hz",
              f"IMU {self.device_fps:5.1f} Hz   TACTILE {self.tactile_fps:5.1f} Hz"),
            L(f"显示: {'完整手' if self.display_mode == 'hand' else '手骨'}",
              f"Display: {'full hand' if self.display_mode == 'hand' else 'bones'}"),
            f"{view_label}: yaw {self.view.yaw:5.1f}°  elev {self.view.elev:5.1f}°  "
            f"roll {self.view.roll:5.1f}°  dist {self.view.dist:.2f}m",
        ]
        if self.display_mode == "hand" and self._mesh_faces is None:
            lines.append(L("完整手网格不可用；显示手骨",
                           "Full hand mesh unavailable; showing bones"))
        if self.safety_intervened:
            lines.append(L("姿势安全修正已激活", "Pose safety correction active"))
        if self.active_contact:
            lines.append(L(f"接触: {self.active_contact}", f"Contact: {self.active_contact}"))
        return lines

    def _display_slots(self) -> np.ndarray:
        slots = self._kpts_slots.copy()
        for slot, rotation in enumerate(self._display_rotations):
            if rotation is not None and np.isfinite(slots[slot]).all():
                slots[slot] = apply_hand_display_rotation(slots[slot], rotation)
        return slots

    def _display_mesh_slots(self) -> list[np.ndarray | None]:
        slots: list[np.ndarray | None] = [None, None]
        for slot, vertices in enumerate(self._mesh_slots):
            if vertices is None:
                continue
            mesh = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
            rotation = self._display_rotations[slot]
            if rotation is not None and np.isfinite(mesh).all():
                mesh = rotation.apply(mesh).astype(np.float32)
            slots[slot] = mesh
        return slots

    @staticmethod
    def _model_control_pivot(slots: np.ndarray) -> np.ndarray:
        """Use one wrist, or the midpoint of two wrists, as model pivot."""
        wrists = [slot[0] for slot in np.asarray(slots)
                  if np.isfinite(slot).all()]
        if not wrists:
            return np.zeros(3, dtype=np.float32)
        return np.mean(np.asarray(wrists, dtype=np.float32), axis=0)

    def _controlled_display_slots(self) -> np.ndarray:
        """Final display joints with the hand-only UI transform applied."""
        slots = np.asarray(self._display_slots(), dtype=np.float32).copy()
        pivot = self._model_control_pivot(slots)
        for index, slot in enumerate(slots):
            if np.isfinite(slot).all():
                slots[index] = (
                    self._model_control_rotation.apply(slot - pivot) + pivot
                ).astype(np.float32)
        return slots

    def _controlled_display_mesh_slots(self) -> list[np.ndarray | None]:
        """Apply the identical hand-only transform to full-hand meshes."""
        source_slots = np.asarray(self._display_slots(), dtype=np.float32)
        pivot = self._model_control_pivot(source_slots)
        result: list[np.ndarray | None] = [None, None]
        for index, mesh in enumerate(self._display_mesh_slots()):
            if mesh is not None and np.isfinite(mesh).all():
                points = np.asarray(mesh, dtype=np.float32).reshape(-1, 3)
                result[index] = (
                    self._model_control_rotation.apply(points - pivot) + pivot
                ).astype(np.float32)
        return result

    def _grid_basis(self) -> np.ndarray | None:
        return None

    def _draw(self) -> np.ndarray:
        # One size for the whole frame, so the camera and the text anchors
        # below cannot disagree if a resize lands mid-draw.
        w, h = self.canvas_w, self.canvas_h
        # Grid radius scales only with camera dist (wheel-controlled), not with the per-frame hand scale
        grid_r = max(0.1, self.view.dist * 0.5)
        img = render_frame_live(
            self._camera(), self._controlled_display_slots(),
            self._hud_lines(),
            self.title if self._show_title_in_canvas else None,
            grid_r=grid_r, grid_basis=self._grid_basis(),
            mesh_slots=self._controlled_display_mesh_slots(),
            mesh_faces=self._mesh_faces,
            display_mode=self.display_mode, size=(w, h))
        # Bottom-anchored: only the distance from the edge follows the canvas,
        # the font size does not (see render_frame_live on size).
        put_text(img, L(
            "方向盘/外圈=旋转手模 中键拖=旋转相机 滚轮=缩放 左键拖=平移 "
            "r=复位 v=保存 [ / ]=触觉大小 q=退出",
            "pad/ring=rotate model M-drag=rotate camera wheel=zoom L-drag=pan "
            "r=reset v=save [ / ]=tactile size q=quit"),
            (16, h - 40), 0.45, (200, 195, 185), 1)
        put_text(img, L("3D 单位: 米; 腕部相对", "3D units: metres; wrist-relative"),
                 (16, h - 16), 0.55, (200, 195, 185), 1)
        self._draw_tactile(img)
        return img

    def set_tactile(self, force_n):
        """Accept the 16x16 finger-force matrix in newtons (None clears the panel), drawn at the fixed lower-left corner."""
        self.tactile_frame = (None if force_n is None
                              else np.asarray(force_n, np.float32).reshape(16, 16))
        self._redraw_requested = True

    def _on_display_mode(self, value) -> None:
        mode = str(value)
        if mode in ("bones", "hand") and mode != self.display_mode:
            self.display_mode = mode
            self._redraw_requested = True

    def _display_mode_options(self) -> list[tuple[str, str]]:
        return [("bones", L("手骨", "Bones")), ("hand", L("完整手", "Full hand"))]

    def _display_label(self) -> str:
        return L("显示", "Display")

    def _lang_button_label(self) -> str:
        # The button shows the language it will switch *to*.
        return L("English", "中文")

    def _selector_button_label(self) -> str:
        return L("选择标定文件", "Select Calibration")

    def _toggle_language(self) -> None:
        set_lang("zh" if is_en() else "en")
        self._qcanvas.retranslate()
        self._redraw_requested = True

    def _draw_tactile(self, img: np.ndarray) -> None:
        """Fixed 2D tactile overlay at the lower-left corner: never rotates, only resizes."""
        # Anchors come from the buffer being drawn into rather than from
        # self.canvas_h, so they cannot disagree with it.
        h = img.shape[0]
        if self.tactile_frame is None:
            return
        frame = self.tactile_frame
        # Force is NaN in the positions no fitted region covers; those cells
        # hold no sensor, so they read as zero force rather than as a NaN patch.
        # No threshold is subtracted here: this panel draws newtons, whose zero
        # is absolute and whose card scales itself, so an ADC-count gate would
        # only erase every cell (the panel is 0..5 N, the old gate counted ADC
        # steps).
        if not np.isfinite(frame).all():
            frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        hand_img = render_hand(
            frame, mirror=self.hand_side == "left",
            side=self.hand_side)
        # tactile_scale is an absolute card size, deliberately not scaled with
        # the canvas: it is a round trip through tactile_overlay's cell<->scale
        # maths, which the grip hit-test and the drag in the bimanual viewers
        # have to invert, and scaling it in one place would detach the handle
        # from the card.
        s = self.tactile_scale
        tw = max(1, int(round(hand_img.shape[1] * s)))
        th = max(1, int(round(hand_img.shape[0] * s)))
        scaled = cv2.resize(hand_img, (tw, th), interpolation=cv2.INTER_LINEAR)
        x0, y0 = 12, h - 60 - th
        img[y0:y0 + th, x0:x0 + tw] = scaled
        cv2.rectangle(img, (x0, y0), (x0 + tw, y0 + th), (120, 130, 150), 1)
        put_text(img, L(f"触觉 x{s:.1f}  [ / ] 缩放", f"Tactile x{s:.1f}  [ / ] resize"),
                 (x0, y0 - 6), 0.45, (200, 195, 185), 1)

    def update(self, joints: np.ndarray, fps: float, frame_count: int,
               missing_labels: list[str], safety_intervened: bool,
               active_contact: str | None = None,
               mesh_vertices: np.ndarray | None = None,
               device_fps: float = 0.0, tactile_fps: float = 0.0):
        """Accept one hand's (21,3) smoothed joints and write them into the matching left/right slot."""
        self.fps = float(fps)
        self.device_fps = float(device_fps or 0.0)
        self.tactile_fps = float(tactile_fps or 0.0)
        self.frame_count = int(frame_count)
        self.missing_labels = list(missing_labels)
        self.safety_intervened = bool(safety_intervened)
        self.active_contact = active_contact
        slots = np.full((2, 21, 3), np.nan, np.float32)
        slots[self.hand_slot] = np.asarray(joints, np.float32).reshape(21, 3)
        self._kpts_slots = slots
        mesh_slots: list[np.ndarray | None] = [None, None]
        if mesh_vertices is not None:
            mesh = np.asarray(mesh_vertices, dtype=np.float32)
            if mesh.shape == (778, 3) and np.isfinite(mesh).all():
                mesh_slots[self.hand_slot] = mesh
        self._mesh_slots = mesh_slots
        self._redraw_requested = True

    def _redraw_due(self, now: float) -> bool:
        """True when a pending repaint may run at the requested redraw cadence."""
        if self._redraw_interval_s <= 0.0 or self._last_redraw_s is None:
            return True
        return now - self._last_redraw_s >= self._redraw_interval_s

    def tick(self) -> bool:
        if self.exit_requested:
            return False
        now = time.monotonic()
        # Adopt a window resize before deciding whether to redraw, so the frame
        # about to be produced is already at the new size.
        self._refresh_canvas_size()
        if self._last_tick_s is None:
            self._last_tick_s = now
        if self.rotating:
            self.view.yaw = (self.view.yaw + self.yaw_rate * (now - self._last_tick_s)) % 360.0
            self._redraw_requested = True
        self._last_tick_s = now
        if self._redraw_requested and self._redraw_due(now):
            self._canvas = self._draw()
            self._redraw_requested = False
            self._last_redraw_s = now
            # Only hand the canvas to Qt when it actually changed; otherwise
            # tick() would trigger a full-window repaint on every loop
            # iteration even when no new data arrived, starving the render
            # thread (this shows up as a stuttering 3D view).
            self._qcanvas.set_canvas(self._canvas)
        pump_events()
        return not self.exit_requested

    def close(self):
        self.view.save()
        self._qcanvas.close()
        pump_events()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=("STM32 glove: live 3D view of the 21 solved keypoints "
                     "and finger force, through the SDK"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", "--calib-config", "--calibration-config",
        dest="config", type=Path, default=PROJECT_ROOT / "config" / "config.json",
        help="Legacy combined configuration path",
    )
    parser.add_argument(
        "--calib", "--calibration-file",
        dest="calib", type=Path, default=None,
        help="Standalone IMU calibration JSON",
    )
    parser.add_argument(
        "--select-calibration", action="store_true",
        help="Open the calibration JSON selector before the live viewer")
    parser.add_argument(
        "--calibration-dir", "--calib-dir", type=Path,
        default=PROJECT_ROOT / "calibration",
        help="Folder scanned by the calibration selector")
    parser.add_argument("--serial-port", help="STM32 CDC port; default: bound device")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="Deprecated compatibility option; calibrated mode is required")
    parser.add_argument("--view", default="static", choices=("static", "rotating"),
                        help="Static interactive view or automatically rotating camera")
    parser.add_argument("--display-mode", default="hand", choices=("bones", "hand"),
                        help="Initial 3D display mode: bones or full gray hand mesh")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh",
                        help="UI language: zh (default) or en")
    parser.add_argument("--view-config", type=Path, default=VIEW3D_CONFIG_PATH,
                        help="Saved camera-view JSON")
    parser.add_argument("--view-reference-config", type=Path, default=None,
                        help="Copy initial camera values from another view JSON")
    parser.add_argument("--yaw", type=float, default=None, help="Initial yaw in degrees")
    parser.add_argument("--elev", type=float, default=None, help="Initial elevation in degrees")
    parser.add_argument("--roll", type=float, default=None, help="Initial roll in degrees")
    parser.add_argument("--dist", type=float, default=None, help="Initial camera distance in metres")
    parser.add_argument("--yaw-rate", type=float, default=60.0,
                        help="Rotating-view angular speed in degrees per second")
    parser.add_argument("--fps", type=float, default=30.0, help="Display frame-rate limit")
    parser.add_argument("--joint-smoothing-ms", type=float, default=35.0)
    parser.add_argument("--display-raw", action="store_true",
                        help="Display unsmoothed solved keypoints")
    parser.add_argument("--startup-timeout", type=float, default=15.0,
                        help="Seconds to wait for the first frame; 0 waits forever")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames; 0 means unlimited")
    return parser.parse_args(argv)


def _stale_backlog(frame: HandFrame, newest: HandFrame | None) -> bool:
    """True when ``frame`` is history rather than part of the current burst.

    A poll used to hand over a whole batch, which is what
    :func:`common.frame_pacing.recent_batch` trimmed; ``Glove.read`` hands over
    one frame from a bounded queue instead, so the same decision is taken one
    frame at a time against ``Glove.latest()``: a frame more than a burst behind
    the newest is the tail of a backlog and display must not replay it, while
    everything inside the window is a batch the link legitimately delivered and
    is kept -- keeping only the newest is what made the displayed rate equal the
    batch rate on a Bluetooth dongle.

    The comparison is a plain difference because ``HandFrame.timestamp_us`` is
    on the host's monotonic timeline (``algorithm.hand.safety``'s device clock),
    not the device's wrapping uint32 counter.
    """
    if newest is None or newest is frame:
        return False
    return int(newest.timestamp_us) - int(frame.timestamp_us) > BURST_SPAN_US


def _port_serial(port_name: str | None) -> str | None:
    """Translate an explicit ``--serial-port`` into the serial a hand is pinned by.

    ``open_glove`` resolves a hand by link serial rather than by COM name, so an
    explicit port has to be looked up and handed over as what the system knows
    it as.  A name that is not an attached glove link is an error rather than a
    silent fallback: the caller asked for one specific device.
    """
    if not port_name:
        return None
    wanted = str(port_name).strip().upper()
    try:
        devices = DeviceManager(DEFAULT_REGISTRY).list_devices()
    except Exception as exc:
        raise RuntimeError(
            f"cannot resolve --serial-port {port_name}: {exc}") from exc
    for device in devices:
        if str(device.device).strip().upper() != wanted:
            continue
        serial = str(device.serial_number or "").strip()
        if serial:
            return serial
        break
    attached = ", ".join(str(device.device) for device in devices) or "none"
    raise RuntimeError(
        f"--serial-port {port_name} is not an attached glove link "
        f"(attached: {attached})")


def _selected_calibration(args) -> Path | None:
    if args.select_calibration and args.calib is None:
        selected = select_calibration_files(
            args.calibration_dir,
            (args.side,),
            title=f"Select {args.side.title()} IMU Calibration",
        )
        if selected is None:
            return None
        args.calib = selected[args.side]
    if args.calib is not None:
        return Path(args.calib).expanduser().resolve()
    # The two files this edition ships: a bare checkout has exactly one
    # calibration per side, so the fallback below always finds something.
    filename = ("imu_2d_calibration.json" if args.side == "left"
                else "imu_calibration_right_default.json")
    return (PROJECT_ROOT / "calibration" / filename).resolve()


# How long one Glove.read may block.  The frame rate belongs to the glove, so a
# short block is only about the idle loop: the window still gets its heartbeat
# and the startup timeout still fires on schedule, without a spinning wait.
READ_TIMEOUT_S = 0.05


def main(argv=None) -> int:
    """Run the live single-hand viewer over the client SDK interface."""

    args = _parse_args(argv)
    set_lang(args.lang)
    try:
        calibration = _selected_calibration(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[Calibration error] {exc}", file=sys.stderr)
        return 2
    if calibration is None:
        print("Calibration selection cancelled.")
        return 130
    if not calibration.is_file():
        print(f"[Calibration error] File does not exist: {calibration}",
              file=sys.stderr)
        return 2

    try:
        glove = open_glove(
            args.side,
            calibration=calibration,
            serial_number=_port_serial(args.serial_port),
            registry=DEFAULT_REGISTRY,
            smoothing_ms=args.joint_smoothing_ms,
        ).start()
    except CalibrationError as exc:
        print(f"[Calibration error] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[Device error] {exc}", file=sys.stderr)
        return 2

    # One snapshot of what is carrying this hand; the link is resolved inside
    # open_glove, so this is also the only place the viewer can learn it.
    info = glove.info()

    view_state = LiveViewState(args.view_config)
    if args.view_reference_config is not None:
        reference_view = LiveViewState(args.view_reference_config)
        view_state.yaw = reference_view.yaw
        view_state.elev = reference_view.elev
        view_state.roll = reference_view.roll
        view_state.dist = reference_view.dist
        view_state.pan_px = list(reference_view.pan_px)
    if args.yaw is not None:
        view_state.yaw = float(args.yaw)
    if args.elev is not None:
        view_state.elev = float(args.elev)
    if args.roll is not None:
        view_state.roll = float(args.roll) % 360.0
    if args.dist is not None:
        view_state.dist = float(np.clip(
            args.dist, LiveViewState.DIST_MIN, LiveViewState.DIST_MAX))

    backend = glove.backend

    viewer = None
    if not args.no_display:
        initial_slots = np.full((2, 21, 3), np.nan, np.float32)
        initial_slots[0 if args.side == "left" else 1] = glove.neutral_joints()
        source_label = "USB" if info.link_kind == LINK_USB else "Bluetooth"
        viewer = Live3DViewer(
            view_state,
            initial_slots,
            f"STM32 glove live 21 joints - {source_label} / "
            f"{backend.get('backend', 'solver')} ({args.view})",
            rotating=args.view == "rotating",
            yaw_rate=args.yaw_rate,
            hand_side=args.side,
            display_mode=args.display_mode,
            initial_mesh_vertices=glove.mesh_vertices_m,
            redraw_interval_s=1.0 / max(args.fps, 1.0),
        )

    print(f"Calibration: {calibration}")
    print(f"Link: {info.link_kind} {info.port} [{info.serial_number}]")
    print("Pipeline: open_glove -> Glove.read -> 21 keypoints + newton force")
    print("Tactile: HandFrame.force_n -> 16x16 live panel (newtons)")
    print("Controls: middle-drag rotate, wheel zoom, left-drag pan, "
          "a/d/w/s orbit, e/c roll, r reset, v save, q/ESC quit.")

    start_wait_s = time.monotonic()
    frame_index = 0
    fps_times: list[float] = []
    exit_requested = False
    return_code = 0
    display_interval = 1.0 / max(args.fps, 1.0)
    next_display_s = 0.0

    try:
        while not exit_requested:
            try:
                frame = glove.read(timeout=READ_TIMEOUT_S)
            except StreamTimeoutError:
                # Not a failure yet: the startup timeout below, and the
                # window's own heartbeat, are what decide when silence is.
                frame = None
            else:
                if _stale_backlog(frame, glove.latest()):
                    # Skipping it, rather than drawing it, is the point: the
                    # view must track the newest frame and never replay the
                    # backlog a slow consumer let pile up.
                    continue

            if frame is None:
                if (args.startup_timeout > 0 and frame_index == 0
                        and time.monotonic() - start_wait_s >= args.startup_timeout):
                    detail = glove.health().last_error
                    raise RuntimeError(
                        "Timed out waiting for STM32 data"
                        + (f": {detail}" if detail else ""))
                if viewer is not None and not viewer.tick():
                    exit_requested = True
                time.sleep(0.003)
                continue

            # Finger force in newtons, NaN where no fitted region covers the
            # cell.  The matrix rides on every attitude frame until a new scan
            # replaces it, so it is redrawn as-is rather than gated on
            # ``force_fresh``.
            if frame.force_n is not None and viewer is not None:
                viewer.set_tactile(frame.force_n)

            # The frame carries its own solved copies.  --display-raw picks the
            # unsmoothed keypoints; otherwise the SDK's smoothing_ms low-pass.
            joints = frame.joints_raw_m
            smoothed = frame.joints_smoothed_m
            status = frame.status
            now = time.monotonic()
            fps_times.append(now)
            fps_times = fps_times[-60:]
            live_fps = ((len(fps_times) - 1)
                        / max(fps_times[-1] - fps_times[0], 1e-6)
                        if len(fps_times) >= 2 else 0.0)
            if viewer is not None and now >= next_display_s:
                viewer.update(
                    joints if args.display_raw else smoothed,
                    live_fps,
                    frame_index + 1,
                    [],
                    status.safety_intervened,
                    active_contact=status.active_contact,
                    mesh_vertices=glove.mesh_vertices_m,
                    device_fps=frame.device_fps,
                    tactile_fps=frame.tactile_fps,
                )
                if not viewer.tick():
                    exit_requested = True
                next_display_s = now + display_interval
            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                exit_requested = True
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return_code = 130
    except RuntimeError as exc:
        print(f"[Runtime error] {exc}", file=sys.stderr)
        return_code = 3
    finally:
        glove.close()
        if viewer is not None:
            viewer.close()
    print(f"Finished: {frame_index} frames")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
