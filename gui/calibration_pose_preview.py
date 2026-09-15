"""Lightweight HAND2mm full-hand reference images for the calibration GUIs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from algorithm.lite.hand import HandNumpy
from gui.glove_21_live_3d import (
    Camera,
    align_left_to_right_reference,
    apply_hand_display_rotation,
    hand_display_basis,
    render_frame_live,
)
from gui.live_3d import _load_mano_faces


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
PREVIEW_WIDTH = 660
PREVIEW_HEIGHT = 300
_CONTACT_NAMES = ("index", "middle", "ring", "little")

# The 21-point live 3D view uses yaw=270, elev=8 and roll=240 together with
# its tuned palm display basis.  These are the equivalent camera angles after
# expressing that same eye-to-hand view in this preview's canonical hand
# basis.  Keep the closer distance and automatic pan below so a single hand
# remains large and centered in the calibration panel.
_PREVIEW_YAW = 186.80125157005767
_PREVIEW_ELEV = -44.31525254354176
_PREVIEW_ROLL = 1.524153207538752
_PREVIEW_DIST = 0.38

# When bimanual_display_config.json (the ./glove.sh live tuning) is present,
# the per-step previews are rendered in the fixed live world: each hand sits at
# its bimanual palm target basis (thumbs inward, left/right mirror-symmetric),
# the live camera yaw/elev/roll and pan are fixed per side, and the bimanual
# ground grid is drawn at the wrist (world origin).  Only the hand model
# rotates between steps.  Root-step poses are built in this fixed world so the
# axes never move: fingertips align to the grid normal for up/down, and the
# yaw step turns the level hand about the grid normal.  _LIVE_VIEW_DIST is
# closer than the two-hand live distance so a single hand stays large in the
# panel.  Without the config file the preview falls back to the canonical
# XY-grid orientation and dedicated camera above.
_LIVE_VIEW_YAW = 270.0
_LIVE_VIEW_ELEV = 8.0
_LIVE_VIEW_ROLL = 240.0
_LIVE_VIEW_DIST = 0.42
# Contact poses need a three-quarter palm view: from the normal calibration
# camera the thumb/fingertip pair sits in front of the palm and the remaining
# fingers point almost into the lens.  Orbiting right and slightly upward
# makes both the contact and the non-target fingers readable.  The left-hand
# camera is derived as an exact screen-space mirror below.
_CONTACT_VIEW_YAW_OFFSET = 45.0
_CONTACT_VIEW_ELEV_OFFSET = 22.0
_BIMANUAL_DISPLAY_CONFIG = PROJECT_ROOT / "config" / "bimanual_display_config.json"
# Framing search for the fixed per-side camera: the crop window taken from the
# 1280x720 canvas below, the smallest distance (largest hand) whose one fixed
# pan keeps all 13 steps fully inside it.
_CROP_X0, _CROP_X1, _CROP_Y0, _CROP_Y1 = 190.0, 1090.0, 55.0, 685.0
_LIVE_FRAME_LO, _LIVE_FRAME_HI, _LIVE_FRAME_SAMPLES = 0.28, 0.70, 42


def _default_calibration_path(side: str) -> Path:
    name = "imu_2d_calibration.json" if side == "left" else "imu_calibration.json"
    return PROJECT_ROOT / "calibration" / name


class CalibrationPosePreview:
    """Generate cached BGR full-hand previews using the current lite renderer."""

    def __init__(self):
        self._models = {
            side: HandNumpy(side, BUNDLE_ROOT / "assets/hand")
            for side in ("left", "right")
        }
        neutral_outputs = {
            side: self._models[side].forward(np.zeros((16, 3)))
            for side in ("left", "right")
        }
        self._neutral = {
            side: output.joints
            for side, output in neutral_outputs.items()
        }
        self._neutral_mesh = {
            side: output.verts
            for side, output in neutral_outputs.items()
        }
        self._mesh_faces = {
            side: _load_mano_faces(side)
            for side in ("left", "right")
        }
        left_alignment = align_left_to_right_reference(
            self._neutral["left"], self._neutral["right"])
        self._display_alignment = {
            "left": left_alignment,
            "right": R.identity(),
        }
        # Put the neutral hand on the world XY grid: fingertips toward +Y,
        # back-of-hand normal toward +Z. Root-step rotations can then match
        # the physical instructions relative to the visible world axes.
        right_basis = hand_display_basis(self._neutral["right"])
        self._canonical_rotation = R.from_matrix(right_basis.T)
        contact_geometry = {
            side: self._load_contacts(side) for side in ("left", "right")
        }
        self._contacts = {
            side: {name: geometry[0] for name, geometry in contacts.items()}
            for side, contacts in contact_geometry.items()
        }
        self._contact_meshes = {
            side: {name: geometry[1] for name, geometry in contacts.items()}
            for side, contacts in contact_geometry.items()
        }
        self._live_enabled = False
        self._live_view = {
            "yaw": _PREVIEW_YAW, "elev": _PREVIEW_ELEV,
            "roll": _PREVIEW_ROLL, "dist": _PREVIEW_DIST,
        }
        self._live_dist = {"left": _LIVE_VIEW_DIST, "right": _LIVE_VIEW_DIST}
        self._live_pan = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
        self._contact_view: dict[str, dict[str, float]] = {}
        self._contact_dist = {
            "left": _LIVE_VIEW_DIST, "right": _LIVE_VIEW_DIST,
        }
        self._contact_pan = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
        self._live_down_translation = {"left": None, "right": None}
        self._live_grid_basis = None
        self._live_neutral: dict[str, np.ndarray] = {}
        self._live_neutral_mesh: dict[str, np.ndarray] = {}
        self._live_root: dict[str, dict[str, R]] = {}
        self._live_contacts: dict[str, dict[str, np.ndarray]] = {}
        self._live_contact_meshes: dict[str, dict[str, np.ndarray]] = {}
        try:
            payload = json.loads(
                _BIMANUAL_DISPLAY_CONFIG.read_text(encoding="utf-8"))
            targets = np.asarray(
                payload["palm_target_bases"], dtype=float).reshape(2, 3, 3)
            view = payload.get("view") or {}
            anchor_axis = -targets[1, :, 0].copy()
            grid_forward = targets[1, :, 1].copy()
            grid_normal = np.cross(anchor_axis, grid_forward)
            grid_normal /= max(float(np.linalg.norm(grid_normal)), 1e-9)
            self._live_grid_basis = np.stack(
                [anchor_axis, grid_forward, grid_normal], axis=1)
            self._live_view = {
                "yaw": float(view.get("yaw", _LIVE_VIEW_YAW)),
                "elev": float(view.get("elev", _LIVE_VIEW_ELEV)),
                "roll": float(view.get("roll", _LIVE_VIEW_ROLL)),
                "dist": _LIVE_VIEW_DIST,
            }
            world_up = grid_normal.copy()
            # yaw/roll 预览方向与校准翻转约定一致（掌心相对）：
            #   - yaw：左手"水平右转"（绕 world_up 负向 90°，屏上指尖向右）；
            #     右手取反为"水平左转"（绕 world_up 正向 90°，屏上指尖向左）。
            #   - roll：绕腕→指尖纵轴翻掌 90°（拇指朝上）。右手拇指初始在屏左
            #     （-X），拇指朝上需绕前向轴 +90°；左手物理方向相反为 -90°。
            #     修正记录：原 ± 写反，右手渲染成了左手姿态（仅预览视觉手性，
            #     与求解方向的 selfcheck T5/T6 无关）。
            yaw_turn_right = R.from_rotvec(world_up * np.deg2rad(-90.0))
            yaw_turn_left = R.from_rotvec(world_up * np.deg2rad(90.0))
            for side, slot in (("left", 0), ("right", 1)):
                fixed = R.from_matrix(
                    targets[slot]
                    @ hand_display_basis(self._neutral[side]).T)
                neutral_live = apply_hand_display_rotation(
                    self._neutral[side], fixed)
                self._live_neutral[side] = neutral_live
                self._live_neutral_mesh[side] = self._rotate_points(
                    self._neutral_mesh[side], fixed, self._neutral[side][0])
                lateral = hand_display_basis(neutral_live)[:, 0]
                fingertip = neutral_live[12] - neutral_live[0]
                fingertip /= max(float(np.linalg.norm(fingertip)), 1e-9)
                roll_angle = (np.deg2rad(-90.0) if side == "left"
                              else np.deg2rad(90.0))
                self._live_root[side] = {
                    "up": R.align_vectors(
                        np.stack([world_up, lateral]),
                        np.stack([fingertip, lateral]))[0],
                    "down": R.align_vectors(
                        np.stack([-world_up, lateral]),
                        np.stack([fingertip, lateral]))[0],
                    "yaw": (yaw_turn_left if side == "right" else yaw_turn_right),
                    "roll": R.from_rotvec(fingertip * roll_angle),
                }
                live_map = fixed * (
                    self._canonical_rotation
                    * self._display_alignment[side]).inv()
                self._live_contacts[side] = {
                    name: apply_hand_display_rotation(pose, live_map)
                    for name, pose in self._contacts[side].items()
                }
                self._live_contact_meshes[side] = {
                    name: self._rotate_points(
                        mesh, live_map, self._contacts[side][name][0])
                    for name, mesh in self._contact_meshes[side].items()
                }
            # Fine-calibration preview (contact steps): the left glove's contact
            # anchors render as a right-hand-shaped contact already, so the right
            # panel reuses that exact pose (its own calibration file is absent
            # and would fall back to a flat open hand), and the left panel shows
            # its left-right mirror so it reads as a left hand again.
            left_contact = self._live_contacts["left"]
            self._live_contacts["right"] = {
                name: pose.copy() for name, pose in left_contact.items()
            }
            left_contact_meshes = self._live_contact_meshes["left"]
            self._live_contact_meshes["right"] = {
                name: mesh.copy() for name, mesh in left_contact_meshes.items()
            }
            self._live_contacts["left"] = {
                name: self._mirror_pose_about_camera(pose)
                for name, pose in left_contact.items()
            }
            self._live_contact_meshes["left"] = {
                name: self._mirror_points_about_camera(
                    mesh, left_contact[name][0])
                for name, mesh in left_contact_meshes.items()
            }
            self._live_enabled = True
            right_contact_view = {
                "yaw": (self._live_view["yaw"]
                        + _CONTACT_VIEW_YAW_OFFSET) % 360.0,
                "elev": float(np.clip(
                    self._live_view["elev"] + _CONTACT_VIEW_ELEV_OFFSET,
                    -80.0, 80.0)),
                "roll": self._live_view["roll"],
                "dist": _LIVE_VIEW_DIST,
            }
            self._contact_view = {
                "right": right_contact_view,
                "left": self._mirror_camera_view(right_contact_view),
            }
            for side in ("left", "right"):
                self._live_dist[side], self._live_pan[side] = \
                    self._fit_live_frame(side, range(8), self._live_view)
                self._live_down_translation[side] = \
                    self._fit_down_translation(side)
                self._contact_dist[side], self._contact_pan[side] = \
                    self._fit_live_frame(
                        side, range(8, 13), self._contact_view[side])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._live_enabled = False
            pass
        self._image_cache: dict[tuple[str, int], np.ndarray] = {}
        self._bimanual_image_cache: dict[int, np.ndarray] = {}

    def _align(self, side: str, joints: np.ndarray) -> np.ndarray:
        aligned = apply_hand_display_rotation(
            np.asarray(joints, np.float32), self._display_alignment[side])
        return apply_hand_display_rotation(aligned, self._canonical_rotation)

    @staticmethod
    def _rotate_points(
            points: np.ndarray, rotation: R, pivot: np.ndarray) -> np.ndarray:
        """Rotate an arbitrary point cloud about the hand's wrist."""
        values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        origin = np.asarray(pivot, dtype=np.float32).reshape(3)
        return (rotation.apply(values - origin) + origin).astype(np.float32)

    def _align_mesh(
            self, side: str, vertices: np.ndarray,
            wrist: np.ndarray) -> np.ndarray:
        aligned = self._rotate_points(
            vertices, self._display_alignment[side], wrist)
        return self._rotate_points(aligned, self._canonical_rotation, wrist)

    def _joints_for_step_live(self, side: str, index: int) -> np.ndarray:
        """Per-step pose in the fixed live world (only the hand model moves).

        Root poses are wrist-anchored rotations of the live neutral hand about
        axes of the fixed world grid: up/down align the fingertips with the
        grid normal (opposite signs), yaw turns the level hand about the grid
        normal.  The world grid/camera/pan never move between steps.
        """
        base = self._live_neutral[side]
        if index == 1:
            return apply_hand_display_rotation(
                base, self._live_root[side]["up"])
        if index == 2:
            pose = apply_hand_display_rotation(
                base, self._live_root[side]["down"])
            shift = self._live_down_translation[side]
            return pose + shift if shift is not None else pose
        if index == 3:
            return apply_hand_display_rotation(
                base, self._live_root[side]["roll"])
        if index == 4:
            return apply_hand_display_rotation(
                base, self._live_root[side]["yaw"])
        if 8 <= index <= 11:
            return self._live_contacts[side][
                _CONTACT_NAMES[index - 8]].copy()
        if index == 12:
            return self._live_contacts[side]["little"].copy()
        return base.copy()

    def _mesh_for_step_live(self, side: str, index: int) -> np.ndarray:
        """Return the surface mesh transformed exactly like the step joints."""
        base = self._live_neutral_mesh[side]
        wrist = self._live_neutral[side][0]
        if index == 1:
            return self._rotate_points(base, self._live_root[side]["up"], wrist)
        if index == 2:
            mesh = self._rotate_points(
                base, self._live_root[side]["down"], wrist)
            shift = self._live_down_translation[side]
            return mesh + shift if shift is not None else mesh
        if index == 3:
            return self._rotate_points(
                base, self._live_root[side]["roll"], wrist)
        if index == 4:
            return self._rotate_points(
                base, self._live_root[side]["yaw"], wrist)
        if 8 <= index <= 11:
            return self._live_contact_meshes[side][
                _CONTACT_NAMES[index - 8]].copy()
        if index == 12:
            return self._live_contact_meshes[side]["little"].copy()
        return base.copy()

    def _mirror_pose_about_camera(self, joints: np.ndarray) -> np.ndarray:
        """Left-right mirror of a pose about the camera's screen-vertical plane.

        Reflects each joint about the plane through the wrist normal to the
        camera's screen-right axis, so the rendered hand is flipped about the
        image's vertical centerline while the world grid stays untouched.
        """
        hand = np.asarray(joints, dtype=np.float32).reshape(21, 3)
        return self._mirror_points_about_camera(hand, hand[0])

    def _mirror_points_about_camera(
            self, points: np.ndarray, pivot: np.ndarray) -> np.ndarray:
        """Reflect an arbitrary hand point cloud across the screen centre."""
        view = self._live_view
        probe = Camera(view["yaw"], view["elev"], view["dist"],
                       roll_deg=view["roll"])
        right = probe.right
        wrist = np.asarray(pivot, dtype=float).reshape(3)
        relative = np.asarray(points, dtype=float).reshape(-1, 3) - wrist
        relative = relative - 2.0 * (relative @ right)[:, None] * right[None, :]
        return (relative + wrist).astype(np.float32)

    def _mirror_camera_view(self, view: dict[str, float]) -> dict[str, float]:
        """Mirror a camera across the base view's screen-vertical plane.

        Contact geometry for the left hand is mirrored across this same plane.
        Mirroring the camera as well produces the exact left/right counterpart
        of the clearer right-hand three-quarter view, including screen roll.
        """
        base = self._live_view
        plane_camera = Camera(
            base["yaw"], base["elev"], 1.0, roll_deg=base["roll"])
        mirror = (
            np.eye(3)
            - 2.0 * np.outer(plane_camera.right, plane_camera.right)
        )
        source = Camera(
            view["yaw"], view["elev"], 1.0, roll_deg=view["roll"])
        position = mirror @ source.pos
        desired_up = mirror @ source.up

        yaw = float(np.rad2deg(np.arctan2(position[0], position[2])) % 360.0)
        elev = float(np.rad2deg(np.arcsin(np.clip(
            position[1] / max(float(np.linalg.norm(position)), 1e-9),
            -1.0, 1.0))))
        unrolled = Camera(yaw, elev, 1.0, roll_deg=0.0)
        roll_sin = -float(desired_up @ unrolled.right)
        roll_cos = float(desired_up @ unrolled.up)
        roll = float(np.rad2deg(np.arctan2(roll_sin, roll_cos)) % 360.0)
        return {"yaw": yaw, "elev": elev, "roll": roll,
                "dist": float(view.get("dist", _LIVE_VIEW_DIST))}

    def _fit_down_translation(self, side: str) -> np.ndarray:
        """Upward world shift for the fingertips-down pose so it reads clearly.

        The wrist-anchored down rotation hangs the fingers from the grid
        origin down to the crop's bottom edge, which makes the direction
        ambiguous.  Shift only this one pose upward along the camera's
        screen-up axis -- the world grid and camera stay fixed -- until the
        pose's projected 2D bounding box sits at the crop's vertical center.
        The shift is exact because a translation along cam.up is orthogonal to
        the camera's fwd and right axes, so depth does not change.
        """
        view = self._live_view
        pose = self._mesh_for_step_live(side, 2)
        cam = Camera(view["yaw"], view["elev"], self._live_dist[side],
                     roll_deg=view["roll"], pan_px=self._live_pan[side])
        pts, _ = cam.project(pose)
        cy = 0.5 * (float(pts[:, 1].min()) + float(pts[:, 1].max()))
        target = 0.5 * (_CROP_Y0 + _CROP_Y1)
        center_3d = 0.5 * (pose.min(axis=0) + pose.max(axis=0))
        z_ref = float(cam.project(center_3d[None])[1][0])
        t = -(target - cy) * z_ref / cam.f
        return np.asarray(cam.up, np.float64) * t

    def _fit_live_frame(
            self, side: str, steps, view: dict[str, float]
            ) -> tuple[float, tuple[float, float]]:
        """Fit one fixed distance/pan around the requested preview steps.

        The world never moves within a pose group: yaw/elev/roll and the grid
        are fixed; only distance and pan differ per hand/group.  The smallest
        distance whose single pan keeps every requested step fully inside the
        crop window is chosen.
        """
        steps = tuple(steps)
        for i in range(_LIVE_FRAME_SAMPLES):
            dist = _LIVE_FRAME_LO + (
                _LIVE_FRAME_HI - _LIVE_FRAME_LO) * (i / (_LIVE_FRAME_SAMPLES - 1))
            probe = Camera(view["yaw"], view["elev"], dist,
                           roll_deg=view["roll"])
            points = [
                probe.project(self.mesh_for_step(side, s))[0]
                for s in steps
            ]
            union = np.concatenate(points, axis=0)
            pan = (
                640.0 - 0.5 * (float(union[:, 0].min())
                               + float(union[:, 0].max())),
                360.0 - 0.5 * (float(union[:, 1].min())
                               + float(union[:, 1].max())),
            )
            camera = Camera(view["yaw"], view["elev"], dist,
                            roll_deg=view["roll"], pan_px=pan)
            ok = True
            for s in steps:
                pts, depth = camera.project(self.mesh_for_step(side, s))
                if not ((depth > 0.0).all()
                        and pts[:, 0].min() >= _CROP_X0
                        and pts[:, 0].max() <= _CROP_X1
                        and pts[:, 1].min() >= _CROP_Y0
                        and pts[:, 1].max() <= _CROP_Y1):
                    ok = False
                    break
            if ok:
                return float(dist), pan
        return float(_LIVE_FRAME_HI), (0.0, 0.0)

    def _load_contacts(
            self, side: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        try:
            payload = json.loads(
                _default_calibration_path(side).read_text(encoding="utf-8"))
            calibrator = payload["param_shape_calib"]["contact_calibrator"]
            anchors = calibrator["anchors"]
            corrections = calibrator["corrections"]
            result = {}
            for name in _CONTACT_NAMES:
                output = self._models[side].forward(
                    np.asarray(anchors[name], dtype=float)
                    + np.asarray(corrections[name], dtype=float))
                result[name] = (
                    self._align(side, output.joints),
                    self._align_mesh(
                        side, output.verts, output.joints[0]),
                )
            return result
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            neutral = self._align(side, self._neutral[side])
            neutral_mesh = self._align_mesh(
                side, self._neutral_mesh[side], self._neutral[side][0])
            return {
                name: (neutral.copy(), neutral_mesh.copy())
                for name in _CONTACT_NAMES
            }

    def joints_for_step(self, side: str, step_index: int) -> np.ndarray:
        side = str(side).lower()
        if side not in ("left", "right"):
            raise ValueError(f"side must be left or right, got {side!r}")
        index = max(0, min(int(step_index), 12))
        if self._live_enabled:
            return self._joints_for_step_live(side, index)
        base = self._align(side, self._neutral[side])
        wrist = base[0].copy()
        relative = base - wrist
        if index == 1:
            return R.from_euler("x", 90.0, degrees=True).apply(relative) + wrist
        if index == 2:
            return R.from_euler("x", -90.0, degrees=True).apply(relative) + wrist
        if index == 3:
            return R.from_euler("x", 90.0, degrees=True).apply(relative) + wrist
        if index == 4:
            return R.from_euler("z", -90.0, degrees=True).apply(relative) + wrist
        if 8 <= index <= 11:
            return self._contacts[side][_CONTACT_NAMES[index - 8]].copy()
        if index == 12:
            return self._contacts[side]["little"].copy()
        return base.copy()

    def mesh_for_step(self, side: str, step_index: int) -> np.ndarray:
        """Return the complete 778-vertex hand surface for a preview step."""
        side = str(side).lower()
        if side not in ("left", "right"):
            raise ValueError(f"side must be left or right, got {side!r}")
        index = max(0, min(int(step_index), 12))
        if self._live_enabled:
            return self._mesh_for_step_live(side, index)
        base = self._align_mesh(
            side, self._neutral_mesh[side], self._neutral[side][0])
        wrist = self.joints_for_step(side, 0)[0]
        if index == 1:
            return self._rotate_points(
                base, R.from_euler("x", 90.0, degrees=True), wrist)
        if index == 2:
            return self._rotate_points(
                base, R.from_euler("x", -90.0, degrees=True), wrist)
        if index == 3:
            return self._rotate_points(
                base, R.from_euler("x", 90.0, degrees=True), wrist)
        if index == 4:
            return self._rotate_points(
                base, R.from_euler("z", -90.0, degrees=True), wrist)
        if 8 <= index <= 11:
            return self._contact_meshes[side][
                _CONTACT_NAMES[index - 8]].copy()
        if index == 12:
            return self._contact_meshes[side]["little"].copy()
        return base.copy()

    def render_bgr(self, side: str, step_index: int) -> np.ndarray:
        key = (str(side).lower(), int(step_index))
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached
        joints = self.joints_for_step(*key)
        mesh = self.mesh_for_step(*key)
        displayed = joints
        slots = np.full((2, 21, 3), np.nan, np.float32)
        slot = 0 if key[0] == "left" else 1
        slots[slot] = displayed
        mesh_slots: list[np.ndarray | None] = [None, None]
        mesh_slots[slot] = mesh
        if self._live_enabled:
            is_contact = 8 <= key[1] <= 12
            view = (self._contact_view[key[0]]
                    if is_contact else self._live_view)
            dist = (self._contact_dist[key[0]]
                    if is_contact else self._live_dist[key[0]])
            pan = (self._contact_pan[key[0]]
                   if is_contact else self._live_pan[key[0]])
            camera = Camera(
                view["yaw"], view["elev"], dist,
                roll_deg=view["roll"], pan_px=pan)
            grid_r = max(0.1, dist * 0.5)
            grid_basis = self._live_grid_basis
        else:
            camera_probe = Camera(
                _PREVIEW_YAW, _PREVIEW_ELEV, _PREVIEW_DIST,
                roll_deg=_PREVIEW_ROLL, pan_px=(0.0, 0.0))
            hand_center = 0.5 * (mesh.min(axis=0) + mesh.max(axis=0))
            projected_center, _ = camera_probe.project(hand_center[None])
            pan_px = (
                640.0 - float(projected_center[0, 0]),
                360.0 - float(projected_center[0, 1]),
            )
            camera = Camera(
                _PREVIEW_YAW, _PREVIEW_ELEV, _PREVIEW_DIST,
                roll_deg=_PREVIEW_ROLL, pan_px=pan_px)
            grid_r = 0.12
            grid_basis = None
        # No in-image title / "no pose required" text here: those overlays are
        # drawn by the Qt layer as localized labels, and cv2's Hershey fonts
        # cannot render CJK.  Only the hand, grid and contact markers are drawn.
        # Contact previews intentionally reuse the left calibration pose for
        # both sides, so they also use the matching left-hand face topology.
        face_side = "left" if self._live_enabled and 8 <= key[1] <= 12 \
            else key[0]
        faces = self._mesh_faces[face_side]
        image = render_frame_live(
            camera, slots, [], None, grid_r=grid_r, grid_basis=grid_basis,
            mesh_slots=mesh_slots, mesh_faces=faces, display_mode="hand")
        if 8 <= key[1] <= 11:
            target_tip = (8, 12, 16, 20)[key[1] - 8]
            points, _ = camera.project(displayed[[4, target_tip]])
            for point in points:
                cv2.circle(
                    image, tuple(np.rint(point).astype(int)), 15,
                    (40, 210, 255), 3, cv2.LINE_AA)
        # Remove unused widescreen margins so the hand stays legible inside
        # the GUI panel, then return the BGR uint8 frame for a QImage display.
        image = image[55:685, 190:1090]
        image = cv2.resize(
            image, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_AREA)
        image = np.ascontiguousarray(image)
        self._image_cache[key] = image
        return image

    def render_bimanual_bgr(self, step_index: int) -> np.ndarray:
        """Return left/right full-hand references together in one panel.

        Each side is rendered independently before composition.  This is
        important for the mirrored root yaw and roll poses: duplicating one
        side's image would teach exactly the wrong motion to the other hand.
        """
        index = max(0, min(int(step_index), 12))
        cached = self._bimanual_image_cache.get(index)
        if cached is not None:
            return cached

        source = {
            side: self.render_bgr(side, index)
            for side in ("left", "right")
        }
        gap = 10
        panel_w = (PREVIEW_WIDTH - gap) // 2
        panel_h = 260
        panel_y = (PREVIEW_HEIGHT - panel_h) // 2
        canvas = np.empty((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), np.uint8)
        canvas[:] = (24, 31, 45)
        for side, x0, accent in (
                ("left", 0, (255, 180, 80)),
                ("right", panel_w + gap, (80, 180, 255))):
            image = source[side]
            # The single-hand renderer deliberately includes generous side
            # margins.  Remove them before scaling so both complete hands stay
            # large and legible in the shared preview area.
            crop_margin = max(0, (image.shape[1] - 500) // 2)
            cropped = image[:, crop_margin:image.shape[1] - crop_margin]
            panel = cv2.resize(
                cropped, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            canvas[panel_y:panel_y + panel_h, x0:x0 + panel_w] = panel
            cv2.rectangle(
                canvas, (x0, panel_y),
                (x0 + panel_w - 1, panel_y + panel_h - 1),
                accent, 2, cv2.LINE_AA)
            cv2.putText(
                canvas, "L" if side == "left" else "R",
                (x0 + 12, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, accent, 2, cv2.LINE_AA)

        canvas = np.ascontiguousarray(canvas)
        self._bimanual_image_cache[index] = canvas
        return canvas
