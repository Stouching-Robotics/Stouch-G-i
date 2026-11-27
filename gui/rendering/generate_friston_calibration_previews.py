"""Regenerate the packaged Friston wearing-calibration illustrations.

Run from the source checkout with the desktop Python environment:
    python -m gui.rendering.generate_friston_calibration_previews
"""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from algorithm.friston_rig import FristonRig, RIG_SHA256
from gui.calibration_pose_preview import PREVIEW_HEIGHT, PREVIEW_WIDTH, _REFERENCE_DIR
from gui.friston_display import skeleton_for_side
from gui.live_3d import Camera, _draw_hand_mesh


_SENSOR_ORDER = [0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]
_CONTACT_REFERENCE = [2, 3, 5, 4]
_REFLECT_X = np.diag([-1., 1., 1.])
_BACKGROUND = (38, 30, 23)


def _rotations_world(local, parents):
    world = Rotation.from_quat(local).as_matrix()
    for joint, parent in enumerate(parents[1:], start=1):
        world[joint] = world[parent] @ world[joint]
    return world


def _mirror_pose(pose, right, left):
    right_rest = _rotations_world(right.rest_local_xyzw, right.parents)
    left_rest = _rotations_world(left.rest_local_xyzw, left.parents)
    left_axes = right_rest.transpose(0, 2, 1) @ _REFLECT_X @ left_rest
    world = _REFLECT_X @ _rotations_world(pose, right.parents) @ left_axes
    local = world.copy()
    local[1:] = world[left.parents[1:]].transpose(0, 2, 1) @ world[1:]
    return Rotation.from_matrix(local).as_quat()


def _contact_pose(skeleton, authored, finger):
    """Close the illustrated thumb/finger tip gap using the same Friston rig."""
    rest = Rotation.from_quat(skeleton.rest_local_xyzw)
    reference = (rest.inv() * Rotation.from_quat(authored)).as_rotvec()
    base = 4 + 3 * finger
    joints = [1, 1, 1, 2, 3, base, base, base + 1, base + 2]
    axes = [0, 1, 2, 1, 1, 0, 1, 0, 0]
    seed = reference[joints, axes]
    low = np.deg2rad([-15, -105, -80, -15, -15, 0, -34, 0, 0])
    high = np.deg2rad([100, 30, 55, 90, 90, 95, 34, 120, 85])
    tip = 8 + 4 * finger

    def pose(angles):
        local = np.zeros((16, 3))
        local[joints, axes] = angles
        return (rest * Rotation.from_rotvec(local)).as_quat()

    def residual(angles):
        points = skeleton.keypoints_m(pose(angles))
        return np.r_[1_000. * (points[tip] - points[4]), 0.6 * (angles - seed)]

    fit = least_squares(residual, np.clip(seed, low, high), bounds=(low, high))
    result = pose(fit.x)
    points = skeleton.keypoints_m(result)
    if np.linalg.norm(points[tip] - points[4]) > 0.004:
        raise RuntimeError(f"Friston contact preview {finger} missed its target tip")
    return result


def _right_articulated_poses(right):
    rig = FristonRig("right")

    def authored(index):
        matrices = rig.forward(rig.reference_angles[index]).sensor_matrices
        return right.source_world_to_local_xyzw(matrices[_SENSOR_ORDER, :3, :3])

    poses = {8 + finger: _contact_pose(right, authored(index), finger)
             for finger, index in enumerate(_CONTACT_REFERENCE)}
    # Use the authored four-finger curl and leave the thumb in its relaxed pose.
    poses[13] = authored(6)
    poses[13][:4] = right.rest_local_xyzw[:4]
    return poses


def _step_pose(skeleton, poses, step):
    pose = np.array(poses.get(step, skeleton.rest_local_xyzw), copy=True)
    direction = 1 if skeleton.side == "right" else -1
    rotations = {1: ("x", -90), 2: ("x", 90),
                 3: ("z", 90 * direction), 4: ("y", 90 * direction)}
    if step in rotations:
        axis, angle = rotations[step]
        pose[0] = (Rotation.from_euler(axis, angle, degrees=True)
                   * Rotation.from_quat(pose[0])).as_quat()
    return pose


def _panel(side, step, vertices, faces, width, contact_point):
    sign = 1 if side == "right" else -1
    if step <= 4:
        yaw, elev = 180 + 33 * sign, 24
    elif 8 <= step <= 11:
        yaw, elev = 180 + 45 * sign, -34
    elif step == 13:
        yaw, elev = 180 + 113 * sign, -23
    else:
        yaw, elev = 180, 73

    size = (width, PREVIEW_HEIGHT)
    for dist in np.linspace(.28, .64, 37):
        camera = Camera(yaw, elev, dist, size=size)
        points, _ = camera.project(vertices)
        lo, hi = points.min(axis=0), points.max(axis=0)
        if np.all(hi - lo <= [width - 48, PREVIEW_HEIGHT - 42]):
            break
    camera.pan_px = ((np.asarray(size) - hi - lo) * .5).tolist()
    image = np.full((PREVIEW_HEIGHT, width, 3), _BACKGROUND, dtype=np.uint8)
    if step <= 4:
        # Show the horizontal wrist reference plane for the four rigid poses.
        for value in np.linspace(-.14, .14, 7):
            for segment in (((value, 0., -.14), (value, 0., .14)),
                            ((-.14, 0., value), (.14, 0., value))):
                xy, depth = camera.project(np.asarray(segment))
                if np.all(depth > 0.):
                    a, b = np.rint(xy).astype(int)
                    cv2.line(image, tuple(a), tuple(b), (65, 55, 46), 1, cv2.LINE_AA)
    if not _draw_hand_mesh(image, camera, vertices, faces):
        raise RuntimeError(f"Could not draw {side} Friston step {step}")
    if contact_point is not None:
        xy, _ = camera.project(contact_point[None, :])
        center = tuple(np.rint(xy[0]).astype(int))
        cv2.circle(image, center, 6, _BACKGROUND, -1, cv2.LINE_AA)
        cv2.circle(image, center, 4, (123, 208, 248), -1, cv2.LINE_AA)
    return image


def main():
    left, right = (skeleton_for_side(side) for side in ("left", "right"))
    right_poses = _right_articulated_poses(right)
    left_poses = {step: _mirror_pose(pose, right, left) for step, pose in right_poses.items()}
    _REFERENCE_DIR.mkdir(exist_ok=True, parents=True)
    hashes = {}

    def save(name, image):
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise RuntimeError(f"Could not encode {name}")
        data = encoded.tobytes()
        (_REFERENCE_DIR / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()

    for step in (0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 13):
        panels = []
        for skeleton, poses in ((left, left_poses), (right, right_poses)):
            side = skeleton.side
            pose = _step_pose(skeleton, poses, step)
            vertices = skeleton.mesh_vertices_m(pose)
            contact_point = None
            if 8 <= step <= 11:
                points = skeleton.keypoints_m(pose)
                tip = 8 + 4 * (step - 8)
                contact_point = (points[4] + points[tip]) / 2
            save(f"single_{side}_step_{step:02d}.png",
                 _panel(side, step, vertices, skeleton.faces, PREVIEW_WIDTH, contact_point))
            panels.append(_panel(side, step, vertices, skeleton.faces,
                                 PREVIEW_WIDTH // 2, contact_point))
        dual = np.hstack(panels)
        dual[:, PREVIEW_WIDTH // 2] = (65, 55, 46)
        save(f"dual_step_{step:02d}.png", dual)
    (_REFERENCE_DIR / "manifest.json").write_text(json.dumps({
        "rig": "Friston",
        "rig_reference_sha256": RIG_SHA256,
        "png_sha256": hashes,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
