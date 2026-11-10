"""Low-poly arrow templates deformed along live-to-target rigid trajectories."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


AXIS_LENGTH_RADIUS = 0.65
_BODY_SEGMENTS = 8
_BODY_END = 0.8
_MIN_ENDPOINT_ANGLE = np.deg2rad(0.5)


def _arrow_template() -> tuple[np.ndarray, np.ndarray]:
    """Build one closed straight arrow, shared by every axis and every frame."""
    corners = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
    rings = [np.column_stack((np.full(4, u), corners))
             for u in np.linspace(0., _BODY_END, _BODY_SEGMENTS + 1)]
    rings.append(np.column_stack((np.full(4, _BODY_END),
                                  corners * [0.090 / 0.035, 0.035 / 0.020])))
    vertices = np.vstack((*rings, [[1., 0., 0.]]))
    faces = []
    for ring in range(_BODY_SEGMENTS):
        for corner in range(4):
            a, b = 4 * ring + corner, 4 * ring + (corner + 1) % 4
            faces.extend(((a, b, b + 4), (a, b + 4, a + 4)))
    faces.extend(((0, 2, 1), (0, 3, 2)))  # Flat tail, no reverse head.
    small, wide = 4 * _BODY_SEGMENTS, 4 * (_BODY_SEGMENTS + 1)
    tip = len(vertices) - 1
    for corner in range(4):
        nxt = (corner + 1) % 4
        faces.extend(((small + corner, small + nxt, wide + nxt),
                      (small + corner, wide + nxt, wide + corner),
                      (wide + corner, wide + nxt, tip)))
    faces = np.asarray(faces, dtype=np.int64)
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return vertices, faces


_TEMPLATE, _FACES = _arrow_template()


def _proper_rotation(value: object, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite 3x3 proper rotation") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 proper rotation")
    if (not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0., atol=1e-6)
            or not np.isclose(np.linalg.det(matrix), 1., rtol=0., atol=1e-6)):
        raise ValueError(f"{name} must be a proper rotation")
    return matrix


def _shortest_local_rotvec(live: np.ndarray, target: np.ndarray) -> np.ndarray:
    quaternion = Rotation.from_matrix(live.T @ target).as_quat()
    # At exactly pi both turns are equally short. Choose in the live frame,
    # so an arbitrary world/view rotation cannot reverse this tie-break.
    if abs(quaternion[3]) <= 32 * np.finfo(float).eps:
        quaternion[3] = 0.
        magnitudes = np.abs(quaternion[:3])
        pivot = np.flatnonzero(magnitudes >= magnitudes.max() - 1e-12)[0]
        if quaternion[pivot] < 0.:
            quaternion = -quaternion
    elif quaternion[3] < 0.:
        quaternion = -quaternion
    sine = np.linalg.norm(quaternion[:3])
    if sine < 1e-15:
        return np.zeros(3)
    return quaternion[:3] * (2 * np.arctan2(sine, quaternion[3]) / sine)


def rotation_arrow_meshes(
    live: np.ndarray, target: np.ndarray, hand_radius: float,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Deform a reusable arrow onto each moving axis's actual circular path.

    For the single shortest rigid rotation ``live -> target``, axis i follows
    ``R(t)[:, i] * AXIS_LENGTH_RADIUS * hand_radius``. The tail centre is the
    live endpoint and the sole apex is the target endpoint. All three paths
    therefore describe the same motion, including compound and large turns.
    An unchanged endpoint (e.g. X during a pure X turn) has no arrow.

    Each returned mesh is (axis index, 41 vertices, the shared 78 triangles).
    Only vertex positions change per frame; there is no per-frame remeshing,
    topology traversal or GPU requirement.
    """
    live = _proper_rotation(live, "live")
    target = _proper_rotation(target, "target")
    try:
        radius_array = np.asarray(hand_radius)
        if radius_array.ndim != 0:
            raise ValueError
        radius = float(radius_array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("hand_radius must be a positive finite scalar") from exc
    if not np.isfinite(radius) or radius <= 0.:
        raise ValueError("hand_radius must be a positive finite scalar")

    rotvec = _shortest_local_rotvec(live, target)
    angle = np.linalg.norm(rotvec)
    if angle < _MIN_ENDPOINT_ANGLE:
        return []
    rotation_axis = live @ (rotvec / angle)
    length = AXIS_LENGTH_RADIUS * radius
    result = []
    for axis in range(3):
        start, goal = live[:, axis] * length, target[:, axis] * length
        endpoint_angle = 2 * np.arcsin(np.clip(np.linalg.norm(goal - start) / (2 * length), 0., 1.))
        if endpoint_angle < _MIN_ENDPOINT_ANGLE:
            continue
        centre = np.dot(start, rotation_axis) * rotation_axis
        radial_start = start - centre
        arc_radius = np.linalg.norm(radial_start)
        arc_length = arc_radius * angle
        head_length = min(0.10 * radius, 0.30 * arc_length)
        head_base_t = 1. - head_length / arc_length
        parameters = np.minimum(_TEMPLATE[:, 0] / _BODY_END, 1.) * head_base_t
        parameters[-1] = 1.
        theta = parameters * angle
        radial = (np.cos(theta)[:, None] * radial_start
                  + np.sin(theta)[:, None] * np.cross(rotation_axis, radial_start))
        curve = centre + radial
        # The moving frame (tangent, radial, -rotation_axis) is right-handed.
        # Shrink cross-sections for tiny arcs to keep the head and shaft clear.
        half_width = min(0.035 * radius, 0.10 * arc_radius, 0.10 * arc_length)
        half_depth = min(0.020 * radius, 0.10 * arc_radius, 0.07 * arc_length)
        vertices = (curve + (_TEMPLATE[:, 1] * half_width / arc_radius)[:, None] * radial
                    - (_TEMPLATE[:, 2] * half_depth)[:, None] * rotation_axis)
        vertices[-1] = goal  # Exact endpoint despite round-off near pi.
        result.append((axis, vertices, _FACES))
    return result
