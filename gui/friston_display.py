"""Friston geometry shared by the desktop renderers."""

from functools import cache

import numpy as np


@cache
def skeleton_for_side(side: str):
    # Keep the SDK geometry import lazy so keypoint-only recordings can play
    # without opening the rig or loading its resources.
    from sdk.friston import load_skeleton

    return load_skeleton(side)


def mesh_array(vertices) -> np.ndarray | None:
    if vertices is None:
        return None
    points = np.asarray(vertices, dtype=np.float32)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all()):
        return None
    return points


def mesh_for_pose(side: str, joint_xyzw) -> np.ndarray | None:
    if joint_xyzw is None:
        return None
    return mesh_array(skeleton_for_side(side).mesh_vertices_m(joint_xyzw))
