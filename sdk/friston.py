"""Friston's public 16-joint skeleton and metre-scale linear blend skinning."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["FristonSkeleton", "load_skeleton"]

_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "friston"
_SCHEMA_VERSION = 1
# Rotation of the reference display world into the public world.
_SOURCE_TO_PUBLIC = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
_TIP_JOINTS = np.array([3, 6, 9, 12, 15])
_KEYPOINT_JOINTS = np.array([
    0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15, 15,
])
_TIP_KEYPOINTS = np.array([4, 8, 12, 16, 20])


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


@dataclass(frozen=True, eq=False)
class FristonSkeleton:
    """A fixed hand rig. All arrays loaded from the asset are read-only.

    The pose order is wrist, followed by three joints per finger from thumb
    through little finger. Poses are full XYZW orientations. The wrist is
    absolute and the other orientations are relative to their respective parent.
    """

    side: str
    joint_names: tuple[str, ...]
    joint_paths: tuple[str, ...]
    parents: np.ndarray
    rest_local_offsets_m: np.ndarray
    rest_local_xyzw: np.ndarray
    source_rest_world: np.ndarray
    tip_offsets_m: np.ndarray
    bind_world_m: np.ndarray
    vertices_m: np.ndarray
    faces: np.ndarray
    skin_joint_indices: np.ndarray
    skin_weights: np.ndarray
    _source_axes: np.ndarray
    _inverse_bind: np.ndarray

    def source_world_to_local_xyzw(self, world) -> np.ndarray:
        """Convert 16 absolute reference bone rotations to a public pose.

        Use the wrist, thumb, index, middle, ring, little ordering and the
        side-matched Friston display world (+Y fingers, +Z dorsal). Each
        matrix acts on column vectors. Right-hand reference input already
        includes the reference rig's X-reflection conjugation.
        """
        source = _finite_array(world, (16, 3, 3), "world")
        u, singular, vh = np.linalg.svd(source)
        source_rotation = u @ vh
        if np.any(np.abs(singular - 1.) > 1e-4) or np.any(np.linalg.det(source_rotation) < 0.):
            raise ValueError("world must contain proper rotation matrices")
        # The reference frames retain low-level rounding from the authored rig.
        absolute = _SOURCE_TO_PUBLIC @ source_rotation @ self._source_axes
        local = absolute.copy()
        local[1:] = absolute[self.parents[1:]].transpose(0, 2, 1) @ absolute[1:]
        return Rotation.from_matrix(local).as_quat()

    def _joint_world(self, joint_xyzw):
        quat = _finite_array(joint_xyzw, (16, 4), "joint_xyzw")
        scale = np.max(np.abs(quat), axis=1, keepdims=True)
        if np.any(scale == 0.):
            raise ValueError("joint_xyzw must contain nonzero quaternions")
        world = np.broadcast_to(np.eye(4), (16, 4, 4)).copy()
        world[:, :3, :3] = Rotation.from_quat(quat / scale).as_matrix()
        world[:, :3, 3] = self.rest_local_offsets_m
        for joint, parent in enumerate(self.parents[1:], start=1):
            world[joint] = world[parent] @ world[joint]
        return world

    def keypoints_m(self, joint_xyzw) -> np.ndarray:
        """Return the wrist and four points per finger, in metres."""
        world = self._joint_world(joint_xyzw)
        points = world[_KEYPOINT_JOINTS, :3, 3]
        tip_frames = world[_TIP_JOINTS, :3, :3]
        points[_TIP_KEYPOINTS] += np.einsum("tij,tj->ti", tip_frames, self.tip_offsets_m)
        return points

    def mesh_vertices_m(self, joint_xyzw) -> np.ndarray:
        """Return the posed mesh vertices in metres, in the order used by faces."""
        skin = self._joint_world(joint_xyzw) @ self._inverse_bind
        weighted = np.einsum(
            "vk,vkij->vij", self.skin_weights, skin[self.skin_joint_indices, :3, :]
        )
        return np.einsum("vij,vj->vi", weighted[:, :, :3], self.vertices_m) + weighted[:, :, 3]


def load_skeleton(side: str) -> FristonSkeleton:
    """Load the shipped left or right rig from its NumPy asset."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    return _load_skeleton(side)


@cache
def _load_skeleton(side: str) -> FristonSkeleton:
    with np.load(_ASSET_DIR / f"{side}_hand.npz", allow_pickle=False) as asset:
        if asset["schema_version"].item() != _SCHEMA_VERSION or asset["side"].item() != side:
            raise ValueError(f"incompatible Friston skeleton asset for {side}")
        arrays = {name: asset[name] for name in (
            "parents", "rest_local_offsets_m", "rest_local_xyzw", "source_rest_world",
            "tip_offsets_m", "bind_world_m", "vertices_m", "faces", "skin_joint_indices",
            "skin_weights",
        )}
        arrays["_source_axes"] = asset["source_axes"]
        arrays["_inverse_bind"] = np.linalg.inv(arrays["bind_world_m"])
        for array in arrays.values():
            array.setflags(write=False)
        return FristonSkeleton(
            side=side,
            joint_names=tuple(asset["joint_names"].tolist()),
            joint_paths=tuple(asset["joint_paths"].tolist()),
            **arrays,
        )
