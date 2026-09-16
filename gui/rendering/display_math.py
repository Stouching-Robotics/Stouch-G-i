"""Shared MANO 21-keypoint display math (pure NumPy/SciPy, no torch).

These transforms turn solved hand keypoints into the on-screen pose used by
both the live 3D viewers and the offline replay renderer.  They live here so
that :mod:`gui.rendering.replay` can import them without dragging in the live
viewer's torch/manotorch/Qt dependency graph; :mod:`gui.live_3d` and
:mod:`gui.live_3d_bimanual` re-export the same symbols, so the live view and
replay stay pixel-for-pixel identical from a single source of truth.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


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


def _validate_palm_target_bases(value) -> np.ndarray:
    targets = np.asarray(value, dtype=float).reshape(2, 3, 3)
    if not np.isfinite(targets).all():
        raise RuntimeError("Bimanual palm bases contain non-finite values")
    for slot, basis in enumerate(targets):
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5):
            raise RuntimeError(f"Bimanual palm base slot {slot} is not orthogonal")
        if not np.isclose(np.linalg.det(basis), 1.0, atol=1e-5):
            raise RuntimeError(f"Bimanual palm base slot {slot} is not a rotation")
    return targets


def stabilize_bimanual_palm_bases(
        hands: np.ndarray, target_bases: np.ndarray) -> np.ndarray:
    """Remove independent root drift while preserving articulated geometry.

    Each hand receives exactly one proper rigid rotation about its wrist.  The
    rotation maps the current non-thumb MCP palm basis onto the corresponding
    startup target basis; thumb/finger articulation and all within-hand
    distances are otherwise untouched.
    """
    slots = np.asarray(hands, dtype=np.float32).reshape(2, 21, 3).copy()
    targets = np.asarray(target_bases, dtype=float).reshape(2, 3, 3)
    for slot in range(2):
        current_basis = hand_display_basis(slots[slot])
        matrix = targets[slot] @ current_basis.T
        if np.linalg.det(matrix) < 0.0:
            raise ValueError("bimanual palm stabilization produced a reflection")
        slots[slot] = apply_hand_display_rotation(
            slots[slot], R.from_matrix(matrix))
    return slots


def fixed_bimanual_display_rotations(
        hands: np.ndarray, target_bases: np.ndarray) -> list[R]:
    """Compute one startup rotation per hand; later root motion is preserved."""
    slots = np.asarray(hands, dtype=np.float32).reshape(2, 21, 3)
    targets = _validate_palm_target_bases(target_bases)
    rotations = []
    for slot in range(2):
        current_basis = hand_display_basis(slots[slot])
        matrix = targets[slot] @ current_basis.T
        if np.linalg.det(matrix) < 0.0:
            raise ValueError("fixed bimanual display alignment produced a reflection")
        rotations.append(R.from_matrix(matrix))
    return rotations


def remap_hand_root_motion(
        hand: np.ndarray, target_basis: np.ndarray,
        local_rotvec_sign: np.ndarray) -> np.ndarray:
    """Remap root-rotation components without changing finger articulation."""
    shown = np.asarray(hand, dtype=np.float32).reshape(21, 3)
    target = np.asarray(target_basis, dtype=float).reshape(3, 3)
    signs = np.asarray(local_rotvec_sign, dtype=float).reshape(3)
    current = hand_display_basis(shown)
    delta_world = current @ target.T
    delta_local = target.T @ delta_world @ target
    local_rotvec = R.from_matrix(delta_local).as_rotvec()
    corrected_local = R.from_rotvec(local_rotvec * signs).as_matrix()
    corrected_world = target @ corrected_local @ target.T
    correction = corrected_world @ delta_world.T
    return apply_hand_display_rotation(shown, R.from_matrix(correction))
