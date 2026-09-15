"""Low-overhead pressure markers for the live MANO/HAND mesh view.

The pressure matrix is kept in the firmware orientation at the viewer
boundary.  This module converts it to the V1.4 canonical orientation, maps the
195 active cells to neutral HAND mesh faces once, and evaluates those same
face barycentric coordinates on each live mesh.  Geometry is deliberately
independent of pressure magnitude: an active cell gets one fixed-size,
fixed-height marker and pressure only selects its colour.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from algorithm.lite.hand import HandNumpy


# Keep the 3D marker palette aligned with the compact 2D matrix card, but do
# not import that view's private implementation details.  This keeps the
# pressure-on-mesh path independent from future card-layout changes.
_PRESSURE_BANDS_CORRECTED = (
    (333, (95, 58, 30)),
    (666, (199, 134, 22)),
    (999, (94, 197, 34)),
    (1333, (21, 204, 250)),
    (1666, (22, 115, 249)),
    (float("inf"), (68, 68, 239)),
)
_PRESSURE_BANDS_RAW = (
    (2000, (95, 58, 30)),
    (2400, (199, 134, 22)),
    (2800, (94, 197, 34)),
    (3200, (21, 204, 250)),
    (3600, (22, 115, 249)),
    (float("inf"), (68, 68, 239)),
)


MATRIX_ROWS = 16
MATRIX_COLS = 16
_VALID_CELLS = np.asarray(
    [(x, y) for x in range(1, 16) for y in range(3, 12)]
    + [(x, y) for x in range(1, 16) for y in range(12, 16)],
    dtype=np.int32,
)

# V1.4's measured fingertip layout, in the oriented canonical hand frame.
_FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
_FINGERTIP_ALONG_FRACTIONS = (0.50, 0.66, 0.82, 0.96)
_FINGERTIP_WIDTH_SCALES = (0.62, 0.57, 0.49, 0.36)
_FINGER_RADII_M = (0.0105, 0.0092, 0.0098, 0.0091, 0.0079)

# These are intentionally fixed in screen space.  Pressure does not alter
# either the apparent marker size or the protrusion distance.
_MARKER_RADIUS_PX = 3
_PROTRUSION_M = 0.0020

# Low-pressure cells start as a warm, low-saturation grey close to the hand
# mesh.  Increasing pressure then moves through a restrained blue -> cyan ->
# green -> yellow -> red heat map in BGR order.
_HEAT_STOPS = np.asarray([0.0, 0.20, 0.40, 0.60, 0.80, 1.0], dtype=np.float32)
_HEAT_BGR = np.asarray([
    (185.0, 180.0, 175.0),
    (190.0, 110.0, 75.0),
    (205.0, 180.0, 80.0),
    (80.0, 175.0, 90.0),
    (55.0, 190.0, 210.0),
    (45.0, 45.0, 235.0),
], dtype=np.float32)


def _orient(points: np.ndarray, side: str) -> np.ndarray:
    """Convert native HAND coordinates to V1.4's palm-facing frame."""

    value = np.asarray(points, dtype=np.float64)
    sign = 1.0 if side == "right" else -1.0
    result = np.empty_like(value)
    result[..., 0] = sign * value[..., 2]
    result[..., 1] = -sign * value[..., 0]
    result[..., 2] = -value[..., 1]
    return result


def _deorient(points: np.ndarray, side: str) -> np.ndarray:
    """Convert V1.4's palm-facing frame back to native HAND coordinates."""

    value = np.asarray(points, dtype=np.float64)
    sign = 1.0 if side == "right" else -1.0
    result = np.empty_like(value)
    result[..., 0] = -sign * value[..., 1]
    result[..., 1] = -value[..., 2]
    result[..., 2] = sign * value[..., 0]
    return result


def canonical_pressure_matrix(matrix: np.ndarray, side: str) -> np.ndarray:
    """Return the V1.4 ``[x, y]`` matrix for a raw firmware-oriented frame."""

    values = np.asarray(matrix, dtype=np.float32).reshape(MATRIX_ROWS, MATRIX_COLS)
    if str(side).lower() == "right":
        return values
    # This is the same left-glove correction used by V1.5's LiveDataModel.
    return np.ascontiguousarray(values[::-1, ::-1].T)


def _pressure_color(value: np.ndarray, baseline_corrected: bool) -> np.ndarray:
    bands = (_PRESSURE_BANDS_CORRECTED if baseline_corrected
             else _PRESSURE_BANDS_RAW)
    upper = np.asarray([item[0] for item in bands], dtype=np.float32)
    palette = np.asarray([item[1] for item in bands], dtype=np.uint8)
    index = np.searchsorted(upper, np.maximum(value, 0.0), side="left")
    return palette[np.clip(index, 0, len(palette) - 1)]


def _heat_colors(value: np.ndarray, baseline_corrected: bool) -> np.ndarray:
    values = np.maximum(0.0, np.asarray(value, dtype=np.float32))
    # Raw frames are around 2000 at rest; baseline-corrected frames are around
    # zero.  Keeping this reference fixed means an unloaded cell remains grey
    # instead of being promoted to blue just because another cell is pressed.
    reference = 0.0 if baseline_corrected else 2000.0
    span = 1800.0 if baseline_corrected else 2095.0
    level = (values - reference) / span
    level = np.clip(level, 0.0, 1.0)
    return np.stack([
        np.interp(level, _HEAT_STOPS, _HEAT_BGR[:, channel])
        for channel in range(3)
    ], axis=-1).astype(np.uint8)


def _heat_display_values(
        values: np.ndarray, baseline_corrected: bool,
        threshold: float) -> np.ndarray:
    """Keep every cell visible while making sub-threshold cells grey."""

    values = np.maximum(0.0, np.asarray(values, dtype=np.float32))
    peak = float(np.max(values, initial=0.0))
    baseline = float(np.percentile(values, 20.0))
    contact_floor = baseline + max(60.0, max(1.0, peak - baseline) * 0.16)
    if threshold > 0.0:
        contact_floor = max(contact_floor, float(threshold))
    reference = 0.0 if baseline_corrected else 2000.0
    return np.where(values > contact_floor, values, reference)


def _barycentric_projection(
        point: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a point onto candidate triangles and return points + barycentrics."""

    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = point[None, :] - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    denominator = d00 * d11 - d01 * d01
    safe = np.abs(denominator) > 1e-12
    v = np.zeros_like(denominator)
    w = np.zeros_like(denominator)
    v[safe] = (d11[safe] * d20[safe] - d01[safe] * d21[safe]) / denominator[safe]
    w[safe] = (d00[safe] * d21[safe] - d01[safe] * d20[safe]) / denominator[safe]
    u = 1.0 - v - w
    barycentric = np.stack([u, v, w], axis=1)
    barycentric = np.clip(barycentric, 0.0, 1.0)
    barycentric /= np.maximum(barycentric.sum(axis=1, keepdims=True), 1e-12)
    projected = np.einsum("ij,ijk->ik", barycentric, triangles)
    return projected, barycentric


class PressureHandMapper:
    """Cached V1.4 pressure-cell anchors for one HAND mesh side."""

    def __init__(self, side: str, faces: np.ndarray, assets_root: Path):
        self.side = str(side).lower()
        if self.side not in ("left", "right"):
            raise ValueError(f"invalid hand side: {side!r}")
        self.faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        if self.faces.size == 0:
            raise ValueError("MANO/HAND faces are empty")

        hand = HandNumpy(side=self.side, npz_path=str(assets_root))
        neutral = hand.forward(
            np.zeros((16, 3), dtype=np.float64),
            np.zeros(10, dtype=np.float64),
        )
        self.neutral_vertices = np.asarray(neutral.verts, dtype=np.float64)
        oriented_vertices = _orient(self.neutral_vertices, self.side)
        oriented_joints = _orient(np.asarray(neutral.joints), self.side)
        targets_oriented = self._build_targets(oriented_vertices, oriented_joints)
        targets_native = _deorient(targets_oriented, self.side)
        self.face_indices, self.barycentric, self.normal_sign = (
            self._anchor_targets(targets_native))
        self.cell_indices = _VALID_CELLS.copy()
        self.grid_lookup = np.full((MATRIX_ROWS, MATRIX_COLS), -1, dtype=np.int32)
        for index, (x, y) in enumerate(self.cell_indices):
            self.grid_lookup[x, y] = index
        self._tile_x_neighbors, self._tile_y_neighbors = (
            self._build_tile_neighbors())

    def _build_tile_neighbors(self) -> tuple[np.ndarray, np.ndarray]:
        """Build local matrix neighbors without crossing finger boundaries."""

        x_neighbors = np.empty(len(self.cell_indices), dtype=np.int32)
        y_neighbors = np.empty(len(self.cell_indices), dtype=np.int32)
        for index, (x, y) in enumerate(self.cell_indices):
            x = int(x)
            y = int(y)
            if y <= 11:
                x_min, x_max = 1, 15
            else:
                x_min = 1 + ((x - 1) // 3) * 3
                x_max = x_min + 2
            x_neighbor = x + 1 if x < x_max else x - 1
            y_neighbor = y + 1 if y < (11 if y <= 11 else 15) else y - 1
            x_index = int(self.grid_lookup[x_neighbor, y])
            y_index = int(self.grid_lookup[x, y_neighbor])
            if x_index < 0 or y_index < 0:
                raise RuntimeError(f"missing tile neighbor for pressure cell {(x, y)}")
            x_neighbors[index] = x_index
            y_neighbors[index] = y_index
        return x_neighbors, y_neighbors

    @staticmethod
    def _surface_position(vertices: np.ndarray, xy: np.ndarray) -> np.ndarray:
        delta = vertices[:, :2] - np.asarray(xy, dtype=np.float64)
        nearest_count = min(12, len(vertices))
        nearest = np.argpartition(
            np.einsum("ij,ij->i", delta, delta), nearest_count - 1
        )[:nearest_count]
        surface_z = float(np.max(vertices[nearest, 2])) + 0.0012
        return np.array([xy[0], xy[1], surface_z], dtype=np.float64)

    def _build_targets(
            self, vertices: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """Port V1.4's 15x9 palm and five 3x4 fingertip layouts."""

        targets: list[np.ndarray] = []
        wrist_xy = joints[0, :2]
        thumb_root = joints[1, :2]
        index_root = joints[5, :2]
        middle_root = joints[9, :2]
        little_root = joints[17, :2]
        thumb_side_sign = 1.0 if thumb_root[0] > little_root[0] else -1.0

        heel_y = float(wrist_xy[1]) + 0.018
        heel_band = vertices[np.abs(vertices[:, 1] - heel_y) <= 0.010]
        if len(heel_band) >= 4:
            heel_low_x, heel_high_x = np.percentile(heel_band[:, 0], (12.0, 88.0))
        else:
            heel_low_x = float(wrist_xy[0]) - 0.034
            heel_high_x = float(wrist_xy[0]) + 0.034
        thumb_is_low_x = thumb_root[0] < little_root[0]
        heel_thumb = np.array([
            heel_low_x if thumb_is_low_x else heel_high_x, heel_y])
        heel_little = np.array([
            heel_high_x if thumb_is_low_x else heel_low_x, heel_y])
        upper_thumb = np.array([
            float(index_root[0]) + thumb_side_sign * 0.025,
            float(thumb_root[1]) * 0.50 + float(index_root[1]) * 0.50,
        ])
        upper_middle = middle_root.astype(np.float64, copy=True)
        upper_little = np.array([
            float(little_root[0]) - thumb_side_sign * 0.007,
            float(little_root[1]),
        ])

        for x in range(1, 16):
            fraction = (x - 1.0) / 14.0
            lower = heel_thumb * (1.0 - fraction) + heel_little * fraction
            inverse = 1.0 - fraction
            upper = (
                upper_thumb * inverse * inverse
                + upper_middle * 2.0 * inverse * fraction
                + upper_little * fraction * fraction)
            for y in range(3, 12):
                row_fraction = (y - 3.0) / 8.0
                xy = lower * (1.0 - row_fraction) + upper * row_fraction
                targets.append(self._surface_position(vertices, xy))

        for group, joint_indices in enumerate(_FINGER_CHAINS):
            start_x = 1 + group * 3
            chain = joints[np.asarray(joint_indices), :2]
            distal_joint = chain[-2]
            tip = chain[-1]
            tangent = tip - distal_joint
            tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
            across = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            radius = _FINGER_RADII_M[group]
            for x in range(start_x, start_x + 3):
                across_offset = (x - start_x - 1.0)
                for y, along_fraction, width_scale in zip(
                        range(12, 16), _FINGERTIP_ALONG_FRACTIONS,
                        _FINGERTIP_WIDTH_SCALES):
                    del y
                    xy = (
                        distal_joint * (1.0 - along_fraction)
                        + tip * along_fraction
                        + across * (across_offset * radius * width_scale))
                    targets.append(self._surface_position(vertices, xy))
        result = np.asarray(targets, dtype=np.float64)
        if result.shape != (len(_VALID_CELLS), 3):
            raise RuntimeError(f"unexpected pressure target count: {result.shape}")
        return result

    def _anchor_targets(
            self, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        face_centres = self.neutral_vertices[self.faces].mean(axis=1)
        face_indices = np.empty(len(targets), dtype=np.int32)
        barycentric = np.empty((len(targets), 3), dtype=np.float64)
        normal_sign = np.ones(len(targets), dtype=np.float64)
        oriented_faces = _orient(self.neutral_vertices[self.faces], self.side)

        for index, target in enumerate(targets):
            nearest_vertex = int(np.argmin(
                np.einsum("ij,ij->i", self.neutral_vertices - target, 
                           self.neutral_vertices - target)))
            candidates = np.flatnonzero(
                np.any(self.faces == nearest_vertex, axis=1))
            if len(candidates) == 0:
                candidates = np.arange(len(self.faces), dtype=np.int32)
            candidate_triangles = self.neutral_vertices[self.faces[candidates]]
            projected, candidate_bary = _barycentric_projection(
                target, candidate_triangles)
            distances = np.einsum(
                "ij,ij->i", projected - target, projected - target)
            chosen = int(np.argmin(distances))
            face_index = int(candidates[chosen])
            face_indices[index] = face_index
            barycentric[index] = candidate_bary[chosen]

            normal = np.cross(
                oriented_faces[face_index, 1] - oriented_faces[face_index, 0],
                oriented_faces[face_index, 2] - oriented_faces[face_index, 0],
            )
            if float(normal[2]) < 0.0:
                normal_sign[index] = -1.0
        return face_indices, barycentric, normal_sign

    def points_on_mesh(self, mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate cached anchors and outward normals on a live mesh."""

        vertices = np.asarray(mesh, dtype=np.float32).reshape(-1, 3)
        triangles = vertices[self.faces[self.face_indices]]
        bases = np.einsum("ij,ijk->ik", self.barycentric, triangles)
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-9)
        normals *= self.normal_sign[:, None]
        return bases, normals

    def tile_corners_on_mesh(self, mesh: np.ndarray) -> tuple[
            np.ndarray, np.ndarray, np.ndarray]:
        """Return one small, independent surface tile for every pressure cell.

        The x/y axes come from neighboring cells in the same palm or finger
        region.  Tiles are deliberately inset from their centers, leaving a
        visible gap so neighboring pressure cells never become one connected
        patch.
        """

        bases, normals = self.points_on_mesh(mesh)
        x_delta = bases[self._tile_x_neighbors] - bases
        y_delta = bases[self._tile_y_neighbors] - bases
        x_delta -= normals * np.einsum("ij,ij->i", x_delta, normals)[:, None]
        y_delta -= normals * np.einsum("ij,ij->i", y_delta, normals)[:, None]
        x_lengths = np.linalg.norm(x_delta, axis=1)
        y_lengths = np.linalg.norm(y_delta, axis=1)
        x_axis = x_delta / np.maximum(x_lengths[:, None], 1e-9)
        y_axis = y_delta - x_axis * np.einsum(
            "ij,ij->i", y_delta, x_axis)[:, None]
        y_axis -= normals * np.einsum("ij,ij->i", y_axis, normals)[:, None]
        y_lengths = np.linalg.norm(y_axis, axis=1)
        y_axis /= np.maximum(y_lengths[:, None], 1e-9)
        half_x = np.maximum(0.0008, x_lengths * 0.34)
        half_y = np.maximum(0.0008, y_lengths * 0.34)
        centers = bases + normals * np.float32(0.0003)
        corners = np.stack([
            centers + x_axis * half_x[:, None] + y_axis * half_y[:, None],
            centers - x_axis * half_x[:, None] + y_axis * half_y[:, None],
            centers - x_axis * half_x[:, None] - y_axis * half_y[:, None],
            centers + x_axis * half_x[:, None] - y_axis * half_y[:, None],
        ], axis=1)
        return corners, centers, normals


def draw_pressure_color_points(
        image: np.ndarray,
        camera,
        mesh_slots: list[np.ndarray | None],
        pressure_frames: dict[str, np.ndarray | None],
        mappers: dict[str, PressureHandMapper],
        baseline_ready: dict[str, bool],
        threshold: float = 0.0,
) -> None:
    """Draw one independent colour point per pressure cell.

    Every mapped cell remains visible as a quiet grey point at rest.  This mode
    intentionally does not extrude, fill faces, blur, or connect neighboring
    cells.  The marker-only mode remains responsible for the fixed protrusion
    visualization when surface colouring is disabled.
    """

    for slot, side in enumerate(("left", "right")):
        mapper = mappers.get(side)
        mesh = mesh_slots[slot] if slot < len(mesh_slots) else None
        frame = pressure_frames.get(side)
        if mapper is None or mesh is None:
            continue
        vertices = np.asarray(mesh, dtype=np.float32).reshape(-1, 3)
        if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
            continue
        if frame is None:
            cell_values = np.zeros(len(mapper.cell_indices), dtype=np.float32)
        else:
            values = canonical_pressure_matrix(frame, side)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            cell_values = np.maximum(
                0.0, values[mapper.cell_indices[:, 0], mapper.cell_indices[:, 1]])
        display_values = _heat_display_values(
            cell_values, baseline_ready.get(side, False), threshold)
        bases, normals = mapper.points_on_mesh(vertices)
        visible = np.einsum(
            "ij,ij->i", normals,
            np.asarray(camera.pos, dtype=np.float32)[None, :] - bases) > 0.0
        projected, depth = camera.project(bases)
        valid = (visible
                 & np.isfinite(projected).all(axis=1)
                 & np.isfinite(depth) & (depth > 0.0))
        if not np.any(valid):
            continue
        colors = _heat_colors(
            display_values, baseline_ready.get(side, False))
        order = np.argsort(depth, kind="stable")[::-1]
        for index in order:
            if not valid[index]:
                continue
            point = tuple(np.rint(projected[index]).astype(int))
            color = tuple(int(value) for value in colors[index])
            cv2.circle(image, point, _MARKER_RADIUS_PX, color, -1, cv2.LINE_AA)


def draw_pressure_tiles(
        image: np.ndarray,
        camera,
        mesh_slots: list[np.ndarray | None],
        pressure_frames: dict[str, np.ndarray | None],
        mappers: dict[str, PressureHandMapper],
        baseline_ready: dict[str, bool],
        threshold: float = 0.0,
) -> None:
    """Draw independent, number-free pressure-matrix tiles on the mesh."""

    for slot, side in enumerate(("left", "right")):
        mapper = mappers.get(side)
        mesh = mesh_slots[slot] if slot < len(mesh_slots) else None
        frame = pressure_frames.get(side)
        if mapper is None or mesh is None:
            continue
        vertices = np.asarray(mesh, dtype=np.float32).reshape(-1, 3)
        if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
            continue
        if frame is None:
            cell_values = np.zeros(len(mapper.cell_indices), dtype=np.float32)
        else:
            values = canonical_pressure_matrix(frame, side)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            cell_values = np.maximum(
                0.0, values[mapper.cell_indices[:, 0], mapper.cell_indices[:, 1]])
        display_values = _heat_display_values(
            cell_values, baseline_ready.get(side, False), threshold)
        corners, centers, normals = mapper.tile_corners_on_mesh(vertices)
        visible = np.einsum(
            "ij,ij->i", normals,
            np.asarray(camera.pos, dtype=np.float32)[None, :] - centers) > 0.0
        projected, depth = camera.project(centers)
        projected_corners, corner_depth = camera.project(corners.reshape(-1, 3))
        projected_corners = projected_corners.reshape(-1, 4, 2)
        corner_depth = corner_depth.reshape(-1, 4)
        valid = (visible
                 & np.isfinite(projected).all(axis=1)
                 & np.isfinite(projected_corners).all(axis=(1, 2))
                 & np.isfinite(corner_depth).all(axis=1)
                 & (corner_depth > 0.0).all(axis=1)
                 & np.isfinite(depth) & (depth > 0.0))
        if not np.any(valid):
            continue
        colors = _heat_colors(
            display_values, baseline_ready.get(side, False))
        order = np.argsort(depth, kind="stable")[::-1]
        for index in order:
            if not valid[index]:
                continue
            polygon = np.rint(projected_corners[index]).astype(np.int32)
            color = tuple(int(value) for value in colors[index])
            cv2.fillConvexPoly(image, polygon, color, cv2.LINE_AA)


def draw_pressure_markers(
        image: np.ndarray,
        camera,
        mesh_slots: list[np.ndarray | None],
        pressure_frames: dict[str, np.ndarray | None],
        mappers: dict[str, PressureHandMapper],
        baseline_ready: dict[str, bool],
        threshold: float = 0.0,
) -> None:
    """Draw fixed geometry pressure markers over already-rendered hand meshes."""

    for slot, side in enumerate(("left", "right")):
        mapper = mappers.get(side)
        mesh = mesh_slots[slot] if slot < len(mesh_slots) else None
        frame = pressure_frames.get(side)
        if mapper is None or mesh is None or frame is None:
            continue
        values = canonical_pressure_matrix(frame, side)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        cell_values = np.maximum(0.0, values[mapper.cell_indices[:, 0],
                                               mapper.cell_indices[:, 1]])
        peak = float(np.max(cell_values, initial=0.0))
        baseline = float(np.percentile(cell_values, 20.0))
        span = max(1.0, peak - baseline)
        contact_floor = baseline + max(60.0, span * 0.16)
        if threshold > 0.0:
            contact_floor = max(contact_floor, float(threshold))
        active = cell_values > contact_floor
        if not np.any(active):
            continue

        bases, normals = mapper.points_on_mesh(mesh)
        view_vector = np.asarray(camera.pos, dtype=np.float32)[None, :] - bases
        visible = np.einsum("ij,ij->i", normals, view_vector) > 0.0
        tops = bases + normals * np.float32(_PROTRUSION_M)
        projected_base, base_depth = camera.project(bases)
        projected_top, top_depth = camera.project(tops)
        valid = (active & visible
                 & np.isfinite(projected_base).all(axis=1)
                 & np.isfinite(projected_top).all(axis=1)
                 & (base_depth > 0.0) & (top_depth > 0.0))
        if not np.any(valid):
            continue

        colors = _pressure_color(cell_values, baseline_ready.get(side, False))
        order = np.argsort(top_depth, kind="stable")[::-1]
        for index in order:
            if not valid[index]:
                continue
            color = tuple(int(value) for value in colors[index])
            stem = tuple(max(0, int(value * 0.62)) for value in color)
            base = tuple(np.rint(projected_base[index]).astype(int))
            top = tuple(np.rint(projected_top[index]).astype(int))
            cv2.line(image, base, top, stem, 2, cv2.LINE_AA)
            cv2.circle(image, top, _MARKER_RADIUS_PX, color, -1, cv2.LINE_AA)


__all__ = [
    "PressureHandMapper",
    "canonical_pressure_matrix",
    "draw_pressure_color_points",
    "draw_pressure_tiles",
    "draw_pressure_markers",
]
