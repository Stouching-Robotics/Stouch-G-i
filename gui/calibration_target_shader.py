"""Depth-tested, view-dependent wireframe shading for the ghost target hand.

This is a small software fragment shader, so the calibration guide does not
need an OpenGL context. Orthographic coordinates use x right, y down, z closer.
"""

from __future__ import annotations

import numpy as np


_FRAGMENT_BATCH_SIZE = 131072
_FRONT_LINE_SIGMA = .5


def _smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0., 1.)
    return t * t * (3. - 2. * t)


class TargetWireframeShader:
    """Reuse mesh topology; shade only its nearest surface at each pixel."""

    def __init__(self, faces, *, color=(125, 188, 245)):
        try:
            faces = np.asarray(faces)
            if not faces.size:
                faces = np.empty((0, 3), dtype=np.intp)
            if (faces.ndim != 2 or faces.shape[1] != 3
                    or not np.isfinite(faces).all() or np.any(faces < 0)
                    or not np.equal(faces, np.floor(faces)).all()):
                raise ValueError
            self._faces = faces.astype(np.intp, copy=True)
            rgb = np.asarray(color, dtype=float)
            if (rgb.shape != (3,) or not np.isfinite(rgb).all()
                    or np.any(rgb < 0) or np.any(rgb > 255)):
                raise ValueError
            self._color = rgb.astype(np.uint8)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('invalid target mesh topology or color') from exc
        self._faces.setflags(write=False)

    @staticmethod
    def _points(value, name):
        try:
            points = np.asarray(value, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'{name} must be finite Nx3 projected vertices') from exc
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f'{name} must be finite Nx3 projected vertices')
        return points

    def _fragments(self, points, width, height, *, wire=True):
        """Vectorized triangle samples, bounded even for screen-sized faces."""
        triangles = points[self._faces]
        xy = triangles[:, :, :2]
        a, b, c = np.moveaxis(xy, 1, 0)
        denominator = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                       - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        low = np.clip(np.ceil(xy.min(axis=1) - .5), [0, 0], [width, height]).astype(np.intp)
        high = np.clip(np.floor(xy.max(axis=1) - .5), [-1, -1], [width - 1, height - 1]).astype(np.intp)
        sizes = np.maximum(high - low + 1, 0)
        usable = (np.abs(denominator) > 1e-10) & np.all(sizes > 0, axis=1)
        face_ids = np.flatnonzero(usable)
        if not len(face_ids):
            return
        a, b, c, denominator = a[usable], b[usable], c[usable], denominator[usable]
        low, sizes = low[usable], sizes[usable]
        # Barycentric gradients also give perpendicular screen-space distance
        # to an edge, without rasterizing or double-drawing shared edge lines.
        gradients = np.stack((np.column_stack((b[:, 1] - c[:, 1], c[:, 0] - b[:, 0])),
                              np.column_stack((c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]))), axis=1)
        gradients /= denominator[:, None, None]
        constants = np.column_stack((b[:, 0] * c[:, 1] - c[:, 0] * b[:, 1],
                                     c[:, 0] * a[:, 1] - a[:, 0] * c[:, 1])) / denominator[:, None]
        if wire:
            all_gradients = np.concatenate((gradients, -gradients.sum(axis=1, keepdims=True)), axis=1)
            altitudes = 1. / np.maximum(np.linalg.norm(all_gradients, axis=2), 1e-15)
        counts = sizes[:, 0] * sizes[:, 1]
        ends = np.cumsum(counts)
        starts = ends - counts
        triangle_z = triangles[usable, :, 2]
        for start in range(0, int(ends[-1]), _FRAGMENT_BATCH_SIZE):
            sequence = np.arange(start, min(start + _FRAGMENT_BATCH_SIZE, int(ends[-1])))
            if ends[-1] <= _FRAGMENT_BATCH_SIZE:
                tri = np.repeat(np.arange(len(counts)), counts)
            else:
                tri = np.searchsorted(ends, sequence, side='right')
            local = sequence - starts[tri]
            x = low[tri, 0] + local % sizes[tri, 0]
            y = low[tri, 1] + local // sizes[tri, 0]
            w0 = gradients[tri, 0, 0] * (x + .5) + gradients[tri, 0, 1] * (y + .5) + constants[tri, 0]
            w1 = gradients[tri, 1, 0] * (x + .5) + gradients[tri, 1, 1] * (y + .5) + constants[tri, 1]
            w2 = 1. - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            selected = tri[inside]
            w0, w1, w2 = w0[inside], w1[inside], w2[inside]
            ids = face_ids[selected]
            depth = (triangle_z[selected, 0] * w0 + triangle_z[selected, 1] * w1
                     + triangle_z[selected, 2] * w2)
            weights = distance = None
            if wire:
                weights = np.column_stack((w0, w1, w2))
                distance = np.maximum(np.minimum(np.minimum(w0 * altitudes[selected, 0],
                    w1 * altitudes[selected, 1]), w2 * altitudes[selected, 2]), 0.)
            yield y[inside] * width + x[inside], depth, ids, weights, distance

    def _normal_facing(self, points):
        # Undo the screen-y reflection before calculating outward normals.
        camera = points.copy()
        camera[:, 1] *= -1.
        triangles = camera[self._faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        vertex_normals = np.zeros_like(camera)
        for corner in range(3):
            np.add.at(vertex_normals, self._faces[:, corner], normals)
        return vertex_normals[:, 2] / np.maximum(np.linalg.norm(vertex_normals, axis=1), 1e-15)

    def render(self, target_px, live_px, width, height):
        """Return straight RGBA; nearer/equal live surfaces hide target pixels.

        Hidden target faces never contribute rear wires through a front face.
        The visible wire's opacity and Gaussian edge feather respond smoothly
        to camera-space depth and outward normal, with no post-blur depth leak.
        """
        if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0
               for n in (width, height)):
            raise ValueError('width and height must be positive integers')
        target = self._points(target_px, 'target')
        live = None if live_px is None else self._points(live_px, 'live')
        if live is not None and live.shape != target.shape:
            raise ValueError('live and target must use the same mesh topology')
        if self._faces.size and self._faces.max() >= len(target):
            raise ValueError('target face index is outside the vertex array')
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        if not self._faces.size:
            return rgba
        target_depth = np.full(width * height, -np.inf)
        live_depth = None
        if live is not None:
            live_depth = np.full(width * height, -np.inf)
            for pixels, depth, _ids, _weights, _distance in self._fragments(live, width, height, wire=False):
                np.maximum.at(live_depth, pixels, depth)
        facing = self._normal_facing(target)
        low, high = target[:, 2].min(), target[:, 2].max()
        flat_rgba = rgba.reshape(-1, 4)
        for pixels, depth, ids, weights, distance in self._fragments(target, width, height):
            np.maximum.at(target_depth, pixels, depth)
            nearest = depth >= target_depth[pixels] - 1e-10
            pixels, depth, ids = pixels[nearest], depth[nearest], ids[nearest]
            weights, distance = weights[nearest], distance[nearest]
            normal_z = np.sum(facing[self._faces[ids]] * weights, axis=1)
            front = .25 + .75 * _smoothstep(-.2, .8, normal_z)
            depth_weight = _smoothstep(low, high, depth) if high - low > 1e-9 else np.ones_like(depth)
            visibility = front * (.2 + .8 * depth_weight)
            # Analytic soft lines stay on the depth-tested surface. Rearward
            # lines are wider/softer and fainter, rather than blurred through
            # nearer fingers after compositing.
            sigma = _FRONT_LINE_SIGMA + .85 * (1. - visibility)
            # Normalize the feather as it widens: blur must not make a
            # mid-depth line brighter than the sharp foreground line.
            wire = (np.exp(-.5 * (distance / sigma) ** 2) * (_FRONT_LINE_SIGMA / sigma)
                    * (18. + 158. * visibility) / 255.)
            fill = 6. + 26. * visibility
            alpha = np.clip(fill + (255. - fill) * wire, 0., 255.).astype(np.uint8)
            flat_rgba[pixels, :3] = self._color
            flat_rgba[pixels, 3] = alpha
        if live_depth is not None:
            span = max(float(np.ptp(target[:, 2])), float(np.ptp(live[:, 2])), 1.)
            # Live wins a coplanar tie, avoiding wire flicker once aligned.
            flat_rgba[live_depth >= target_depth - span * 1e-5] = 0
        return rgba
