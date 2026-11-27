"""Independent rigid-hand orientation guidance in the neutral Friston frame."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QMenu, QWidget
from common.i18n import L
from common.calibration_root import processed_world_up as _processed_world_up, solve_world_vertical_pitch_frame
from gui.live_3d import align_left_to_right_reference
from gui.calibration_arrow_geometry import AXIS_LENGTH_RADIUS, rotation_arrow_meshes
from gui.calibration_target_shader import TargetWireframeShader
from gui.calibration_render_backend import CalibrationRenderBackend

_POSES = ('horizontal', 'vertical_up', 'vertical_down', 'roll', 'yaw', 'calc')
_COLORS = (QColor(242, 79, 82), QColor(91, 210, 112), QColor(89, 151, 255))
_DEFAULT_YAW, _DEFAULT_PITCH = -45., 35.264389682754654
# The wrist guide uses -X for the level fingertips and +Y for dorsal/up.
_GUIDE_FROM_FRISTON = R.from_euler("y", -90, degrees=True).as_matrix()


def _target(side, name):
    specs = {'vertical_up': ('z', -90), 'vertical_down': ('z', 90),
             'roll': ('x', 90 if side == 'left' else -90),
             'yaw': ('y', 90 if side == 'right' else -90)}
    axis, angle = specs.get(name, ('x', 0))
    return R.from_euler(axis, angle, degrees=True)


def _single_rotation(value):
    try:
        if not isinstance(value, R) or value.as_quat().shape != (4,):
            return None
        return value if np.isfinite(value.as_quat()).all() else None
    except (ValueError, TypeError):
        return None


def _valid_matrix(value):
    try:
        a = np.asarray(value, dtype=float)
        if (a.shape == (3, 3) and np.isfinite(a).all()
                and np.allclose(a.T @ a, np.eye(3), atol=1e-5)
                and abs(np.linalg.det(a) - 1) < 1e-5):
            return a
    except (TypeError, ValueError):
        pass
    return None


def _draw_arrowhead(painter, start, end, size=9):
    direction = np.asarray(end) - np.asarray(start)
    norm = np.linalg.norm(direction)
    if norm <= 1e-6:
        return
    direction = direction / norm
    normal = np.array([-direction[1], direction[0]])
    for tip in (end - direction * size + normal * size * .45,
                end - direction * size - normal * size * .45):
        painter.drawLine(QPointF(*end), QPointF(*tip))


def build_orientation_state(side: str, pose_name: str, base: R | None,
                            current: R | None, captured: dict[str, R], *,
                            world_from_model: np.ndarray | None = None) -> dict:
    """Signed correction is log(live^-1 * target), in current local axes.

    V uses historical captures, optionally aided by the world vertical convention.
    """
    side_ok = side in ('left', 'right') and pose_name in _POSES
    target = _target(side, pose_name).as_matrix() if side_ok else np.eye(3)
    result = dict(live_rotation=None, target_rotation=target, correction_deg=None,
                  reference_ready=False, reference_rank=0, reference_poses=[], reference_source=None,
                  message=L('参考轴建立中', 'Establishing reference axes'))
    if not side_ok:
        result['message'] = L('无效的手侧或姿态', 'Invalid side or pose')
        return result
    base, current = _single_rotation(base), _single_rotation(current)
    if base is None:
        result['message'] = L('缺少有效水平参考', 'Missing valid horizontal reference')
        return result
    v = _valid_matrix(world_from_model) if world_from_model is not None else None
    if world_from_model is not None and v is None:
        result['message'] = L('参考矩阵无效', 'Invalid reference matrix')
        return result
    if v is not None:
        result['reference_source'] = 'provided'
    if v is None:
        model_axes, world_axes, used = [], [], []
        for name in ('vertical_up', 'vertical_down', 'roll', 'yaw'):
            # A current step may have a previous capture: only the caller's
            # explicitly supplied historical captures are considered.
            pose = _single_rotation(captured.get(name)) if isinstance(captured, dict) else None
            if pose is None:
                continue
            a = (pose * base.inv()).as_rotvec()
            length = np.linalg.norm(a)
            if not np.radians(20) <= length <= np.radians(160):
                continue
            b = _target(side, name).as_rotvec()
            model_axes.append(b / np.linalg.norm(b))
            world_axes.append(a / length)
            used.append(name)
        result['reference_poses'] = used
        if model_axes:
            a, b = np.asarray(world_axes).T, np.asarray(model_axes).T
            measured_rank = np.linalg.matrix_rank(a, tol=0.1)
            model_rank = np.linalg.matrix_rank(b, tol=0.1)
            rank = min(measured_rank, model_rank)
            result['reference_rank'] = int(rank)
            if rank >= 2:
                u, _, vt = np.linalg.svd(a @ b.T)
                candidate = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
                residual = np.degrees(np.arccos(np.clip(np.sum(a * (candidate @ b), axis=0), -1, 1)))
                result['reference_residual_deg'] = float(np.max(residual))
                if np.max(residual) <= 30:
                    v = candidate
                    result['reference_source'] = 'captured-axes'
                else:
                    result['message'] = L('参考轴不一致，请重新采集', 'Inconsistent reference axes; recapture')
            elif rank == 1 and model_rank == 1:
                # Incoming rotations have already passed the receiver's axis
                # conjugation; physical world +Z is processed world +Y today.
                # Derive that direction from the receiver, rather than from
                # the legacy solver fallback. Never fit to the live pose.
                pitches = [world_axes[i] * (-1 if name == 'vertical_up' else 1)
                           for i, name in enumerate(used)
                           if name in ('vertical_up', 'vertical_down')]
                if pitches:
                    mean = np.sum(pitches, axis=0)
                    norm = np.linalg.norm(mean)
                    if norm > 1e-6:
                        z = mean / norm
                        consistency = np.degrees(np.arccos(np.clip(np.asarray(pitches) @ z, -1, 1)))
                        try:
                            candidate, _ = solve_world_vertical_pitch_frame(z)
                            residual = np.degrees(np.arccos(np.clip(np.asarray(pitches) @ candidate[:, 2], -1, 1)))
                            result['reference_residual_deg'] = float(np.max(residual))
                            if np.max(consistency) <= 30 and np.max(residual) <= 30:
                                v = candidate
                                result['reference_source'] = 'gravity-and-captured-pitch'
                        except ValueError:
                            pass
                        if v is None:
                            result['message'] = L('竖直参考与采集轴不一致，请重新采集', 'Vertical reference conflicts with captured axis; recapture')
                    else:
                        result['message'] = L('采集轴不一致，请重新采集', 'Inconsistent captured axes; recapture')
            elif rank < 2:
                result['message'] = L('采集轴退化，请重新采集', 'Degenerate captured axes; recapture')
        if v is None:
            return result
    result['reference_ready'] = True
    if current is None:
        result['message'] = L('缺少实时手腕数据', 'Missing live wrist data')
        return result
    relative = (current * base.inv()).as_matrix()
    live = v.T @ relative @ v
    result['live_rotation'] = live
    result['correction_deg'] = np.degrees(R.from_matrix(live.T @ target).as_rotvec())
    result['message'] = L('按箭头旋转至目标', 'Rotate toward the target following the arrows')
    return result


class CalibrationOrientationGuide(QWidget):
    """Orbitable rigid mesh: automatic GPU surfaces with CPU compatibility."""
    def __init__(self, preview, parent=None, *, render_mode=None):
        super().__init__(parent)
        self._preview = preview
        self._state = None
        self._side = 'right'
        self._yaw, self._pitch = _DEFAULT_YAW, _DEFAULT_PITCH
        self._last_mouse = None
        self._left_baseline = None
        self._target_shaders = {}
        self._render_backend = CalibrationRenderBackend(render_mode)
        self.destroyed.connect(self._render_backend.close)
        self.setMinimumSize(250, 200)

    def set_state(self, side: str, state: dict | None) -> None:
        self._side, self._state = side, state
        self.setVisible(state is not None)
        self.update()

    @property
    def render_backend_info(self):
        return self._render_backend.info

    def set_render_mode(self, mode):
        self._render_backend.set_mode(mode)
        self.update()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        choices = {}
        for mode, label in (('auto', L('自动（GPU 优先 / 重试）', 'Auto (prefer GPU / retry)')),
                            ('software', L('兼容模式（CPU）', 'Compatibility (CPU)'))):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(mode == self._render_backend.mode)
            choices[action] = mode
        chosen = menu.exec(event.globalPos())
        if chosen in choices:
            self.set_render_mode(choices[chosen])

    def _camera(self):
        return R.from_euler('yx', [self._yaw, self._pitch], degrees=True).as_matrix()

    def _display_neutral(self, side, points):
        """Wrist-centred display geometry in the shared root reference."""
        wrist = np.asarray(self._preview._neutral[side][0], dtype=float)
        centred = np.asarray(points, dtype=float) - wrist
        if side == 'left':
            if self._left_baseline is None:
                self._left_baseline = align_left_to_right_reference(
                    self._preview._neutral['left'], self._preview._neutral['right']).as_matrix()
            centred = centred @ self._left_baseline.T
        return centred @ _GUIDE_FROM_FRISTON.T

    def _project(self, points, width, height, scale, pan=(0., 0.)):
        p = np.asarray(points) @ self._camera().T
        return np.column_stack((width * .5 + pan[0] + p[:, 0] * scale,
                                height * .53 + pan[1] - p[:, 1] * scale, p[:, 2] * scale))

    @staticmethod
    def _rotation_arrows(radius, live, target):
        if live is None or target is None:
            return []
        return rotation_arrow_meshes(live, target, radius)

    def _framing(self, mesh, target, live, width, height, *, arrow_meshes=None):
        """One camera fit for both hands, all axis endpoints and correction arcs."""
        radius = max(float(np.max(np.linalg.norm(mesh, axis=1))), 1e-6)
        axis_len = radius * AXIS_LENGTH_RADIUS
        origin_and_axes = np.vstack((np.zeros(3), np.eye(3) * axis_len))
        items = []
        for rotation in (target, live):
            if rotation is not None:
                items.extend((mesh @ rotation.T, origin_and_axes @ rotation.T))
        if arrow_meshes is None:
            arrow_meshes = self._rotation_arrows(radius, live, target)
        items.extend(vertices for _axis, vertices, _faces in arrow_meshes)
        view = np.concatenate(items) @ self._camera().T
        low, high = view[:, :2].min(axis=0), view[:, :2].max(axis=0)
        span = np.maximum(high - low, 1e-8)
        # Reserve two header and two footer lines plus space for axis labels,
        # arrowheads, and the soft edge of the ghost mesh.
        left, right = 18., max(19., width - 18.)
        top, bottom = 48., max(49., height - 48.)
        scale = min((right - left) / span[0], (bottom - top) / span[1])
        middle = (low + high) * .5
        pan = ((left + right) * .5 - width * .5 - middle[0] * scale,
               (top + bottom) * .5 - height * .53 + middle[1] * scale)
        return scale, pan, (left, top, right, bottom)

    def _paint_rotation_arrows(self, painter, meshes, width, height, scale, pan):
        prepared = []
        for axis, vertices, faces in sorted(meshes, key=lambda mesh: mesh[0]):
            triangles = self._project(vertices, width, height, scale, pan)[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            shading = .57 + .43 * np.abs(normals @ [.25, -.45, .857]) / np.maximum(
                np.linalg.norm(normals, axis=1), 1e-12)
            order = np.argsort(triangles[:, :, 2].mean(axis=1), kind='stable')
            head = np.any(faces >= len(vertices) - 5, axis=1)
            prepared.append((axis, triangles, shading, order, head))
        # Orientation gizmos stay visible over the hand. At large turns two
        # trajectories can share a circle; stable axis layers avoid alternating
        # coincident facets. Draw every head last so another shaft cannot hide
        # its target. Each mesh retains its camera-space shading/self-depth.
        for heads in (False, True):
            for axis, triangles, shading, order, head in prepared:
                rgb = np.array(_COLORS[axis].getRgb()[:3])
                for index in order[head[order] == heads]:
                    shaded = rgb * shading[index]
                    color = QColor(int(shaded[0]), int(shaded[1]), int(shaded[2]))
                    painter.setPen(QPen(color, .5))
                    painter.setBrush(color)
                    painter.drawPolygon(QPolygonF([QPointF(float(x), float(y))
                                                  for x, y in triangles[index, :, :2]]))

    @staticmethod
    def _text_line(painter, text, x, y, width):
        """Fit each localized line inside a narrow bimanual panel."""
        painter.save()
        font = painter.font()
        font.setPixelSize(13)
        while font.pixelSize() > 8 and QFontMetrics(font).horizontalAdvance(text) > width:
            font.setPixelSize(font.pixelSize() - 1)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(1, int(width)))
        painter.drawText(QRectF(x, y, max(1, width), 19),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse = event.position()

    def mouseMoveEvent(self, event):
        if self._last_mouse is not None:
            delta = event.position() - self._last_mouse
            self._yaw += delta.x() * .5
            self._pitch = float(np.clip(self._pitch + delta.y() * .5, -89, 89))
            self._last_mouse = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self._last_mouse = None

    def mouseDoubleClickEvent(self, event):
        self._yaw, self._pitch = _DEFAULT_YAW, _DEFAULT_PITCH
        self._last_mouse = None
        self.update()

    def _paint_software_surfaces(self, painter, target_points, live_points, faces, width, height):
        if live_points is not None:
            triangles = live_points[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            norms = np.linalg.norm(normals, axis=1)
            shading = .57 + .43 * np.abs(normals @ [.25, -.45, .857]) / np.maximum(norms, 1e-12)
            for index in np.argsort(triangles[:, :, 2].mean(axis=1)):
                tri = triangles[index]
                painter.setPen(Qt.PenStyle.NoPen)
                factor = shading[index]
                painter.setBrush(QColor(int(223 * factor), int(185 * factor), int(150 * factor), 225))
                painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in tri[:, :2]]))
        source_faces = self._preview._mesh_faces[self._side]
        cached = self._target_shaders.get(self._side)
        if cached is None or cached[0] is not source_faces:
            cached = (source_faces, TargetWireframeShader(faces))
            self._target_shaders[self._side] = cached
        rgba = cached[1].render(target_points, live_points, int(width), int(height))
        layer = QImage(rgba.data, int(width), int(height), rgba.strides[0], QImage.Format.Format_RGBA8888)
        painter.drawImage(0, 0, layer)

    def _paint(self, painter, width, height):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(0, 0, width, height, QColor(24, 31, 45))
        state = self._state
        if state is None or self._side not in ('left', 'right'):
            return
        try:
            mesh = self._display_neutral(self._side, self._preview._neutral_mesh[self._side])
            faces = np.asarray(self._preview._mesh_faces[self._side], dtype=int)
            if mesh.ndim != 2 or mesh.shape[1] != 3 or not np.isfinite(mesh).all():
                return
            radius = max(float(np.max(np.linalg.norm(mesh, axis=1))), 1e-6)
            target = _valid_matrix(state.get('target_rotation'))
            live = _valid_matrix(state.get('live_rotation'))
            if target is None:
                return
            arrow_meshes = (self._rotation_arrows(radius, live, target)
                            if state.get('reference_ready') else [])
            scale, pan, _ = self._framing(mesh, target, live, width, height,
                                        arrow_meshes=arrow_meshes)
            target_points = self._project(mesh @ target.T, width, height, scale, pan)
            live_points = (self._project(mesh @ live.T, width, height, scale, pan)
                           if live is not None else None)
            layer = self._render_backend.render(target_points, live_points, faces, int(width), int(height))
            if layer is None:
                # Initialization/render failure falls back within this frame;
                # pose, reference, capture gates and gizmos are untouched.
                self._paint_software_surfaces(painter, target_points, live_points, faces, width, height)
            else:
                painter.drawImage(0, 0, layer)
            axis_len = radius * AXIS_LENGTH_RADIUS
            for rotation, dashed in ((target, True), (live, False)):
                if rotation is None:
                    continue
                for i, color in enumerate(_COLORS):
                    points = self._project(np.array([np.zeros(3), rotation[:, i] * axis_len]), width, height, scale, pan)
                    pen = QPen(color, 2 if dashed else 3)
                    if dashed:
                        pen.setStyle(Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(*points[0, :2]), QPointF(*points[1, :2]))
                    _draw_arrowhead(painter, points[0, :2], points[1, :2], 8)
                    painter.drawText(QPointF(*points[1, :2]), 'XYZ'[i])
            self._paint_rotation_arrows(painter, arrow_meshes, width, height, scale, pan)
        except (ValueError, TypeError, IndexError, KeyError, AttributeError):
            painter.setPen(QColor('white'))
            painter.drawText(10, 25, L('手模型数据不可用', 'Hand model unavailable'))
            return
        painter.setPen(QColor(225, 230, 238))
        side = L('左手', 'Left') if self._side == 'left' else L('右手', 'Right')
        self._text_line(painter, side + L('  虚 = 目标  实 = 当前', '  Dashed = target  Solid = current'), 8, 1, width - 60)
        gpu = self._render_backend.active == 'opengl'
        painter.setPen(QColor(107, 211, 225) if gpu else QColor(202, 190, 150))
        self._text_line(painter, 'GPU' if gpu else 'CPU', width - 42, 1, 34)
        painter.setPen(QColor(225, 230, 238))
        self._text_line(painter, L('拖动旋转 / 双击复位 / 右键渲染', 'Drag / double-click reset / right-click renderer'), 8, 21, width - 16)
        info = self._render_backend.info
        detail = info.get('renderer', '') if gpu else L('软件兼容模式', 'Software compatibility')
        self.setToolTip(detail + '\n' + str(info.get('reason', ''))[:240] + '\n' +
                        L('右键可切换自动或兼容模式', 'Right-click to select auto or compatibility mode'))
        self._text_line(painter, str(state.get('message', '')), 8, height - 42, width - 16)
        correction = state.get('correction_deg')
        if correction is not None:
            try:
                c = np.asarray(correction, dtype=float)
                if c.shape == (3,) and np.isfinite(c).all():
                    self._text_line(painter, L('当前轴 XYZ  ', 'Current XYZ  ') + ' '.join(f'{x:+.1f}°' for x in c), 8, height - 22, width - 16)
            except (TypeError, ValueError):
                pass

    def paintEvent(self, event):
        painter = QPainter(self)
        self._paint(painter, self.width(), self.height())
        painter.end()

    def render_bgr(self, width=660, height=360):
        image = QImage(int(width), int(height), QImage.Format.Format_RGB888)
        painter = QPainter(image)
        self._paint(painter, int(width), int(height))
        painter.end()
        data = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(height, image.bytesPerLine())
        return data[:, :width * 3].reshape(height, width, 3)[:, :, ::-1].copy()
