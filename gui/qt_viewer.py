"""Qt (PySide6) display/input layer, replacing the OpenCV HighGUI window of Live3DViewer.

Design notes (composed, to avoid name clashes):
- ``QtCanvas`` and ``_CanvasArea`` are plain QWidgets; they do **not** inherit
  Live3DViewer -- Live3DViewer already has ``update()``/``close()``, and
  inheriting QWidget directly would clash with QWidget.update()/close().
- Live3DViewer only produces the numpy canvas (_draw outputs 1280x720 uint8
  BGR); QtCanvas displays it via ``QImage(Format_BGR888)`` (zero-copy) and
  forwards Qt mouse/keyboard events to controller._on_mouse/_key_command after
  synthesizing them **with cv2 constant names**, so subclasses like BimanualViewer
  can override _on_mouse/_key_command unchanged.
- The event pump runs via ``pump_events()`` (QApplication.processEvents) called
  from Live3DViewer.tick(); it never enters the QApplication.exec() event loop.
  The QApplication is only created when a viewer is constructed, not at import
  time, keeping headless imports (replay/calibration preview) safe.
"""

from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout, QWidget)

# Default tactile threshold applied at startup: cells below this ADC value are
# zeroed so tiny baseline / sensor fluctuations do not flicker the cell numbers.
DEFAULT_TACTILE_THRESHOLD = 60


def ensure_qapp() -> QApplication:
    """Return the global QApplication, creating one if absent.  Only called when a viewer is constructed."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def pump_events() -> None:
    """Pump Qt events once (equivalent to the old cv2.waitKey event handling)."""
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


class _CanvasArea(QWidget):
    """Paint + interaction surface: shows the canvas aspect-fitted and forwards mouse/keyboard events to the controller.

    The controller must provide (same interface as Live3DViewer):
      - ``_on_mouse(event, x, y, flags, param)``  -- uses cv2 constants
      - ``_key_command(key)``
      - ``exit_requested`` attribute
    """

    def __init__(self, parent: QWidget | None, controller, canvas_w: int,
                 canvas_h: int):
        super().__init__(parent)
        self._controller = controller
        self._cw = int(canvas_w)
        self._ch = int(canvas_h)
        self._image: QImage | None = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._view_control = _ViewControl(self, controller)
        self._place_view_control()
        self._baseline_overlay: QLabel | None = None
        self._setup_baseline_overlay()

    def _setup_baseline_overlay(self) -> None:
        """Large centred banner used during the baseline settle delay."""
        self._baseline_overlay = QLabel(self)
        self._baseline_overlay.setAlignment(Qt.AlignCenter)
        self._baseline_overlay.setFocusPolicy(Qt.NoFocus)
        self._baseline_overlay.setAttribute(
            Qt.WA_TransparentForMouseEvents, True)
        self._baseline_overlay.setWordWrap(True)
        self._baseline_overlay.setStyleSheet(
            "QLabel {"
            "  color: #ffd54f;"
            "  background-color: rgba(10, 16, 28, 210);"
            "  border: 2px solid #ffd54f;"
            "  border-radius: 12px;"
            "  font-size: 26px;"
            "  font-weight: bold;"
            "  padding: 8px 20px;"
            "}"
        )
        self._baseline_overlay.hide()
        self._place_baseline_overlay()

    def _place_baseline_overlay(self) -> None:
        if self._baseline_overlay is None:
            return
        # Centre-top of the canvas, leaving room for the top chrome.
        w = int(self.width() * 0.82)
        h = max(70, int(self.height() * 0.13))
        x = (self.width() - w) // 2
        y = int(self.height() * 0.10)
        self._baseline_overlay.setGeometry(x, y, w, h)

    def show_baseline_overlay(self, text: str) -> None:
        if self._baseline_overlay is None:
            return
        if self._baseline_overlay.text() != text:
            self._baseline_overlay.setText(text)
        self._place_baseline_overlay()
        self._baseline_overlay.show()
        self._baseline_overlay.raise_()

    def hide_baseline_overlay(self) -> None:
        if self._baseline_overlay is not None:
            self._baseline_overlay.hide()

    def _place_view_control(self) -> None:
        """Place the compact camera control between record and tactile cards."""
        margin = 18
        preferred_y = 70
        max_y = max(margin, self.height() - self._view_control.height() - margin)
        self._view_control.move(
            max(margin, self.width() - self._view_control.width() - margin),
            min(preferred_y, max_y))
        self._view_control.raise_()

    # ---- Canvas display ----------------------------------------------------
    def set_image(self, qimage: QImage | None) -> None:
        self._image = qimage
        self.update()

    def fitted_rect(self) -> QRect:
        """Fit the canvas into the widget rect at its aspect ratio (letterbox)."""
        r = self.rect()
        if self._image is None or self._image.width() <= 0:
            return QRect(r)
        iw, ih = self._image.width(), self._image.height()
        scale = min(r.width() / iw, r.height() / ih)
        nw = max(1, int(round(iw * scale)))
        nh = max(1, int(round(ih * scale)))
        x = r.x() + (r.width() - nw) // 2
        y = r.y() + (r.height() - nh) // 2
        return QRect(x, y, nw, nh)

    def _to_canvas(self, pos) -> tuple[int, int]:
        """Widget -> canvas coords (inverse-mapped via fitted_rect, correct hit-testing when scaled)."""
        r = self.fitted_rect()
        if r.width() <= 0 or r.height() <= 0:
            return 0, 0
        x = (pos.x() - r.x()) * self._cw / r.width()
        y = (pos.y() - r.y()) * self._ch / r.height()
        return int(np.clip(x, 0, self._cw - 1)), int(np.clip(y, 0, self._ch - 1))

    # ---- Qt event -> cv2 event synthesis ------------------------------------
    def paintEvent(self, ev) -> None:  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 24, 34))
        if self._image is not None:
            p.drawImage(self.fitted_rect(), self._image)
        p.end()

    def resizeEvent(self, ev) -> None:  # noqa: N802
        self._place_view_control()
        self._place_baseline_overlay()
        self.update()

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        x, y = self._to_canvas(ev.position().toPoint())
        if ev.button() == Qt.LeftButton:
            self._controller._on_mouse(
                cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        elif ev.button() == Qt.MiddleButton:
            self._controller._on_mouse(
                cv2.EVENT_MBUTTONDOWN, x, y, 0, None)
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        x, y = self._to_canvas(ev.position().toPoint())
        if ev.button() == Qt.LeftButton:
            self._controller._on_mouse(
                cv2.EVENT_LBUTTONUP, x, y, 0, None)
        elif ev.button() == Qt.MiddleButton:
            # Note: cv2.EVENT_MBUTTONUP == 6 (not 5); must use the symbolic name.
            self._controller._on_mouse(
                cv2.EVENT_MBUTTONUP, x, y, 0, None)
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        x, y = self._to_canvas(ev.position().toPoint())
        buttons = ev.buttons()
        if buttons & Qt.LeftButton:
            flags = cv2.EVENT_FLAG_LBUTTON
        elif buttons & Qt.MiddleButton:
            flags = cv2.EVENT_FLAG_MBUTTON
        else:
            flags = 0
        self._controller._on_mouse(cv2.EVENT_MOUSEMOVE, x, y, flags, None)
        ev.accept()

    def wheelEvent(self, ev) -> None:  # noqa: N802
        delta = ev.angleDelta().y()
        if delta == 0:
            ev.accept()
            return
        # As in cv2: the wheel delta is packed into the high 16 bits of flags (±120).
        flags = (delta & 0xFFFF) << 16
        self._controller._on_mouse(cv2.EVENT_MOUSEWHEEL, 0, 0, flags, None)
        ev.accept()

    def keyPressEvent(self, ev) -> None:  # noqa: N802
        ctrl = self._controller
        if ev.key() == Qt.Key_Escape:
            ctrl.exit_requested = True
            ev.accept()
            return
        txt = ev.text()
        if txt:
            code = ord(txt.lower()[0])
            if code == ord("q"):
                ctrl.exit_requested = True
            else:
                ctrl._key_command(code)
        ev.accept()


class _ViewControl(QWidget):
    """On-canvas orbit pad and circular roll slider.

    The four direction buttons orbit while pressed, the centre restores the
    saved defaults, and dragging anywhere on the outer ring adjusts camera
    roll.  It deliberately talks to the viewer through tiny duck-typed
    callbacks so the same control works for single- and bimanual viewers.
    """

    WIDTH, HEIGHT = 160, 210
    CX, CY = 80.0, 70.0
    OUTER_R, INNER_R = 62.0, 44.0

    def __init__(self, parent: QWidget, controller):
        super().__init__(parent)
        self._controller = controller
        self._pressed: str | None = None
        self._hovered: str | None = None
        self._rolling = False
        self._last_pointer_angle = 0.0
        self._repeat = QTimer(self)
        self._repeat.setInterval(85)
        self._repeat.timeout.connect(self._repeat_direction)
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self.setFocusPolicy(Qt.NoFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.setToolTip(
            "方向键：俯仰/偏航（按住连续旋转）\n"
            "拖动外圈：滚转\n中心：恢复窗口刚打开时的状态\n"
            "底部按钮：切换旋转灵敏度")

    @staticmethod
    def _normalized_delta(current: float, previous: float) -> float:
        return (current - previous + 180.0) % 360.0 - 180.0

    def _pointer_angle(self, pos) -> float:
        return float(np.degrees(np.arctan2(
            pos.x() - self.CX, -(pos.y() - self.CY))))

    def _radius(self, pos) -> float:
        return float(np.hypot(pos.x() - self.CX, pos.y() - self.CY))

    @staticmethod
    def _button_rects() -> dict[str, QRectF]:
        # Small, separated buttons leave breathing room inside the ring and
        # make each direction visually distinct from the centre reset action.
        return {
            "up": QRectF(66, 27, 28, 24),
            "left": QRectF(32, 58, 28, 24),
            "reset": QRectF(66, 58, 28, 24),
            "right": QRectF(100, 58, 28, 24),
            "down": QRectF(66, 89, 28, 24),
        }

    def _hit(self, pos) -> str | None:
        if QRectF(20, 176, 120, 24).contains(QPointF(pos)):
            return "sensitivity"
        radius = self._radius(pos)
        if self.INNER_R <= radius <= self.OUTER_R + 5.0:
            return "roll"
        point = QPointF(pos)
        for name, rect in self._button_rects().items():
            if rect.contains(point):
                return name
        return None

    def _repeat_direction(self) -> None:
        if self._pressed in {"up", "down", "left", "right"}:
            callback = getattr(self._controller, "_view_control_step", None)
            if callback is not None:
                callback(self._pressed)
            self.update()

    def _roll_value(self) -> float:
        return self._view_values()[2] % 360.0

    def _view_values(self) -> tuple[float, float, float]:
        callback = getattr(self._controller, "_view_control_angles", None)
        if callback is not None:
            return tuple(float(value) for value in callback())
        return 0.0, 0.0, 0.0

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        hit = self._hit(ev.position())
        self._pressed = hit
        if hit == "roll":
            self._rolling = True
            self._last_pointer_angle = self._pointer_angle(ev.position())
        elif hit == "reset":
            callback = getattr(self._controller, "_view_control_reset", None)
            if callback is not None:
                callback()
        elif hit == "sensitivity":
            callback = getattr(
                self._controller, "_view_control_cycle_sensitivity", None)
            if callback is not None:
                callback()
        elif hit is not None:
            self._repeat_direction()
            self._repeat.start()
        self.update()
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._rolling and (ev.buttons() & Qt.LeftButton):
            angle = self._pointer_angle(ev.position())
            delta = self._normalized_delta(angle, self._last_pointer_angle)
            self._last_pointer_angle = angle
            callback = getattr(self._controller, "_view_control_roll", None)
            if callback is not None and np.isfinite(delta):
                callback(delta)
            self.update()
        else:
            hovered = self._hit(ev.position())
            if hovered != self._hovered:
                self._hovered = hovered
                self.setCursor(
                    Qt.PointingHandCursor if hovered is not None
                    else Qt.ArrowCursor)
                self.update()
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        self._repeat.stop()
        self._pressed = None
        self._rolling = False
        self.update()
        # Return keyboard shortcuts to the 3D surface after using the pad.
        if self.parentWidget() is not None:
            self.parentWidget().setFocus()
        ev.accept()

    def leaveEvent(self, ev) -> None:  # noqa: N802
        if not self._rolling:
            self._hovered = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
        super().leaveEvent(ev)

    @staticmethod
    def _draw_chevron(painter: QPainter, centre: QPointF, direction: str,
                        color: QColor) -> None:
        points = {
            "up": [(-5, 3), (0, -2), (5, 3)],
            "down": [(-5, -3), (0, 2), (5, -3)],
            "left": [(3, -5), (-2, 0), (3, 5)],
            "right": [(-3, -5), (2, 0), (-3, 5)],
        }[direction]
        painter.setPen(QPen(color, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPolyline(QPolygonF([
            QPointF(centre.x() + x, centre.y() + y) for x, y in points]))

    def paintEvent(self, ev) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Floating translucent panel, kept readable over both bright and dark scenes.
        painter.setPen(QPen(QColor(74, 96, 128, 185), 1.0))
        painter.setBrush(QColor(13, 21, 35, 218))
        painter.drawRoundedRect(QRectF(1, 1, self.WIDTH - 2, self.HEIGHT - 2), 18, 18)

        centre = QPointF(self.CX, self.CY)
        painter.setPen(QPen(QColor(91, 111, 143, 185), 1.0))
        painter.setBrush(QColor(35, 47, 66, 225))
        painter.drawEllipse(centre, self.OUTER_R, self.OUTER_R)
        painter.setBrush(QColor(18, 28, 44, 245))
        painter.drawEllipse(centre, self.INNER_R, self.INNER_R)

        # Ring ticks communicate that this is draggable rather than decorative.
        for index in range(24):
            angle = np.deg2rad(index * 15.0)
            long_tick = index % 3 == 0
            r0 = self.INNER_R + (4.0 if long_tick else 7.0)
            r1 = self.OUTER_R - 5.0
            color = QColor(143, 159, 184, 205 if long_tick else 140)
            painter.setPen(QPen(color, 2.2 if long_tick else 1.2,
                                Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(
                QPointF(self.CX + np.sin(angle) * r0,
                        self.CY - np.cos(angle) * r0),
                QPointF(self.CX + np.sin(angle) * r1,
                        self.CY - np.cos(angle) * r1))

        # Current roll handle.
        roll = np.deg2rad(self._roll_value())
        handle_r = (self.INNER_R + self.OUTER_R) / 2.0
        handle = QPointF(self.CX + np.sin(roll) * handle_r,
                         self.CY - np.cos(roll) * handle_r)
        painter.setPen(QPen(QColor(171, 160, 255), 2.0))
        painter.setBrush(QColor(111, 94, 220))
        painter.drawEllipse(handle, 6.5, 6.5)

        # Direction and reset buttons.
        for name, rect in self._button_rects().items():
            active = name == self._pressed
            hovered = name == self._hovered
            fill = (QColor(66, 83, 112, 245) if active else
                    QColor(47, 62, 85, 242) if hovered else
                    QColor(30, 42, 60, 242))
            painter.setPen(QPen(QColor(90, 110, 141, 220), 1.0))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 8, 8)
            if name in {"up", "down", "left", "right"}:
                self._draw_chevron(painter, rect.center(), name,
                                     QColor(226, 234, 246))
            else:
                painter.setPen(QPen(QColor(226, 234, 246), 2.0,
                                    Qt.DashLine, Qt.RoundCap))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(rect.center(), 6.0, 6.0)

        yaw, pitch, roll_value = self._view_values()
        labels = (("Yaw", yaw), ("Pitch", pitch), ("Roll", roll_value))
        label_font = QFont(self.font())
        label_font.setPixelSize(9)
        painter.setFont(label_font)
        for index, (label, value) in enumerate(labels):
            rect = QRectF(3 + index * 52, 133, 50, 34)
            painter.setPen(QColor(142, 158, 181))
            painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, label)
            value_font = QFont(label_font)
            value_font.setPixelSize(12)
            value_font.setBold(True)
            painter.setFont(value_font)
            painter.setPen(QColor(235, 240, 248))
            painter.drawText(rect.adjusted(0, 16, 0, 0),
                             Qt.AlignHCenter | Qt.AlignTop,
                             f"{value:0.0f}\N{DEGREE SIGN}")
            painter.setFont(label_font)

        sensitivity_rect = QRectF(20, 176, 120, 24)
        sensitivity_active = self._pressed == "sensitivity"
        sensitivity_hovered = self._hovered == "sensitivity"
        painter.setPen(QPen(QColor(100, 119, 153, 230), 1.0))
        painter.setBrush(
            QColor(72, 68, 132, 245) if sensitivity_active else
            QColor(58, 55, 108, 240) if sensitivity_hovered else
            QColor(38, 48, 72, 240))
        painter.drawRoundedRect(sensitivity_rect, 9, 9)
        callback = getattr(
            self._controller, "_view_control_sensitivity_label", None)
        sensitivity_label = (
            callback() if callback is not None else "Sensitivity: Standard")
        sensitivity_font = QFont(label_font)
        sensitivity_font.setPixelSize(10)
        sensitivity_font.setBold(True)
        painter.setFont(sensitivity_font)
        painter.setPen(QColor(224, 230, 244))
        painter.drawText(
            sensitivity_rect, Qt.AlignCenter, sensitivity_label)
        painter.end()


class QtCanvas(QWidget):
    """Qt window: top bar (optional tactile selector left, display controls right),
    middle = _CanvasArea, bottom = tactile threshold slider."""

    def __init__(self, controller, title: str, w: int = 1280, h: int = 720,
                 show_slider: bool = True,
                 display_modes: list[tuple[str, str]] | None = None,
                 display_mode: str | None = None,
                 lang_button: bool = True,
                 show_baseline: bool = False,
                 show_surface_color: bool = False,
                 tactile_panel_modes: list[tuple[str, str]] | None = None,
                 tactile_panel_mode: str | None = None,
                 show_tactile_toggle: bool = True,
                 show_orientation_button: bool = False,
                 show_selector_button: bool = False):
        super().__init__()
        # Dark navy-blue chrome to match the in-canvas palette.
        self.setStyleSheet(
            "QWidget { background-color: #121826; color: #cfe0f5; }"
            "QLabel { background: transparent; }"
            "QComboBox, QPushButton { background-color: #1b2740; "
            "border: 1px solid #2e3f5e; border-radius: 4px; padding: 2px 10px; }"
            "QComboBox:hover, QPushButton:hover { background-color: #24334f; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background-color: #1b2740; "
            "color: #cfe0f5; selection-background-color: #2e4a7a; }"
            "QCheckBox { background: transparent; color: #cfe0f5; "
            "spacing: 6px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; }"
            "QSlider::groove:horizontal { background: #1b2740; height: 6px; "
            "border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #3b6ea5; width: 14px; "
            "margin: -4px 0; border-radius: 7px; }"
        )
        self._controller = controller
        self._display_modes = list(display_modes or [])
        self._tactile_panel_modes = list(tactile_panel_modes or [])
        self.setWindowTitle(title or "Live 3D")
        self._area = _CanvasArea(self, controller, w, h)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._display_combo: QComboBox | None = None
        self._display_label: QLabel | None = None
        self._lang_button: QPushButton | None = None
        self._slider: QSlider | None = None
        self._tactile_toggle: QCheckBox | None = None
        self._tactile_panel_label: QLabel | None = None
        self._tactile_panel_combo: QComboBox | None = None
        self._baseline_toggle: QCheckBox | None = None
        self._pressure_mode_label: QLabel | None = None
        self._pressure_mode_combo: QComboBox | None = None
        self._baseline_button: QPushButton | None = None
        self._baseline_status: QLabel | None = None
        self._orientation_button: QPushButton | None = None
        self._selector_button: QPushButton | None = None

        # Top bar: the optional tactile-panel selector stays at the left edge;
        # display, language and pressure controls remain right-aligned.
        if (self._display_modes or lang_button or show_baseline
                or show_surface_color or self._tactile_panel_modes
                or show_orientation_button or show_selector_button):
            top = QWidget()
            top_layout = QHBoxLayout(top)
            top_layout.setContentsMargins(10, 6, 10, 6)
            if self._tactile_panel_modes:
                self._tactile_panel_label = QLabel()
                self._tactile_panel_label.setFocusPolicy(Qt.NoFocus)
                self._tactile_panel_combo = QComboBox()
                self._tactile_panel_combo.setFocusPolicy(Qt.NoFocus)
                self._tactile_panel_combo.setMinimumHeight(36)
                self._tactile_panel_combo.setMinimumWidth(150)
                self._populate_tactile_panel_combo()
                self._select_tactile_panel_mode(tactile_panel_mode)
                self._tactile_panel_combo.currentIndexChanged.connect(
                    self._on_tactile_panel_mode)
                self._retranslate_tactile_panel()
                top_layout.addWidget(self._tactile_panel_label)
                top_layout.addWidget(self._tactile_panel_combo)
            top_layout.addStretch(1)
            if self._display_modes:
                self._display_combo = QComboBox()
                self._display_combo.setFocusPolicy(Qt.NoFocus)
                big = self._display_combo.font()
                big.setPointSize(13)
                big.setBold(True)
                self._display_combo.setFont(big)
                self._display_combo.setMinimumHeight(36)
                self._display_combo.setMinimumWidth(170)
                self._populate_display_combo()
                if display_mode is not None:
                    self._select_display_mode(display_mode)
                self._display_combo.currentIndexChanged.connect(
                    self._on_display_mode)
                self._display_label = QLabel()
                self._display_label.setFocusPolicy(Qt.NoFocus)
                label_font = self._display_label.font()
                label_font.setPointSize(13)
                self._display_label.setFont(label_font)
                self._retranslate_display_label()
                top_layout.addWidget(self._display_label)
                top_layout.addWidget(self._display_combo)
            if lang_button:
                self._lang_button = QPushButton()
                self._lang_button.setFocusPolicy(Qt.NoFocus)
                btn_font = self._lang_button.font()
                btn_font.setPointSize(13)
                btn_font.setBold(True)
                self._lang_button.setFont(btn_font)
                self._lang_button.setMinimumHeight(36)
                self._lang_button.clicked.connect(self._on_lang_toggle)
                self._retranslate_lang_button()
                top_layout.addWidget(self._lang_button)
            if show_selector_button:
                self._selector_button = QPushButton()
                self._selector_button.setFocusPolicy(Qt.NoFocus)
                btn_font = self._selector_button.font()
                btn_font.setPointSize(13)
                btn_font.setBold(True)
                self._selector_button.setFont(btn_font)
                self._selector_button.setMinimumHeight(36)
                self._selector_button.clicked.connect(self._on_selector_button)
                self._retranslate_selector_button()
                top_layout.addWidget(self._selector_button)
            if show_baseline:
                if show_tactile_toggle:
                    self._tactile_toggle = QCheckBox()
                    self._tactile_toggle.setFocusPolicy(Qt.NoFocus)
                    self._tactile_toggle.setChecked(True)
                    self._tactile_toggle.toggled.connect(
                        self._on_show_tactile_toggled)
                self._baseline_toggle = QCheckBox()
                self._baseline_toggle.setFocusPolicy(Qt.NoFocus)
                self._baseline_toggle.toggled.connect(self._on_baseline_toggled)
                self._baseline_button = QPushButton()
                self._baseline_button.setFocusPolicy(Qt.NoFocus)
                self._baseline_button.setMinimumHeight(36)
                self._baseline_button.setEnabled(False)
                self._baseline_button.clicked.connect(
                    self._on_recollect_baseline)
                self._baseline_status = QLabel()
                self._baseline_status.setFocusPolicy(Qt.NoFocus)
                self._retranslate_baseline()
                if self._tactile_toggle is not None:
                    top_layout.addWidget(self._tactile_toggle)
                top_layout.addWidget(self._baseline_toggle)
                top_layout.addWidget(self._baseline_button)
                top_layout.addWidget(self._baseline_status)
            if show_surface_color:
                self._pressure_mode_label = QLabel()
                self._pressure_mode_label.setFocusPolicy(Qt.NoFocus)
                self._pressure_mode_combo = QComboBox()
                self._pressure_mode_combo.setFocusPolicy(Qt.NoFocus)
                self._pressure_mode_combo.setMinimumHeight(36)
                self._pressure_mode_combo.setMinimumWidth(120)
                self._populate_pressure_mode_combo()
                self._pressure_mode_combo.currentIndexChanged.connect(
                    self._on_pressure_mode)
                self._retranslate_pressure_mode()
                top_layout.addWidget(self._pressure_mode_label)
                top_layout.addWidget(self._pressure_mode_combo)
            if show_orientation_button:
                self._orientation_button = QPushButton()
                self._orientation_button.setFocusPolicy(Qt.NoFocus)
                self._orientation_button.setMinimumHeight(36)
                self._orientation_button.clicked.connect(
                    self._on_recollect_orientation)
                self._retranslate_orientation_button()
                top_layout.addWidget(self._orientation_button)
            layout.addWidget(top)

        layout.addWidget(self._area, 1)

        if show_slider:
            bottom = QWidget()
            bottom_layout = QHBoxLayout(bottom)
            bottom_layout.setContentsMargins(8, 3, 8, 3)
            self._slider = QSlider(Qt.Horizontal)
            self._slider.setRange(0, 1000)
            self._slider.setValue(DEFAULT_TACTILE_THRESHOLD)
            # do not steal keyboard focus
            self._slider.setFocusPolicy(Qt.NoFocus)
            self._slider.setToolTip("tactile threshold (low cells zeroed)")
            self._slider.valueChanged.connect(self._on_slider)

            label = QLabel("tactile_thr")
            label.setFocusPolicy(Qt.NoFocus)
            bottom_layout.addWidget(label)
            bottom_layout.addWidget(self._slider, 1)
            layout.addWidget(bottom)

        extra = 28 if show_slider else 0
        if (self._display_modes or lang_button or show_baseline
                or show_surface_color or self._tactile_panel_modes
                or show_orientation_button or show_selector_button):
            extra += 48
        self.resize(w, h + extra)
        self.show()
        self._area.setFocus()

    # ---- (re)translation --------------------------------------------------
    def _display_options(self) -> list[tuple[str, str]]:
        fn = getattr(self._controller, "_display_mode_options", None)
        if fn is not None:
            return [tuple(item) for item in fn()]
        return self._display_modes

    def _populate_display_combo(self) -> None:
        if self._display_combo is None:
            return
        self._display_combo.blockSignals(True)
        self._display_combo.clear()
        for value, label in self._display_options():
            self._display_combo.addItem(label, value)
        self._display_combo.blockSignals(False)

    def _select_display_mode(self, mode: str | None) -> None:
        if self._display_combo is None:
            return
        index = self._display_combo.findData(mode)
        if index >= 0:
            self._display_combo.setCurrentIndex(index)

    def _retranslate_display_label(self) -> None:
        if self._display_label is None:
            return
        fn = getattr(self._controller, "_display_label", None)
        text = fn() if fn is not None else ""
        self._display_label.setText(text)
        self._display_label.setVisible(bool(text))

    def _retranslate_lang_button(self) -> None:
        if self._lang_button is None:
            return
        fn = getattr(self._controller, "_lang_button_label", None)
        self._lang_button.setText(fn() if fn is not None else "English")

    def _retranslate_selector_button(self) -> None:
        if self._selector_button is None:
            return
        fn = getattr(self._controller, "_selector_button_label", None)
        self._selector_button.setText(fn() if fn is not None else "Calibration")

    def retranslate(self) -> None:
        """Refresh combo items + language button text after a language change."""
        if self._display_combo is not None:
            current = self._display_combo.currentData()
            self._populate_display_combo()
            self._select_display_mode(current)
        self._retranslate_display_label()
        self._retranslate_lang_button()
        self._retranslate_baseline()
        self._retranslate_tactile_panel()
        self._retranslate_pressure_mode()
        self._retranslate_orientation_button()
        self._retranslate_selector_button()

    def _on_lang_toggle(self) -> None:
        fn = getattr(self._controller, "_toggle_language", None)
        if fn is not None:
            fn()

    def _on_selector_button(self) -> None:
        # The owning live session polls this flag and returns to the
        # calibration-file selector.
        if hasattr(self._controller, "request_selector"):
            self._controller.request_selector = True

    def _on_slider(self, value: int) -> None:
        callback = getattr(self._controller, "_on_tactile_thr", None)
        if callback is not None:
            callback(value)

    def _on_baseline_toggled(self, checked: bool) -> None:
        callback = getattr(self._controller, "_on_use_baseline_toggled", None)
        if callback is not None:
            callback(bool(checked))

    def _on_recollect_baseline(self) -> None:
        callback = getattr(self._controller, "_on_recollect_baseline", None)
        if callback is not None:
            callback()

    def _on_show_tactile_toggled(self, checked: bool) -> None:
        callback = getattr(self._controller, "_on_show_tactile_toggled", None)
        if callback is not None:
            callback(bool(checked))

    def _on_recollect_orientation(self) -> None:
        callback = getattr(self._controller, "_on_recollect_orientation", None)
        if callback is not None:
            callback()

    def _tactile_panel_options(self) -> list[tuple[str, str]]:
        fn = getattr(self._controller, "_tactile_panel_mode_options", None)
        if fn is not None:
            return [tuple(item) for item in fn()]
        return self._tactile_panel_modes

    def _populate_tactile_panel_combo(self) -> None:
        if self._tactile_panel_combo is None:
            return
        self._tactile_panel_combo.blockSignals(True)
        self._tactile_panel_combo.clear()
        for value, label in self._tactile_panel_options():
            self._tactile_panel_combo.addItem(label, value)
        self._tactile_panel_combo.blockSignals(False)

    def _select_tactile_panel_mode(self, mode: str | None) -> None:
        if self._tactile_panel_combo is None:
            return
        index = self._tactile_panel_combo.findData(mode or "off")
        self._tactile_panel_combo.setCurrentIndex(max(0, index))

    def _on_tactile_panel_mode(self, index: int) -> None:
        callback = getattr(self._controller, "_on_tactile_panel_mode", None)
        if callback is not None and self._tactile_panel_combo is not None:
            callback(self._tactile_panel_combo.itemData(index))

    def _pressure_mode_options(self) -> list[tuple[str, str]]:
        fn = getattr(self._controller, "_pressure_display_mode_options", None)
        if fn is not None:
            return [tuple(item) for item in fn()]
        return [("points", "Pressure points"), ("tiles", "Tiles")]

    def _populate_pressure_mode_combo(self) -> None:
        if self._pressure_mode_combo is None:
            return
        self._pressure_mode_combo.blockSignals(True)
        self._pressure_mode_combo.clear()
        for value, label in self._pressure_mode_options():
            self._pressure_mode_combo.addItem(label, value)
        self._pressure_mode_combo.blockSignals(False)
        self._select_pressure_mode(
            getattr(self._controller, "pressure_display_mode", "none"))

    def _select_pressure_mode(self, mode: str | None) -> None:
        if self._pressure_mode_combo is None:
            return
        index = self._pressure_mode_combo.findData(mode or "none")
        self._pressure_mode_combo.setCurrentIndex(max(0, index))

    def _on_pressure_mode(self, index: int) -> None:
        callback = getattr(self._controller, "_on_pressure_display_mode", None)
        if callback is not None and self._pressure_mode_combo is not None:
            callback(self._pressure_mode_combo.itemData(index))

    def _retranslate_baseline(self) -> None:
        if self._baseline_toggle is None:
            return
        fn = getattr(self._controller, "_baseline_toggle_label", None)
        self._baseline_toggle.setText(fn() if fn is not None else "使用基线")
        fn2 = getattr(self._controller, "_baseline_recollect_label", None)
        self._baseline_button.setText(
            fn2() if fn2 is not None else "重新采集基线")
        if self._tactile_toggle is not None:
            fn3 = getattr(self._controller, "_tactile_toggle_label", None)
            self._tactile_toggle.setText(
                fn3() if fn3 is not None else "显示压力矩阵")
        self._refresh_baseline_status()

    def _retranslate_tactile_panel(self) -> None:
        if self._tactile_panel_combo is None:
            return
        current = self._tactile_panel_combo.currentData() or "off"
        self._populate_tactile_panel_combo()
        self._select_tactile_panel_mode(current)
        fn = getattr(self._controller, "_tactile_panel_label", None)
        if self._tactile_panel_label is not None:
            self._tactile_panel_label.setText(
                fn() if fn is not None else "Tactile panel")

    def _retranslate_pressure_mode(self) -> None:
        if self._pressure_mode_combo is None:
            return
        current = self._pressure_mode_combo.currentData() or "none"
        self._populate_pressure_mode_combo()
        self._select_pressure_mode(current)
        fn = getattr(self._controller, "_pressure_display_mode_label", None)
        if self._pressure_mode_label is not None:
            self._pressure_mode_label.setText(
                fn() if fn is not None else "Pressure display")

    def _retranslate_orientation_button(self) -> None:
        if self._orientation_button is None:
            return
        fn = getattr(self._controller, "_orientation_recollect_label", None)
        self._orientation_button.setText(
            fn() if fn is not None else "Recapture orientation")

    def _refresh_baseline_status(self) -> None:
        if self._baseline_status is None:
            return
        fn = getattr(self._controller, "_baseline_status_text", None)
        self._baseline_status.setText(fn() if fn is not None else "")

    def set_baseline_status(self, text: str) -> None:
        if self._baseline_status is not None:
            self._baseline_status.setText(text)

    def set_baseline_button_enabled(self, enabled: bool) -> None:
        if self._baseline_button is not None:
            self._baseline_button.setEnabled(bool(enabled))

    def show_baseline_overlay(self, text: str) -> None:
        """Show a large centre-top banner on the canvas."""
        self._area.show_baseline_overlay(text)

    def hide_baseline_overlay(self) -> None:
        """Hide the large centre-top banner."""
        self._area.hide_baseline_overlay()

    def _on_display_mode(self, index: int) -> None:
        callback = getattr(self._controller, "_on_display_mode", None)
        if callback is not None:
            callback(self._display_combo.itemData(index))

    def set_canvas(self, bgr: np.ndarray) -> None:
        """Give the 720x1280x3 uint8 BGR canvas to Qt for display (zero-copy QImage).

        QImage does not copy the pixels, so the numpy array reference must be
        kept alive until the next set_canvas; otherwise the memory is freed and
        the draw uses a dangling pointer.
        """
        arr = np.ascontiguousarray(bgr)
        h, w = arr.shape[:2]
        qimage = QImage(
            arr.data, w, h, arr.strides[0],
            QImage.Format.Format_BGR888)
        self._bgr = arr  # keep-alive
        self._image = qimage
        self._area.set_image(qimage)
        self._area._view_control.update()

    def closeEvent(self, ev) -> None:  # noqa: N802
        self._controller.exit_requested = True
        ev.accept()
