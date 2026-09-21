"""Qt (PySide6) display/input layer, replacing the OpenCV HighGUI window of Live3DViewer.

Design notes (composed, to avoid name clashes):
- ``QtCanvas`` and ``_CanvasArea`` are plain QWidgets; they do **not** inherit
  Live3DViewer -- Live3DViewer already has ``update()``/``close()``, and
  inheriting QWidget directly would clash with QWidget.update()/close().
- Live3DViewer only produces the numpy canvas (_draw outputs a uint8 BGR array
  whose size follows the window, see canvas_pixel_size_for); QtCanvas displays
  it via ``QImage(Format_BGR888)`` (zero-copy) and
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

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSlider, QVBoxLayout, QWidget)

from common.i18n import L  # noqa: E402

# Default tactile threshold applied at startup: cells below this ADC value are
# zeroed so tiny baseline / sensor fluctuations do not flicker the cell numbers.
DEFAULT_TACTILE_THRESHOLD = 60

# Canvas sizing.  The live 3D scene is CPU-rasterised, so the canvas follows
# the window's own pixels instead of a fixed 1280x720 that Qt then magnified:
# that magnification was why a fullscreen window went blurry while a windowed
# one stayed sharp.  The ceiling is a performance budget, not an aesthetic one
# -- cost is linear in pixel count and the auto viewer's default redraw rate is
# --fps 60, i.e. a 16.7 ms budget.
DEFAULT_CANVAS_W, DEFAULT_CANVAS_H = 1280, 720
MIN_CANVAS_W, MIN_CANVAS_H = 640, 360
MAX_CANVAS_W, MAX_CANVAS_H = 3840, 2160
# Measured _draw() cost in the default "hand" display mode, against the
# 16.7 ms budget: 1280x712 -> 7.0 ms (42%), 1920x952 -> 10.2 ms (61%),
# 1980x1044 -> 11.5 ms (69%).  This cap therefore covers a maximised window on
# a 1080p panel natively with room to spare, and leaves a 1440p or 4K window
# rendering at this size for Qt to scale up -- still far sharper than the
# ~1.5x nearest-neighbour upscale this replaced.  It is deliberately NOT
# raised to 2560 * 1440: measured, that canvas costs ~17.8 ms, which is over
# budget and would drop the default 60 fps.
MAX_CANVAS_PIXELS = 1920 * 1080
# Render sizes are floored to a multiple of this: it keeps every scanline
# 32-bit aligned on any Qt backend for free, and flooring (rather than
# rounding) guarantees the widget is never smaller than the canvas, so Qt is
# never asked to downscale the result of a size change.
CANVAS_SNAP = 4


def canvas_pixel_size_for(w: int, h: int, dpr: float = 1.0) -> tuple[int, int]:
    """Render target in device pixels for a widget of this logical size.

    Pure, so it can be asserted offline (tools/self_test.py).  A degenerate
    or not-yet-laid-out size falls back to the design baseline; an oversized
    one is shrunk uniformly to MAX_CANVAS_PIXELS with the aspect preserved.
    """
    if w < MIN_CANVAS_W or h < MIN_CANVAS_H:
        return DEFAULT_CANVAS_W, DEFAULT_CANVAS_H
    scale = max(1.0, float(dpr))
    cw = min(int(round(w * scale)), MAX_CANVAS_W)
    ch = min(int(round(h * scale)), MAX_CANVAS_H)
    pixels = cw * ch
    if pixels > MAX_CANVAS_PIXELS:
        shrink = (MAX_CANVAS_PIXELS / float(pixels)) ** 0.5
        cw = max(1, int(cw * shrink))
        ch = max(1, int(ch * shrink))
    cw = max(MIN_CANVAS_W, cw // CANVAS_SNAP * CANVAS_SNAP)
    ch = max(MIN_CANVAS_H, ch // CANVAS_SNAP * CANVAS_SNAP)
    return cw, ch


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

    # Emitted whenever the floating orbit pad is shown or hidden, so the owning
    # QtCanvas can keep its top-bar toggle button in sync with a close that came
    # from the pad itself.
    view_control_visibility_changed = Signal(bool)

    def __init__(self, parent: QWidget | None, controller, canvas_w: int,
                 canvas_h: int):
        super().__init__(parent)
        self._controller = controller
        self._cw = int(canvas_w)
        self._ch = int(canvas_h)
        self._image: QImage | None = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._view_control_shown = True
        self._view_control = _ViewControl(self, controller)
        self._place_view_control()
        self._baseline_overlay: QLabel | None = None
        self._setup_baseline_overlay()

    def canvas_pixel_size(self) -> tuple[int, int]:
        """Canvas size in device pixels that matches the current widget geometry."""
        return canvas_pixel_size_for(
            self.width(), self.height(), self.devicePixelRatioF())

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

    def view_control_visible(self) -> bool:
        """Whether the floating orbit pad is currently shown."""
        return self._view_control_shown

    def set_view_control_visible(self, visible: bool) -> None:
        """Show or hide the floating orbit pad.

        Driven by both the pad's own close button and the top bar's toggle; the
        signal keeps whichever control the user did *not* touch in sync.
        """
        visible = bool(visible)
        if visible == self._view_control_shown:
            return
        self._view_control_shown = visible
        self._view_control.setVisible(visible)
        if visible:
            # Re-anchor on the way back, honouring any earlier manual drag.
            self._place_view_control()
        self.view_control_visibility_changed.emit(visible)

    def _place_view_control(self) -> None:
        """Place the compact camera control between record and tactile cards.

        Once the user has dragged the panel somewhere, that position is what is
        kept: a resize only nudges it back inside the window.  Re-anchoring it
        to the default corner would silently undo the move on the next resize.
        """
        margin = 18
        if self._view_control.was_moved_by_user():
            self._view_control._move_to(self._view_control.pos())
            return
        preferred_y = 70
        max_y = max(margin, self.height() - self._view_control.height() - margin)
        self._view_control.move(
            max(margin, self.width() - self._view_control.width() - margin),
            min(preferred_y, max_y))
        self._view_control.raise_()

    # ---- Canvas display ----------------------------------------------------
    def set_image(self, qimage: QImage | None) -> None:
        self._image = qimage
        # Keep the cached canvas size in step with the image.  _to_canvas maps
        # widget coordinates into canvas coordinates with it, so a stale value
        # would silently break every hit-test -- the record button, the tactile
        # resize grips, drag-pan -- as soon as the canvas stops being the
        # 1280x720 it was constructed with.  They fail by not responding, not
        # by raising, so this must stay wired to the image itself.
        if qimage is not None and qimage.width() > 0:
            self._cw, self._ch = qimage.width(), qimage.height()
        self.update()

    def fitted_rect(self) -> QRect:
        """Fit the canvas into the widget rect at its aspect ratio (letterbox).

        With a canvas sized from canvas_pixel_size() this is an exact 1:1 blit
        of the widget rect, so nothing is scaled and the sharpness of the
        canvas reaches the screen unchanged.
        """
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
            # Qt defaults to nearest-neighbour, which reads as blocky for the
            # one frame where a resize outruns the renderer, and for any window
            # large enough to hit the MAX_CANVAS_PIXELS cap.  At 1:1 this is a
            # no-op, and it costs nothing measurable.
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
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

    The panel is also *movable*: pressing any part of it that is not a button
    and not the roll ring grabs the whole widget, so a user can drag it off
    whatever it happens to be covering.  That is why it floats on the canvas
    instead of living in the top bar -- the thing it covers changes with the
    view.
    """

    WIDTH, HEIGHT = 160, 210
    CX, CY = 80.0, 70.0
    OUTER_R, INNER_R = 62.0, 44.0
    # Small, deliberately understated: it sits in the panel's top-right corner,
    # clear of the roll ring, and must not compete with the pad itself.  The
    # top bar's toggle button is the discoverable way back.
    CLOSE_RECT = QRectF(136, 10, 16, 16)

    def __init__(self, parent: QWidget, controller):
        super().__init__(parent)
        self._controller = controller
        self._pressed: str | None = None
        self._hovered: str | None = None
        self._rolling = False
        self._last_pointer_angle = 0.0
        # Offset from the pointer to the widget's top-left while dragging the
        # panel itself; ``None`` means no grab is in progress.
        self._drag_offset: QPoint | None = None
        # Once the user has placed the panel, the canvas stops re-anchoring it
        # to its default corner on every resize.
        self._moved_by_user = False
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
            "底部按钮：切换旋转灵敏度\n"
            "拖动空白处：移动此面板\n"
            "右上角 ×：关闭（顶栏按钮可随时重新打开）")

    def was_moved_by_user(self) -> bool:
        """Whether the user has placed the panel, rather than the default anchor."""
        return self._moved_by_user

    def _request_close(self) -> None:
        """Hide the panel; the top bar's toggle button is the way back.

        Routed through the parent canvas rather than a bare ``hide()`` so a
        close driven from the panel keeps the toggle button's label in sync.
        """
        parent = self.parentWidget()
        setter = getattr(parent, "set_view_control_visible", None)
        if setter is not None:
            setter(False)
        else:
            self.hide()

    def _move_to(self, top_left: QPoint) -> None:
        """Move the whole panel, keeping it inside the canvas it floats over."""
        parent = self.parentWidget()
        x, y = int(top_left.x()), int(top_left.y())
        if parent is not None:
            x = max(0, min(x, max(0, parent.width() - self.width())))
            y = max(0, min(y, max(0, parent.height() - self.height())))
        self._moved_by_user = True
        self.move(x, y)
        self.raise_()

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
        point = QPointF(pos)
        # Checked first: the close button overlays the panel corner, so it has
        # to win over the "move" grip that covers everything else.
        if self.CLOSE_RECT.contains(point):
            return "close"
        if QRectF(20, 176, 120, 24).contains(point):
            return "sensitivity"
        radius = self._radius(pos)
        if self.INNER_R <= radius <= self.OUTER_R + 5.0:
            return "roll"
        for name, rect in self._button_rects().items():
            if rect.contains(point):
                return name
        # Everything else inside the panel is a grip.  The readout strip under
        # the ring, the corners and the slack between the direction buttons are
        # not controls, so pressing there moves the panel rather than doing
        # nothing -- which is the only way to uncover what it is hiding.
        if QRectF(0, 0, self.WIDTH, self.HEIGHT).contains(point):
            return "move"
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
        if hit == "move":
            self._drag_offset = (
                ev.globalPosition().toPoint() - self.frameGeometry().topLeft())
            self.setCursor(Qt.ClosedHandCursor)
        elif hit == "roll":
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
        elif hit == "close":
            self._request_close()
        elif hit is not None:
            self._repeat_direction()
            self._repeat.start()
        self.update()
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._drag_offset is not None and (ev.buttons() & Qt.LeftButton):
            self._move_to(
                ev.globalPosition().toPoint() - self._drag_offset)
            ev.accept()
            return
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
                if hovered == "move":
                    self.setCursor(Qt.SizeAllCursor)
                elif hovered is not None:
                    self.setCursor(Qt.PointingHandCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
                self.update()
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        dragged = self._drag_offset is not None
        self._repeat.stop()
        self._pressed = None
        self._rolling = False
        self._drag_offset = None
        if dragged:
            # The pointer is still over the panel, so restore the neutral
            # cursor; the next move re-derives the hover one.
            self.setCursor(Qt.ArrowCursor)
        self.update()
        # Return keyboard shortcuts to the 3D surface after using the pad.
        if self.parentWidget() is not None:
            self.parentWidget().setFocus()
        ev.accept()

    def leaveEvent(self, ev) -> None:  # noqa: N802
        if not self._rolling and self._drag_offset is None:
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

        # Close button: a small x in the panel's top-right corner.  Held at low
        # contrast until hovered so it stays secondary to the pad itself.
        close_rect = self.CLOSE_RECT
        close_active = self._pressed == "close"
        close_hovered = self._hovered == "close"
        painter.setPen(QPen(QColor(100, 119, 153, 200), 1.0))
        painter.setBrush(
            QColor(96, 52, 62, 245) if close_active else
            QColor(74, 44, 54, 240) if close_hovered else
            QColor(30, 42, 60, 200))
        painter.drawRoundedRect(close_rect, 5, 5)
        painter.setPen(QPen(
            QColor(255, 226, 226) if (close_active or close_hovered)
            else QColor(176, 190, 210),
            1.6, Qt.SolidLine, Qt.RoundCap))
        pad = 4.0
        painter.drawLine(
            QPointF(close_rect.left() + pad, close_rect.top() + pad),
            QPointF(close_rect.right() - pad, close_rect.bottom() - pad))
        painter.drawLine(
            QPointF(close_rect.right() - pad, close_rect.top() + pad),
            QPointF(close_rect.left() + pad, close_rect.bottom() - pad))
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
                 show_selector_button: bool = False,
                 show_dongle_reboot_button: bool = False,
                 show_view_control_button: bool = True):
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
        # Windows can choose different fallback faces for Chinese and Latin
        # glyphs.  A single UI face keeps both language modes at identical
        # metrics, preventing the toolbar from looking oversized after a
        # language switch.  Qt falls back safely on non-Windows platforms.
        self.setFont(QFont("Microsoft YaHei UI", 10))
        # The canvas itself aspect-fits its 1280x720 source, so it is safe to
        # let the window shrink to a normal laptop-sized work area.  Controls
        # that cannot fit horizontally live in a scrollable toolbar below.
        self.setMinimumSize(720, 480)
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
        self._dongle_reboot_button: QPushButton | None = None
        self._view_control_button: QPushButton | None = None

        # Top bar: the optional tactile-panel selector stays at the left edge;
        # display, language and pressure controls remain right-aligned.
        if (self._display_modes or lang_button or show_baseline
                or show_surface_color or self._tactile_panel_modes
                or show_orientation_button or show_selector_button
                or show_dongle_reboot_button or show_view_control_button):
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
                self._retranslate_display_label()
                top_layout.addWidget(self._display_label)
                top_layout.addWidget(self._display_combo)
            if lang_button:
                self._lang_button = QPushButton()
                self._lang_button.setFocusPolicy(Qt.NoFocus)
                btn_font = self._lang_button.font()
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
                btn_font.setBold(True)
                self._selector_button.setFont(btn_font)
                self._selector_button.setMinimumHeight(36)
                self._selector_button.clicked.connect(self._on_selector_button)
                self._retranslate_selector_button()
                top_layout.addWidget(self._selector_button)
            if show_dongle_reboot_button:
                self._dongle_reboot_button = QPushButton()
                self._dongle_reboot_button.setFocusPolicy(Qt.NoFocus)
                btn_font = self._dongle_reboot_button.font()
                btn_font.setBold(True)
                self._dongle_reboot_button.setFont(btn_font)
                self._dongle_reboot_button.setMinimumHeight(36)
                self._dongle_reboot_button.clicked.connect(
                    self._on_dongle_reboot)
                self._retranslate_dongle_reboot_button()
                top_layout.addWidget(self._dongle_reboot_button)
                # Born hidden: the controller reveals it once a Bluetooth link
                # is detected (see set_dongle_reboot_visible).
                self._dongle_reboot_button.setVisible(False)
            if show_view_control_button:
                self._view_control_button = QPushButton()
                self._view_control_button.setFocusPolicy(Qt.NoFocus)
                btn_font = self._view_control_button.font()
                btn_font.setBold(True)
                self._view_control_button.setFont(btn_font)
                self._view_control_button.setMinimumHeight(36)
                self._view_control_button.clicked.connect(
                    self._on_view_control_toggle)
                self._retranslate_view_control_button()
                top_layout.addWidget(self._view_control_button)
                # The pad's own close button hides it directly, so the toggle
                # label follows the canvas rather than only its own clicks.
                self._area.view_control_visibility_changed.connect(
                    self._on_view_control_visibility_changed)
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
                # Born hidden: the controller reveals the group once a tactile
                # panel view is selected (see set_baseline_controls_visible).
                self.set_baseline_controls_visible(False)
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
            # Do not elide or hide controls at narrower widths.  The content
            # keeps its natural minimum width and the bar gains a horizontal
            # scrollbar only when needed; at ordinary desktop widths it still
            # behaves like the original single-row toolbar.
            top.setMinimumWidth(top_layout.minimumSize().width())
            toolbar = QScrollArea()
            toolbar.setWidget(top)
            toolbar.setWidgetResizable(True)
            toolbar.setFrameShape(QFrame.NoFrame)
            toolbar.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            toolbar.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            toolbar.setMinimumHeight(48)
            toolbar.setMaximumHeight(66)
            layout.addWidget(toolbar)

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
                or show_orientation_button or show_selector_button
                or show_dongle_reboot_button):
            extra += 48
        # A 1280x720 canvas plus chrome is taller than the usable height of
        # many 1366x768 laptops.  Start within the available work area instead
        # of opening partly off-screen; users can still enlarge it afterwards.
        available = QApplication.primaryScreen().availableGeometry()
        target_w = min(w, max(720, int(available.width() * 0.95)))
        target_h = min(h + extra, max(480, int(available.height() * 0.92)))
        self.resize(target_w, target_h)
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

    def _retranslate_dongle_reboot_button(self) -> None:
        if self._dongle_reboot_button is None:
            return
        fn = getattr(self._controller, "_dongle_reboot_button_label", None)
        self._dongle_reboot_button.setText(
            fn() if fn is not None else "Reboot Dongle")

    def _retranslate_view_control_button(self) -> None:
        """Label the pad toggle after the pad's *current* state, not this call."""
        if self._view_control_button is None:
            return
        if self._area.view_control_visible():
            self._view_control_button.setText(L("隐藏旋转盘", "Hide Pad"))
        else:
            self._view_control_button.setText(L("显示旋转盘", "Show Pad"))

    def _on_view_control_toggle(self) -> None:
        self._area.set_view_control_visible(
            not self._area.view_control_visible())

    def _on_view_control_visibility_changed(self) -> None:
        # Signal carries the new state, but the label re-reads it from the
        # canvas so the two can never disagree.
        self._retranslate_view_control_button()

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
        self._retranslate_dongle_reboot_button()
        self._retranslate_view_control_button()

    def _on_lang_toggle(self) -> None:
        fn = getattr(self._controller, "_toggle_language", None)
        if fn is not None:
            fn()

    def _on_selector_button(self) -> None:
        # The owning live session polls this flag and returns to the
        # calibration-file selector.
        if hasattr(self._controller, "request_selector"):
            self._controller.request_selector = True

    def _on_dongle_reboot(self) -> None:
        callback = getattr(self._controller, "_on_reboot_dongle", None)
        if callback is not None:
            callback()

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

    def set_dongle_reboot_visible(self, visible: bool) -> None:
        """Show the dongle reboot button only while a Bluetooth link exists.

        ``AT+REBOOT`` is a Bluetooth-dongle command: on a wired glove there is
        nothing to send it to, so the button stays out of the bar until a dongle
        link is actually detected.  Only visibility is driven -- the button, its
        label and its handler are unchanged.
        """
        if self._dongle_reboot_button is not None:
            self._dongle_reboot_button.setVisible(bool(visible))

    def set_baseline_controls_visible(self, visible: bool) -> None:
        """Show the pressure-baseline controls only while a tactile panel is on.

        The toggle and its re-collect button act on the tactile panel's own
        maps, so the group appears together with one of the panel's views and is
        hidden while the panel is off.  Only visibility is driven here: the
        toggle's state and the button's function are left exactly as they were,
        so hiding the group never switches baseline correction off behind the
        user's back.
        """
        visible = bool(visible)
        for widget in (self._baseline_toggle, self._baseline_button,
                       self._baseline_status):
            if widget is not None:
                widget.setVisible(visible)

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

    def canvas_pixel_size(self) -> tuple[int, int]:
        """Canvas size the renderer should produce for the window's current geometry."""
        return self._area.canvas_pixel_size()

    def set_canvas(self, bgr: np.ndarray) -> None:
        """Give the uint8 BGR canvas to Qt for display (zero-copy QImage).

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
