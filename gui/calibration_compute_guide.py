"""One shared computation image for both hands and interface languages."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication


def render_compute_guide_bgr(*, dual: bool = False, busy: bool = False,
                             lang: str = "zh", width: int = 660,
                             height: int = 300) -> np.ndarray:
    """Render the same Calculating image, independently of mode/language.

    Mode/state keywords remain accepted by existing callers. The caller owns
    QApplication; this helper only paints an in-memory image on demand.
    """
    if (isinstance(width, bool) or not isinstance(width, int) or width <= 0
            or isinstance(height, bool) or not isinstance(height, int) or height <= 0):
        raise ValueError("width and height must be positive integers")
    if not isinstance(QApplication.instance(), QApplication):
        raise RuntimeError("render_compute_guide_bgr requires a QApplication")
    image = QImage(width, height, QImage.Format_RGBA8888)
    image.fill(QColor("#111c2b"))
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.TextAntialiasing)
        font = QFont()
        font.setFamilies(["Segoe UI", "Arial", "Microsoft YaHei", "DejaVu Sans", "sans-serif"])
        font.setPixelSize(max(1, round(36 * min(width / 660, height / 300))))
        painter.setFont(font)
        painter.setPen(QColor("#e6edf5"))
        painter.drawText(image.rect(), Qt.AlignCenter, "Calculating")
    finally:
        painter.end()
    rgba = np.frombuffer(image.constBits(), dtype=np.uint8)
    rgba = rgba.reshape(height, image.bytesPerLine())[:, :width * 4].reshape(height, width, 4)
    return np.ascontiguousarray(rgba[..., [2, 1, 0]])
