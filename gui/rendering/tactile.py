"""Plaintext 16x16 tactile visualization shared by live view and replay."""

from __future__ import annotations

import cv2
import numpy as np


# Right hand: verified against the live 16x16 grid (2026-08-29).  All five
# fingers form a vertical band in columns 12-15, stacked along rows:
# thumb 1-3 / index 4-6 / middle 7-9 / ring 10-12 / pinky 13-15.
# Palm = rows 1-15 x cols 3-11.  On the current right glove the previous
# display orientation was reversed on both axes, so rows run forward and
# columns backward to rotate only the palm panel by 180 degrees.
HAND_REGIONS_RIGHT = {
    "thumb": {"name": "Thumb", "rows": [1, 2, 3], "cols": [15, 14, 13, 12]},
    "index": {"name": "Index", "rows": [4, 5, 6], "cols": [15, 14, 13, 12]},
    "middle": {"name": "Middle", "rows": [7, 8, 9], "cols": [15, 14, 13, 12]},
    "ring": {"name": "Ring", "rows": [10, 11, 12], "cols": [15, 14, 13, 12]},
    "pinky": {"name": "Pinky", "rows": [13, 14, 15], "cols": [15, 14, 13, 12]},
    "palm": {"name": "Palm", "rows": list(range(1, 16)), "cols": list(range(11, 2, -1))},
}
# Left hand: all five fingers form a horizontal band in rows 0-3:
# pinky cols 0-2 / ring 3-5 / middle 6-8 / index 9-11 / thumb 12-14.
# Palm = rows 4-12 x cols 0-14.  Reverse both palm axes so the left-palm
# panel is upright after the renderer's left-hand rotation/mirroring.
HAND_REGIONS_LEFT = {
    "thumb": {"name": "Thumb", "rows": [3, 2, 1, 0], "cols": [12, 13, 14]},
    "index": {"name": "Index", "rows": [3, 2, 1, 0], "cols": [9, 10, 11]},
    "middle": {"name": "Middle", "rows": [3, 2, 1, 0], "cols": [6, 7, 8]},
    "ring": {"name": "Ring", "rows": [3, 2, 1, 0], "cols": [3, 4, 5]},
    "pinky": {"name": "Pinky", "rows": [3, 2, 1, 0], "cols": [0, 1, 2]},
    "palm": {
        "name": "Palm",
        "rows": list(range(12, 3, -1)),
        "cols": list(range(14, -1, -1)),
    },
}

HAND_W, HAND_H = 800, 220
FINGER_CELL = 16
GAP = 15


def render_hand(data: np.ndarray, mirror: bool = False,
                side: str = "right") -> np.ndarray:
    """Render one pressure matrix as the compact live-view hand panel."""

    values = np.asarray(data, np.float32).reshape(16, 16)
    values = np.maximum(0, values)
    vmax = max(float(values.max()), 1.0)
    lut = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8).reshape(1, 256),
        cv2.COLORMAP_VIRIDIS,
    )[0]
    regions = HAND_REGIONS_RIGHT if side == "right" else HAND_REGIONS_LEFT
    canvas = np.full((HAND_H, HAND_W, 3), (40, 30, 24), dtype=np.uint8)
    # Fingers render upright (taller than wide): rotate only when the finger
    # region is taller than it is wide in matrix space.  The current layouts
    # are 3 rows x 4 cols, so no rotation is applied.
    rotate = len(regions["thumb"]["rows"]) > len(regions["thumb"]["cols"])
    thumb = regions["thumb"]
    finger_w = (len(thumb["cols"]) if rotate else len(thumb["rows"])) * FINGER_CELL
    finger_h = (len(thumb["rows"]) if rotate else len(thumb["cols"])) * FINGER_CELL
    palm_rows = len(regions["palm"]["rows"])
    palm_cols = len(regions["palm"]["cols"])
    palm_vertical = palm_rows if rotate else palm_cols
    palm_horizontal = palm_cols if rotate else palm_rows
    palm_y = 20 + finger_h + 25
    palm_cell = min(FINGER_CELL, max(4, (HAND_H - palm_y - 16) // palm_vertical))
    palm_x = (HAND_W - palm_horizontal * palm_cell) // 2
    start_x = (HAND_W - (5 * finger_w + 4 * GAP)) // 2
    layout = {
        "thumb": (start_x, 20),
        "index": (start_x + finger_w + GAP, 20),
        "middle": (start_x + 2 * (finger_w + GAP), 20),
        "ring": (start_x + 3 * (finger_w + GAP), 20),
        "pinky": (start_x + 4 * (finger_w + GAP), 20),
        "palm": (palm_x, palm_y),
    }
    for name, region in regions.items():
        sx, sy = layout[name]
        rows, cols = region["rows"], region["cols"]
        cell = palm_cell if name == "palm" else FINGER_CELL
        block_width = (len(cols) if rotate else len(rows)) * cell
        block_height = (len(rows) if rotate else len(cols)) * cell
        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(cols):
                value = float(values[row, column])
                color_index = int(min(255, value / vmax * 255))
                blue, green, red = (int(x) for x in lut[color_index])
                if rotate:
                    x_index, y_index = column_index, len(rows) - 1 - row_index
                else:
                    x_index, y_index = row_index, column_index
                x1, y1 = sx + x_index * cell, sy + y_index * cell
                x2, y2 = x1 + cell, y1 + cell
                if mirror:
                    x1, x2 = HAND_W - x2, HAND_W - x1
                cv2.rectangle(canvas, (x1, y1), (x2, y2),
                              (blue, green, red), -1)
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)
                if cell >= 12:
                    cv2.putText(canvas, str(int(value)), (x1 + 2, y1 + 12),
                                cv2.FONT_HERSHEY_PLAIN, 0.45,
                                text_color, 1, cv2.LINE_AA)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (105, 115, 140), 1)
        label_x = sx + max(0, (block_width - 28) // 2)
        if mirror:
            label_x = (HAND_W - (sx + block_width)
                       + max(0, (block_width - 28) // 2))
        cv2.putText(
            canvas,
            region["name"],
            (label_x, sy + block_height + 12),
            cv2.FONT_HERSHEY_PLAIN,
            0.65,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    return canvas


GRID_CELL = 24
GRID_MARGIN = 24
GRID_W = GRID_MARGIN + 16 * GRID_CELL + 4
GRID_H = GRID_MARGIN + 16 * GRID_CELL + 4


def render_grid(data: np.ndarray, cell: int = GRID_CELL) -> np.ndarray:
    """Render the full 16x16 matrix with row/column indices.

    Complement to :func:`render_hand`: the hand view hides the cells outside
    the six hand regions (row 15 and columns 13-15 on the left-hand layout,
    rows 0-2 and column 0 on the right), while this grid shows all 256 cells
    so presses in the hidden corner stay visible.
    """
    values = np.asarray(data, np.float32).reshape(16, 16)
    values = np.maximum(0, values)
    vmax = max(float(values.max()), 1.0)
    lut = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8).reshape(1, 256), cv2.COLORMAP_VIRIDIS)[0]
    canvas = np.full((GRID_H, GRID_W, 3), (40, 30, 24), np.uint8)
    for row in range(16):
        for col in range(16):
            color_index = int(min(255, float(values[row, col]) / vmax * 255))
            blue, green, red = (int(x) for x in lut[color_index])
            x0, y0 = GRID_MARGIN + col * cell, GRID_MARGIN + row * cell
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell),
                          (blue, green, red), -1)
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)
            cv2.putText(canvas, str(int(values[row, col])),
                        (x0 + 1, y0 + cell - 5),
                        cv2.FONT_HERSHEY_PLAIN, 0.35, text_color, 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell),
                          (105, 115, 140), 1)
    for col in range(16):
        cv2.putText(canvas, str(col),
                    (GRID_MARGIN + col * cell + cell // 2 - 4, GRID_MARGIN - 8),
                    cv2.FONT_HERSHEY_PLAIN, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    for row in range(16):
        cv2.putText(canvas, str(row),
                    (8, GRID_MARGIN + row * cell + cell // 2 + 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


__all__ = ["render_hand", "render_grid"]
