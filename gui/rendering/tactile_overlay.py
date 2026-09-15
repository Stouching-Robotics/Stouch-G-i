"""Active-region tactile UI cards for the live and replay canvases.

The authoritative sensor-to-finger mapping remains in ``tactile.py``.  This
module only projects those existing regions onto a compact 15x13 display and
draws it: five color-coded finger zones above a full-width blue palm grid.
No solver, calibration, recording, threshold, or channel mapping lives here.
"""

from __future__ import annotations

import cv2
import numpy as np

from gui.rendering.tactile import HAND_REGIONS_LEFT, HAND_REGIONS_RIGHT


_SIDES = ("left", "right")
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_FINGER_ORDER = {
    "left": ("pinky", "ring", "middle", "index", "thumb"),
    "right": _FINGER_NAMES,
}
_GRID_COLS = 15
_GRID_ROWS = 13
_FINGER_SHORT = {
    "thumb": "THM",
    "index": "IDX",
    "middle": "MID",
    "ring": "RNG",
    "pinky": "PNK",
}
# Finger identity accents, BGR: red, amber, green, blue, violet.
_REGION_BGR = {
    "thumb": (108, 107, 255),
    "index": (85, 190, 255),
    "middle": (128, 214, 78),
    "ring": (255, 165, 72),
    "pinky": (255, 118, 184),
    "palm": (235, 184, 54),
}

# Fixed discrete colour scales (glove-test v1.4): one interval -> one colour.
# Two scales are needed because raw and baseline-corrected values occupy very
# different ADC ranges; a single scale would collapse one of them to a single
# colour.  The raw scale spans the full 12-bit ADC range (used when the
# baseline option is off, so raw values do not all fall into the top band);
# the corrected scale covers the baseline-subtracted contact range (used when
# the option is on).  Colours are BGR.
_PRESSURE_BANDS_CORRECTED = (
    (333, (95, 58, 30)),            # #1e3a5f dark blue
    (666, (199, 134, 22)),          # #1686c7 blue
    (999, (94, 197, 34)),           # #22c55e green
    (1333, (21, 204, 250)),         # #facc15 yellow
    (1666, (22, 115, 249)),         # #f97316 orange
    (float("inf"), (68, 68, 239)),  # #ef4444 red, no upper limit
)
_PRESSURE_BANDS_RAW = (
    (2000, (95, 58, 30)),           # #1e3a5f dark blue (~empty-glove floor)
    (2400, (199, 134, 22)),         # #1686c7 blue
    (2800, (94, 197, 34)),          # #22c55e green
    (3200, (21, 204, 250)),         # #facc15 yellow
    (3600, (22, 115, 249)),         # #f97316 orange
    (float("inf"), (68, 68, 239)),  # #ef4444 red, no upper limit
)


def _band_colors(values: np.ndarray, bands: tuple) -> np.ndarray:
    """Map a pressure array to fixed discrete colour bands (one per cell)."""
    values = np.maximum(0.0, np.asarray(values, np.float32))
    upper = np.asarray([band[0] for band in bands], np.float32)
    palette = np.asarray([band[1] for band in bands], np.uint8)
    indices = np.searchsorted(upper, values, side="left")
    indices = np.clip(indices, 0, len(bands) - 1)
    return palette[indices]

_CARD_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
_CARD_CACHE_MAX = 8
_TEMPLATE_CACHE: dict[tuple[str, int], tuple] = {}
_CARD_HEADER = 34
_CARD_PAD_X = 10
_CARD_PAD_TOP = 6
_AXIS_LEFT = 25
_REGION_LABEL_HEIGHT = 17
_X_TICK_HEIGHT = 14
_CARD_FOOTER = 21
_CARD_BOTTOM_MARGIN = 60


def _display_matrix(values: np.ndarray, side: str) -> np.ndarray:
    """Place every active region using render_hand's established orientation."""

    regions = HAND_REGIONS_LEFT if side == "left" else HAND_REGIONS_RIGHT
    _, bounds = _REGION_LAYOUT[side]
    displayed = np.zeros((_GRID_ROWS, _GRID_COLS), np.float32)
    for name, config in regions.items():
        block = values[np.ix_(config["rows"], config["cols"])]
        # Exact cell order used by render_hand(..., mirror=side == "left").
        oriented = block[::-1, ::-1] if side == "left" else block.T
        col0, row0, col1, row1 = bounds[name]
        displayed[row0:row1, col0:col1] = oriented
    return np.ascontiguousarray(displayed)


def _build_region_layout(side: str):
    """Build a compact, mirrored layout from the authoritative region table."""

    regions = HAND_REGIONS_LEFT if side == "left" else HAND_REGIONS_RIGHT
    region_at = np.full((_GRID_ROWS, _GRID_COLS), "", dtype="U6")
    bounds: dict[str, tuple[int, int, int, int]] = {}

    cursor = 0
    finger_height = None
    for name in _FINGER_ORDER[side]:
        config = regions[name]
        raw_shape = (len(config["rows"]), len(config["cols"]))
        display_shape = raw_shape if side == "left" else raw_shape[::-1]
        height, width = display_shape
        if finger_height is None:
            finger_height = height
        if height != finger_height:
            raise ValueError(f"Inconsistent {side} finger height: {name}")
        bounds[name] = (cursor, 0, cursor + width, height)
        region_at[0:height, cursor:cursor + width] = name
        cursor += width

    palm = regions["palm"]
    palm_raw_shape = (len(palm["rows"]), len(palm["cols"]))
    palm_height, palm_width = (
        palm_raw_shape if side == "left" else palm_raw_shape[::-1])
    palm_top = int(finger_height or 0)
    bounds["palm"] = (0, palm_top, palm_width, palm_top + palm_height)
    region_at[palm_top:palm_top + palm_height, 0:palm_width] = "palm"

    if cursor != _GRID_COLS or palm_width != _GRID_COLS:
        raise ValueError(f"Unexpected {side} tactile width")
    if palm_top + palm_height != _GRID_ROWS or np.any(region_at == ""):
        raise ValueError(f"Unexpected {side} tactile height")
    return region_at, bounds


_REGION_LAYOUT = {side: _build_region_layout(side) for side in _SIDES}


def _rounded_card_mask(height: int, width: int, radius: int = 12) -> np.ndarray:
    """Return a uint8 rounded-rectangle mask."""

    height, width = int(height), int(width)
    radius = max(1, min(int(radius), height // 2, width // 2))
    mask = np.zeros((height, width), np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    for center in (
            (radius, radius), (width - radius - 1, radius),
            (radius, height - radius - 1),
            (width - radius - 1, height - radius - 1)):
        cv2.circle(mask, center, radius, 255, -1, cv2.LINE_AA)
    return mask


def _centered_text(
        image: np.ndarray,
        text: str,
        center_x: int,
        baseline_y: int,
        scale: float,
        color,
        thickness: int = 1) -> None:
    size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        image, text, (int(center_x - size[0] / 2), int(baseline_y)),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_region_outlines(
        card: np.ndarray,
        grid_x: int,
        grid_y: int,
        cell: int,
        bounds: dict[str, tuple[int, int, int, int]]) -> None:
    """Draw the exact HAND_REGIONS-derived finger and palm boundaries."""

    for name in (*_FINGER_NAMES, "palm"):
        col0, row0, col1, row1 = bounds[name]
        cv2.rectangle(
            card,
            (grid_x + col0 * cell, grid_y + row0 * cell),
            (grid_x + col1 * cell, grid_y + row1 * cell),
            _REGION_BGR[name], 2, cv2.LINE_AA)


_GLYPH_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GLYPH_TILE_CACHE: dict[tuple, tuple[np.ndarray, int, int]] = {}


def _digit_tile(text: str, scale: float, bg, fg, shadow):
    """Return a cached ``(tile, ox, oy)``: a small AA-rendered digit baked onto
    its solid cell background, plus the putText origin ``(ox, oy)`` inside the
    tile.  Baking the background lets the caller blit with a plain uint8 copy
    instead of a per-pixel alpha composite."""
    key = (text, round(float(scale), 4),
           tuple(int(c) for c in bg),
           tuple(int(c) for c in fg),
           tuple(int(c) for c in shadow))
    cached = _GLYPH_TILE_CACHE.get(key)
    if cached is not None:
        return cached
    (width, height), _ = cv2.getTextSize(text, _GLYPH_FONT, scale, 1)
    pad = 2
    ox, oy = pad, height + pad
    tile = np.empty((height + pad * 2, width + pad * 2, 3), np.uint8)
    tile[:] = np.asarray(bg, np.uint8)
    cv2.putText(tile, text, (ox + 1, oy + 1), _GLYPH_FONT, scale,
                shadow, 1, cv2.LINE_AA)
    cv2.putText(tile, text, (ox, oy), _GLYPH_FONT, scale, fg, 1, cv2.LINE_AA)
    sprite = (tile, ox, oy)
    _GLYPH_TILE_CACHE[key] = sprite
    return sprite


def _blit_tile(card, sprite, origin_x, origin_y):
    """Copy a cached digit tile onto ``card`` with its baseline-left at
    ``(origin_x, origin_y)`` (a plain uint8 block copy)."""
    tile, ox, oy = sprite
    x, y = origin_x - ox, origin_y - oy
    h, w = tile.shape[:2]
    card_h, card_w = card.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(card_w, x + w), min(card_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    card[y0:y1, x0:x1] = tile[y0 - y:y1 - y, x0 - x:x1 - x]


def _draw_cell_values(
        card: np.ndarray,
        displayed: np.ndarray,
        cell_colors: np.ndarray,
        grid_x: int,
        grid_y: int,
        cell: int) -> None:
    """Draw every mapped sensor value with size and contrast adaptation.

    Digits are baked once into small cached tiles and blitted as plain uint8
    copies, removing the ~780 anti-aliased ``putText`` calls per frame that
    dominated the budget when the matrix and the full-hand mesh were shown."""
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            label = str(int(round(float(displayed[row, col]))))
            scale = 0.45 if cell >= 18 else 0.36
            (width, height), _ = cv2.getTextSize(label, _GLYPH_FONT, scale, 1)
            available = max(5, cell - 2)
            if width > available:
                scale *= available / float(width)
                (width, height), _ = cv2.getTextSize(
                    label, _GLYPH_FONT, scale, 1)

            x0 = grid_x + col * cell
            y0 = grid_y + row * cell
            origin_x = x0 + max(1, (cell - width) // 2)
            origin_y = y0 + max(height + 1, (cell + height) // 2)
            bg = tuple(int(channel) for channel in cell_colors[row, col])
            blue, green, red = bg
            luminance = 0.114 * blue + 0.587 * green + 0.299 * red
            foreground = ((12, 18, 25) if luminance >= 150
                          else (244, 247, 252))
            shadow = ((238, 242, 248) if luminance >= 150
                      else (5, 10, 16))
            sprite = _digit_tile(label, scale, bg, foreground, shadow)
            _blit_tile(card, sprite, origin_x, origin_y)


def _render_card_template(side: str, cell: int) -> tuple:
    """Render and cache all visual elements that do not change per frame."""

    cache_key = (side, cell)
    cached = _TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    grid_w = _GRID_COLS * cell
    grid_h = _GRID_ROWS * cell
    card_w = _CARD_PAD_X + _AXIS_LEFT + grid_w + _CARD_PAD_X
    grid_y = (_CARD_HEADER + _CARD_PAD_TOP
              + _REGION_LABEL_HEIGHT + _X_TICK_HEIGHT)
    card_h = grid_y + grid_h + _CARD_FOOTER

    top = np.array((55, 34, 13), np.float32)
    bottom = np.array((30, 20, 9), np.float32)
    mix = np.linspace(0.0, 1.0, card_h, dtype=np.float32)[:, None]
    rows = (top[None, :] + (bottom - top)[None, :] * mix).astype(np.uint8)
    card = np.empty((card_h, card_w, 3), np.uint8)
    card[...] = rows[:, None, :]

    side_accent = (240, 201, 76) if side == "left" else (84, 180, 255)
    badge_center = (18, 17)
    cv2.circle(card, badge_center, 9, side_accent, -1, cv2.LINE_AA)
    cv2.putText(
        card, "L" if side == "left" else "R", (13, 21),
        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (15, 29, 48), 1, cv2.LINE_AA)
    cv2.putText(
        card, "LEFT PRESSURE MAP" if side == "left" else "RIGHT PRESSURE MAP",
        (34, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (255, 242, 234), 1, cv2.LINE_AA)

    cv2.line(card, (10, 33), (card_w - 11, 33), (88, 61, 43), 1, cv2.LINE_AA)
    cv2.line(card, (13, 1), (card_w - 14, 1), side_accent, 2, cv2.LINE_AA)

    grid_x = _CARD_PAD_X + _AXIS_LEFT
    region_at, bounds = _REGION_LAYOUT[side]

    # The left and right hand orders are horizontal mirrors of one another.
    ordered_fingers = _FINGER_ORDER[side]
    label_baseline = _CARD_HEADER + _CARD_PAD_TOP + 12
    for name in ordered_fingers:
        col0, _, col1, _ = bounds[name]
        center_x = grid_x + int(round((col0 + col1) * cell / 2.0))
        _centered_text(
            card, _FINGER_SHORT[name], center_x, label_baseline,
            0.31 if cell >= 16 else 0.25, _REGION_BGR[name])

    tick_baseline = grid_y - 5
    for col in range(_GRID_COLS):
        _centered_text(
            card, str(col + 1), grid_x + col * cell + cell // 2,
            tick_baseline, 0.25 if cell < 16 else 0.30, (180, 166, 151))

    # Draw only the 195 mapped cells; inactive raw-matrix rows/columns are not
    # part of this compact visual. Live colors are painted in one operation.
    grid_line = (105, 87, 69)
    base_fill = _PRESSURE_BANDS_CORRECTED[0][1]
    for display_row in range(_GRID_ROWS):
        y0 = grid_y + display_row * cell
        y1 = y0 + cell
        axis_value = 15 - display_row
        _centered_text(
            card, str(axis_value), grid_x - 13,
            y0 + cell // 2 + 4, 0.25 if cell < 16 else 0.30,
            (180, 166, 151))
        for display_col in range(_GRID_COLS):
            x0 = grid_x + display_col * cell
            x1 = x0 + cell
            cv2.rectangle(
                card, (x0, y0), (x1 - 1, y1 - 1), base_fill, -1)
            cv2.rectangle(
                card, (x0, y0), (x1, y1), grid_line, 1, cv2.LINE_AA)

    _draw_region_outlines(card, grid_x, grid_y, cell, bounds)

    footer_y = grid_y + grid_h + 15
    cv2.putText(
        card, "ACTIVE MAP 15 x 13  /  PALM 15 x 9",
        (grid_x, footer_y), cv2.FONT_HERSHEY_SIMPLEX,
        0.31 if cell >= 16 else 0.25, (199, 174, 156), 1, cv2.LINE_AA)

    mask = _rounded_card_mask(card_h, card_w)
    inner = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    card[mask > inner] = np.asarray((111, 81, 59), np.uint8)
    card[1:3, 13:card_w - 13] = side_accent

    # Leave grid/region borders untouched while replacing all valid interiors.
    cell_interior = np.zeros((cell, cell), dtype=bool)
    cell_interior[1:-1, 1:-1] = True
    paint_mask = np.kron(region_at != "", cell_interior)
    cached = (card, mask, paint_mask, grid_x, grid_y, bounds)
    _TEMPLATE_CACHE[cache_key] = cached
    return cached


def _render_card(
        frame: np.ndarray,
        side: str,
        scale: float,
        bands: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Render one compact 15x13 mapping card from sanitized pressure data."""

    shown_scale = min(max(float(scale), 0.1), 0.7)
    # At the application's default scale (0.6), 21 px keeps the per-cell numbers
    # readable without covering too much of the 3D hand view. The ] key can
    # still grow the cards further when needed.
    cell = max(11, int(round(21.0 * shown_scale / 0.6)))
    template, mask, paint_mask, grid_x, grid_y, bounds = (
        _render_card_template(side, cell))
    card = template.copy()

    values = np.maximum(0.0, np.asarray(frame, np.float32))
    displayed = _display_matrix(values, side)
    peak = max(0.0, float(values.max()))
    cell_colors = _band_colors(displayed, bands)
    expanded = np.repeat(np.repeat(cell_colors, cell, axis=0), cell, axis=1)
    grid_h = _GRID_ROWS * cell
    grid_w = _GRID_COLS * cell
    grid = card[grid_y:grid_y + grid_h, grid_x:grid_x + grid_w]
    grid[paint_mask] = expanded[paint_mask]

    # Repaint just six outlines so anti-aliased edges remain crisp at any heat.
    _draw_region_outlines(card, grid_x, grid_y, cell, bounds)
    _draw_cell_values(card, displayed, cell_colors, grid_x, grid_y, cell)
    if card.shape[1] >= 260:
        peak_text = f"MAX {peak:.0f}"
        (peak_w, _), _ = cv2.getTextSize(
            peak_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(
            card, peak_text, (card.shape[1] - peak_w - 11, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (199, 174, 156), 1,
            cv2.LINE_AA)
    return card, mask


def draw_bimanual_tactile_overlay(
        img: np.ndarray,
        tactile_frames: dict[str, np.ndarray | None],
        threshold: float = 0.0,
        scale: float = 0.6,
        baseline_corrected: bool = True) -> np.ndarray:
    """Draw compact active-region pressure cards on a 1280x720 BGR frame.

    ``baseline_corrected`` selects the fixed colour scale: the corrected
    (baseline-subtracted) bands when True, else the full raw-ADC bands.
    """

    bands = (
        _PRESSURE_BANDS_CORRECTED if baseline_corrected
        else _PRESSURE_BANDS_RAW)
    for side in _SIDES:
        value = tactile_frames.get(side)
        if value is None:
            continue
        frame = np.asarray(value, dtype=np.float32).reshape(16, 16)
        if not np.isfinite(frame).any():
            continue
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        if threshold > 0:
            frame = np.where(frame < float(threshold), 0.0, frame)

        cache_key = (side, frame.tobytes(), float(threshold), float(scale),
                     baseline_corrected)
        cached = _CARD_CACHE.get(cache_key)
        if cached is None:
            cached = _render_card(frame, side, scale, bands)
            if len(_CARD_CACHE) >= _CARD_CACHE_MAX:
                _CARD_CACHE.pop(next(iter(_CARD_CACHE)))
            _CARD_CACHE[cache_key] = cached
        card, mask = cached

        card_h, card_w = card.shape[:2]
        x0 = 12 if side == "left" else img.shape[1] - card_w - 12
        y0 = img.shape[0] - _CARD_BOTTOM_MARGIN - card_h
        roi = img[y0:y0 + card_h, x0:x0 + card_w]
        # The reference uses opaque navy panels. OpenCV's masked copy keeps the
        # rounded corners while avoiding two full-frame float conversions.
        cv2.copyTo(card, mask, roi)
    return img


__all__ = ["draw_bimanual_tactile_overlay"]
