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
    # The first band is deliberately wide, and there is nothing below it.
    #
    # An unloaded glove does not sit at one value: measured on the real right
    # hand over 18 s it spans 2020-2239, because each sensor carries its own
    # offset.  Narrow 100-count steps at the bottom therefore spend their
    # whole range on *offsets* rather than on pressure -- the resting hand
    # came out 95.3% blue with the rest green and yellow, and those specks sat
    # right on the band edges and flickered as the device refreshed.  One
    # wide blue band covers the unloaded spread and the 2000 floor the viewer
    # clamps to, so a resting hand is one flat colour and only a real contact
    # changes it.  The corrected scale, which is the calibrated one, keeps its
    # fine steps.
    (2300, (199, 134, 22)),         # #1686c7 blue: unloaded, and the floor
    (2500, (94, 197, 34)),          # #22c55e green
    (2800, (21, 204, 250)),         # #facc15 yellow
    (3200, (22, 115, 249)),         # #f97316 orange
    (3600, (48, 90, 249)),          # orange-red
    (float("inf"), (68, 68, 239)),  # #ef4444 red, no upper limit
)
_PRESSURE_BANDS_FORCE_N = (
    (1.0, (95, 58, 30)),
    (2.0, (199, 134, 22)),
    (3.0, (94, 197, 34)),
    (4.0, (21, 204, 250)),
    (5.0, (22, 115, 249)),
    (float("inf"), (68, 68, 239)),
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
# Held to a few entries for the same reason as the card cache, and one more:
# dragging a card to size walks it through every cell on the way, so an
# uncapped cache would keep a full-size template for each step of the drag.
_TEMPLATE_CACHE: dict[tuple[str, int], tuple] = {}
_TEMPLATE_CACHE_MAX = 8


def _remember(cache: dict, limit: int, key, value) -> None:
    """Add one entry to a cache, dropping the oldest first when it is full.

    Every cache in this module is bounded and evicts in insertion order.  The
    limits are sized for the working set of a running viewer, not as a
    performance lever: they exist so that a long session in one of the modes
    whose key space is open-ended -- ``linear_matrix`` labels every cell with
    an unrounded force -- cannot grow a cache without limit.
    """

    if len(cache) >= limit:
        cache.pop(next(iter(cache)))
    cache[key] = value
_CARD_HEADER = 34
_CARD_PAD_X = 10
_CARD_PAD_TOP = 6
_AXIS_LEFT = 25
_REGION_LABEL_HEIGHT = 17
_X_TICK_HEIGHT = 14
_CARD_FOOTER = 21
_CARD_BOTTOM_MARGIN = 60

# Cell size at the viewer's default tactile scale (0.6).  The pixel constants
# above and the glyph sizes below are all tuned for a 21 px cell, so a card
# built at that size is unchanged; ``_ui_factor`` scales them for the larger
# cells the viewer reaches by growing the scale, keeping a grown card in
# proportion instead of leaving a small header and small digits in a big grid.
_CARD_CELL_AT_DEFAULT_SCALE = 28.0
_CARD_CELL_TUNED_FOR = 21.0
# Below this the per-cell numbers stop being legible, so it is a hard floor even
# when the frame is too small to honour it.
_CARD_CELL_MIN = 11
# Placement of the two cards: outer margin (``x0`` in the overlay), the gap kept
# between them, and the breathing room left above them.
_CARD_EDGE_MARGIN = 12
_CARD_CENTER_GAP = 24
_CARD_TOP_MARGIN = 8
# The drag handle painted beside each card's inner top corner: its size, the
# gap left to the card, and how far past the painted square the cursor may
# land and still grab it (an 18 px target is small on a scaled-down canvas).
# The handle sits outside the card because the inside of that corner holds the
# side badge or the peak reading, and every cell of the map is data.
_CARD_GRIP = 18
_CARD_GRIP_GAP = 4
_CARD_GRIP_SLOP = 7


def _ui_factor(cell: int) -> float:
    """Scale pixel constants and glyph sizes for a card built at ``cell``."""

    return max(1.0, float(cell) / _CARD_CELL_TUNED_FOR)


def _card_size(cell: int) -> tuple[int, int]:
    """Pixel size of a card built at ``cell``, mirroring the template's math.

    This must equal the shape of what ``_render_card`` returns for the same
    cell -- the resize handles are placed from this function alone, without
    rendering a card first, so a drift here would put the handle off the card.
    """

    ui = _ui_factor(cell)
    pad_x = int(round(_CARD_PAD_X * ui))
    width = 2 * pad_x + int(round(_AXIS_LEFT * ui)) + _GRID_COLS * cell
    height = (int(round(_CARD_HEADER * ui)) + int(round(_CARD_PAD_TOP * ui))
              + int(round(_REGION_LABEL_HEIGHT * ui))
              + int(round(_X_TICK_HEIGHT * ui))
              + _GRID_ROWS * cell + int(round(_CARD_FOOTER * ui)))
    return width, height


def _card_fits(cell: int, frame_shape) -> bool:
    """Whether the two cards both stay on frame at this cell size.

    The cards sit in the lower corners, so the limit is that they fit side by
    side with a gap between them -- and inside the frame's height.  Anything
    larger would push a card off screen or onto its twin.
    """

    height, width = frame_shape[:2]
    card_w, card_h = _card_size(cell)
    return (2 * card_w + 2 * _CARD_EDGE_MARGIN + _CARD_CENTER_GAP <= width
            and card_h + _CARD_BOTTOM_MARGIN + _CARD_TOP_MARGIN <= height)


def _cell_for_scale(scale: float) -> int:
    """The cell size a scale asks for, before the frame gets a say."""

    shown_scale = min(max(float(scale), 0.1), 2.0)
    return int(round(_CARD_CELL_AT_DEFAULT_SCALE * shown_scale / 0.6))


def _scale_for_cell(cell: int) -> float:
    """The scale that draws the cards at ``cell`` -- ``_cell_for_scale`` inverted."""

    return float(cell) * 0.6 / _CARD_CELL_AT_DEFAULT_SCALE


def _card_cell(scale: float, frame_shape) -> int:
    """Cell size in pixels for one card, clamped so both cards stay on frame."""

    requested = _cell_for_scale(scale)
    for cell in range(max(_CARD_CELL_MIN, requested), _CARD_CELL_MIN - 1, -1):
        if _card_fits(cell, frame_shape):
            return cell
    return _CARD_CELL_MIN


def tactile_cell_span(frame_shape) -> tuple[int, int]:
    """The smallest and largest cell size the cards can be drawn at here.

    Below the floor the per-cell numbers stop being legible, and above the
    ceiling a card no longer fits beside its twin or within the frame's height
    -- so on a 1280x720 canvas a card grows only about a quarter larger than
    the default before it is against the frame, whatever size is asked for.
    """

    height, width = frame_shape[:2]
    ceiling = min(width // _GRID_COLS, height // _GRID_ROWS)
    for cell in range(max(_CARD_CELL_MIN, ceiling), _CARD_CELL_MIN - 1, -1):
        if _card_fits(cell, frame_shape):
            return _CARD_CELL_MIN, cell
    return _CARD_CELL_MIN, _CARD_CELL_MIN


def tactile_cell_for_scale(scale: float, frame_shape) -> int:
    """The cell size the cards are drawn at for ``scale`` on this frame."""

    return _card_cell(scale, frame_shape)


def tactile_scale_for_cell(cell: int, frame_shape) -> float:
    """The ``scale`` that draws the cards at ``cell``, held to what fits.

    ``scale`` is the only size the rest of the viewer carries, so a resize
    that wants a particular cell size has to express it as one.
    """

    smallest, largest = tactile_cell_span(frame_shape)
    return _scale_for_cell(min(max(int(cell), smallest), largest))


def _place_card(
        frame_shape, side: str, card_w: int, card_h: int
) -> tuple[int, int, int, int]:
    """Where a card of this size lands: ``(x0, y0, x1, y1)`` in frame pixels.

    The two cards are anchored to the bottom corners, so the left one grows
    to the right and the right one to the left when the card gets bigger.
    """

    height, width = frame_shape[:2]
    x0 = (_CARD_EDGE_MARGIN if side == "left"
          else width - card_w - _CARD_EDGE_MARGIN)
    y0 = height - _CARD_BOTTOM_MARGIN - card_h
    return x0, y0, x0 + card_w, y0 + card_h


def _grip_rect(card_rect, side: str) -> tuple[int, int, int, int]:
    """The resize handle's box, beside a card's inner top corner.

    "Inner" is the corner facing the middle of the frame -- the left card's
    top-right, the right card's top-left -- because that is the edge the card
    moves when it is resized.  The handle sits just clear of the card rather
    than on it: the inside of that corner carries the side badge or the peak
    reading, and the rest of the header is labels.
    """

    x0, y0, x1, _ = card_rect
    left = (x1 + _CARD_GRIP_GAP if side == "left"
            else x0 - _CARD_GRIP_GAP - _CARD_GRIP)
    return left, y0, left + _CARD_GRIP, y0 + _CARD_GRIP


def _drawable_sides(tactile_frames) -> tuple[str, ...]:
    """The sides the overlay will actually paint a card for."""

    drawable = []
    for side in _SIDES:
        value = tactile_frames.get(side)
        if value is None:
            continue
        frame = np.asarray(value, dtype=np.float32).reshape(16, 16)
        if np.isfinite(frame).any():
            drawable.append(side)
    return tuple(drawable)


def tactile_card_rects(
        frame_shape, tactile_frames, scale: float
) -> dict[str, tuple[int, int, int, int]]:
    """The box each card occupies right now, keyed by side.

    Computed from the same size and placement math the renderer uses, so a
    caller can hit-test without drawing a frame first.  A side with no card
    drawn gets no entry: there is nothing on screen there.
    """

    cell = _card_cell(scale, frame_shape)
    card_w, card_h = _card_size(cell)
    return {
        side: _place_card(frame_shape, side, card_w, card_h)
        for side in _drawable_sides(tactile_frames)
    }


def tactile_grip_rects(
        frame_shape, tactile_frames, scale: float
) -> dict[str, tuple[int, int, int, int]]:
    """The resize handles on screen right now, keyed by side.

    These are the boxes ``draw_bimanual_tactile_overlay`` paints, widened by
    ``_CARD_GRIP_SLOP`` so the cursor does not have to land on the last pixel.
    """

    # Rect coordinates are inclusive at both ends, so the left handle's reach
    # stops one pixel short of the middle and the right one begins on it.
    midpoint = frame_shape[1] // 2
    handles = {}
    for side, card_rect in tactile_card_rects(
            frame_shape, tactile_frames, scale).items():
        x0, y0, x1, y1 = _grip_rect(card_rect, side)
        hit = [x0 - _CARD_GRIP_SLOP, y0 - _CARD_GRIP_SLOP,
               x1 + _CARD_GRIP_SLOP, y1 + _CARD_GRIP_SLOP]
        # The hit box may reach past the painted square, but not across the
        # middle of the frame: at the largest card size the two would
        # otherwise overlap, and a press has to land on one handle, not two.
        if side == "left":
            hit[2] = min(hit[2], midpoint - 1)
        else:
            hit[0] = max(hit[0], midpoint)
        handles[side] = tuple(hit)
    return handles


def _draw_resize_grip(img: np.ndarray, side: str, card_rect) -> None:
    """Paint the drag handle into a card's inner top corner.

    A double-headed diagonal, aimed the way the corner actually travels: the
    left card's inner corner moves right and up, the right card's moves left
    and up.
    """

    x0, y0, x1, y1 = _grip_rect(card_rect, side)
    accent = (240, 201, 76) if side == "left" else (84, 180, 255)
    cv2.rectangle(img, (x0, y0), (x1, y1), (38, 31, 26), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), accent, 1, cv2.LINE_AA)
    if side == "left":
        near, far = (x0 + 5, y1 - 5), (x1 - 5, y0 + 5)
    else:
        near, far = (x1 - 5, y1 - 5), (x0 + 5, y0 + 5)
    middle = ((near[0] + far[0]) // 2, (near[1] + far[1]) // 2)
    cv2.arrowedLine(img, middle, far, accent, 1, cv2.LINE_AA, tipLength=0.4)
    cv2.arrowedLine(img, middle, near, accent, 1, cv2.LINE_AA, tipLength=0.4)


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
# Bounded for the same reason as the cards and templates, and with the same
# kind of margin: a session's working set is a few hundred numbers per hand, so
# these limits are a guard rail rather than something a viewer hits.
_GLYPH_TILE_CACHE_MAX = 8192
_TEXT_METRICS: dict[tuple[str, int], tuple[float, int, int]] = {}
_TEXT_METRICS_MAX = 4096


def _text_metrics(
        label: str, cell: int, scale: float) -> tuple[float, int, int]:
    """The scale and pixel size ``label`` is drawn at inside a ``cell`` box.

    Measured once per ``(label, cell)`` instead of once per cell per frame.
    The auto-shrink below needs a ``cv2.getTextSize`` before it can decide
    whether to shrink at all, and a second one to measure the result, so a
    card whose cells all carry numbers spent roughly 390 of those calls per
    frame -- on one hand 4- and 5-digit readings in a 28 px cell always take
    the shrink path.  The labels repeat heavily: an unloaded glove shows its
    own offset plus a little noise, so a hand's 195 cells carry only a few
    hundred distinct numbers between them.
    """

    key = (label, cell)
    cached = _TEXT_METRICS.get(key)
    if cached is not None:
        return cached
    available = max(5, cell - 2)
    (width, height), _ = cv2.getTextSize(label, _GLYPH_FONT, scale, 1)
    if width > available:
        scale *= available / float(width)
        (width, height), _ = cv2.getTextSize(label, _GLYPH_FONT, scale, 1)
    measured = (scale, width, height)
    _remember(_TEXT_METRICS, _TEXT_METRICS_MAX, key, measured)
    return measured


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
    _remember(_GLYPH_TILE_CACHE, _GLYPH_TILE_CACHE_MAX, key, sprite)
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
        cell: int,
        linear_fitted: bool = False) -> None:
    """Draw every mapped sensor value with size and contrast adaptation.

    Digits are baked once into small cached tiles and blitted as plain uint8
    copies, removing the ~780 anti-aliased ``putText`` calls per frame that
    dominated the budget when the matrix and the full-hand mesh were shown.
    What is left per cell is a cached metric lookup, one glyph-tile lookup and
    one block copy, with the label's size and the background's contrast
    answered from a table the same way the glyph itself is."""

    base_scale = (0.45 if cell >= 18 else 0.36) * _ui_factor(cell)
    # Glyph colours depend only on the band colour, and there are six of those
    # per scale, so the contrast decision is made once per colour rather than
    # once per cell.
    contrast: dict[tuple[int, int, int], tuple] = {}
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            if linear_fitted:
                label = (f"{float(displayed[row, col]):.2f}"
                         if row < 4 else "--")
            else:
                label = str(int(round(float(displayed[row, col]))))
            scale, width, height = _text_metrics(label, cell, base_scale)

            x0 = grid_x + col * cell
            y0 = grid_y + row * cell
            origin_x = x0 + max(1, (cell - width) // 2)
            origin_y = y0 + max(height + 1, (cell + height) // 2)
            bg = (int(cell_colors[row, col, 0]), int(cell_colors[row, col, 1]),
                  int(cell_colors[row, col, 2]))
            glyph_colors = contrast.get(bg)
            if glyph_colors is None:
                blue, green, red = bg
                light = 0.114 * blue + 0.587 * green + 0.299 * red >= 150
                glyph_colors = (
                    ((12, 18, 25) if light else (244, 247, 252)),
                    ((238, 242, 248) if light else (5, 10, 16)),
                )
                contrast[bg] = glyph_colors
            foreground, shadow = glyph_colors
            sprite = _digit_tile(label, scale, bg, foreground, shadow)
            _blit_tile(card, sprite, origin_x, origin_y)


def _render_card_template(side: str, cell: int) -> tuple:
    """Render and cache all visual elements that do not change per frame."""

    cache_key = (side, cell)
    cached = _TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ui = _ui_factor(cell)
    pad_x = int(round(_CARD_PAD_X * ui))
    pad_top = int(round(_CARD_PAD_TOP * ui))
    axis_left = int(round(_AXIS_LEFT * ui))
    header_h = int(round(_CARD_HEADER * ui))
    region_label_h = int(round(_REGION_LABEL_HEIGHT * ui))
    x_tick_h = int(round(_X_TICK_HEIGHT * ui))
    footer_h = int(round(_CARD_FOOTER * ui))
    small = (0.25 if cell < 16 else 0.30) * ui

    grid_w = _GRID_COLS * cell
    grid_h = _GRID_ROWS * cell
    card_w = pad_x + axis_left + grid_w + pad_x
    grid_y = header_h + pad_top + region_label_h + x_tick_h
    card_h = grid_y + grid_h + footer_h

    top = np.array((55, 34, 13), np.float32)
    bottom = np.array((30, 20, 9), np.float32)
    mix = np.linspace(0.0, 1.0, card_h, dtype=np.float32)[:, None]
    rows = (top[None, :] + (bottom - top)[None, :] * mix).astype(np.uint8)
    card = np.empty((card_h, card_w, 3), np.uint8)
    card[...] = rows[:, None, :]

    side_accent = (240, 201, 76) if side == "left" else (84, 180, 255)
    badge_center = (int(round(18 * ui)), int(round(17 * ui)))
    cv2.circle(card, badge_center, int(round(9 * ui)), side_accent, -1,
               cv2.LINE_AA)
    cv2.putText(
        card, "L" if side == "left" else "R",
        (int(round(13 * ui)), int(round(21 * ui))),
        cv2.FONT_HERSHEY_SIMPLEX, 0.39 * ui, (15, 29, 48), 1, cv2.LINE_AA)
    cv2.putText(
        card, "LEFT PRESSURE MAP" if side == "left" else "RIGHT PRESSURE MAP",
        (int(round(34 * ui)), int(round(23 * ui))),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43 * ui,
        (255, 242, 234), 1, cv2.LINE_AA)

    cv2.line(card, (pad_x, header_h - 1), (card_w - pad_x - 1, header_h - 1),
             (88, 61, 43), 1, cv2.LINE_AA)
    cv2.line(card, (13, 1), (card_w - 14, 1), side_accent, 2, cv2.LINE_AA)

    grid_x = pad_x + axis_left
    region_at, bounds = _REGION_LAYOUT[side]

    # The left and right hand orders are horizontal mirrors of one another.
    ordered_fingers = _FINGER_ORDER[side]
    label_baseline = header_h + pad_top + int(round(12 * ui))
    for name in ordered_fingers:
        col0, _, col1, _ = bounds[name]
        center_x = grid_x + int(round((col0 + col1) * cell / 2.0))
        _centered_text(
            card, _FINGER_SHORT[name], center_x, label_baseline,
            (0.31 if cell >= 16 else 0.25) * ui, _REGION_BGR[name])

    tick_baseline = grid_y - int(round(5 * ui))
    for col in range(_GRID_COLS):
        _centered_text(
            card, str(col + 1), grid_x + col * cell + cell // 2,
            tick_baseline, small, (180, 166, 151))

    # Draw only the 195 mapped cells; inactive raw-matrix rows/columns are not
    # part of this compact visual. Live colors are painted in one operation.
    grid_line = (105, 87, 69)
    base_fill = _PRESSURE_BANDS_CORRECTED[0][1]
    for display_row in range(_GRID_ROWS):
        y0 = grid_y + display_row * cell
        y1 = y0 + cell
        axis_value = 15 - display_row
        _centered_text(
            card, str(axis_value), grid_x - int(round(13 * ui)),
            y0 + cell // 2 + 4, small, (180, 166, 151))
        for display_col in range(_GRID_COLS):
            x0 = grid_x + display_col * cell
            x1 = x0 + cell
            cv2.rectangle(
                card, (x0, y0), (x1 - 1, y1 - 1), base_fill, -1)
            cv2.rectangle(
                card, (x0, y0), (x1, y1), grid_line, 1, cv2.LINE_AA)

    _draw_region_outlines(card, grid_x, grid_y, cell, bounds)

    footer_y = grid_y + grid_h + int(round(15 * ui))
    cv2.putText(
        card, "ACTIVE MAP 15 x 13  /  PALM 15 x 9",
        (grid_x, footer_y), cv2.FONT_HERSHEY_SIMPLEX,
        (0.31 if cell >= 16 else 0.25) * ui, (199, 174, 156), 1, cv2.LINE_AA)

    mask = _rounded_card_mask(card_h, card_w)
    inner = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    card[mask > inner] = np.asarray((111, 81, 59), np.uint8)
    card[1:3, 13:card_w - 13] = side_accent

    # Leave grid/region borders untouched while replacing all valid interiors.
    cell_interior = np.zeros((cell, cell), dtype=bool)
    cell_interior[1:-1, 1:-1] = True
    paint_mask = np.kron(region_at != "", cell_interior)
    cached = (card, mask, paint_mask, grid_x, grid_y, bounds)
    _remember(_TEMPLATE_CACHE, _TEMPLATE_CACHE_MAX, cache_key, cached)
    return cached


def _paint_cell_colors(
        grid: np.ndarray, cell_colors: np.ndarray, cell: int,
        paint_mask: np.ndarray) -> None:
    """Fill every mapped cell's interior with its band colour.

    One slice write per mapped cell, straight from the cell-resolution colour
    array.  The obvious vectorised forms are both slower, measured on a real
    479x487 card: expanding the colours to full resolution and writing them
    through the mask with ``np.copyto(..., where=)`` costs 1.20 ms, and the
    3-D boolean fancy index (``grid[mask] = expanded[mask]``) costs 4.67 ms,
    a quarter of a 60 Hz frame for one hand and, before this, half the cost of
    the whole re-render.  Both spend their time materialising a 258 KB colour
    block and then walking the whole 152 KB grid to find the 132 KB that the
    mask allows; the slices touch exactly those bytes and nothing else, and
    come to 0.60 ms.

    ``paint_mask`` is still what decides which cells are mapped -- it is the
    template's own map, and the probe below reads the cell's corner, which
    ``cell_interior`` always covers for a cell that is part of a region.  The
    two are checked against each other in ``tmp/verify_tactile_paint.py``.
    """

    rows, cols = cell_colors.shape[:2]
    inner = cell - 2
    for row in range(rows):
        y0 = row * cell + 1
        for col in range(cols):
            if not paint_mask[y0, col * cell + 1]:
                continue
            x0 = col * cell + 1
            grid[y0:y0 + inner, x0:x0 + inner] = cell_colors[row, col]


def _render_card(
        frame: np.ndarray,
        side: str,
        cell: int,
        bands: tuple,
        linear_fitted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Render one compact 15x13 mapping card from sanitized pressure data."""

    template, mask, paint_mask, grid_x, grid_y, bounds = (
        _render_card_template(side, cell))
    card = template.copy()

    values = np.maximum(0.0, np.asarray(frame, np.float32))
    displayed = _display_matrix(values, side)
    peak = max(0.0, float(values.max()))
    cell_colors = _band_colors(displayed, bands)
    grid_h = _GRID_ROWS * cell
    grid_w = _GRID_COLS * cell
    grid = card[grid_y:grid_y + grid_h, grid_x:grid_x + grid_w]
    _paint_cell_colors(grid, cell_colors, cell, paint_mask)

    # Repaint just six outlines so anti-aliased edges remain crisp at any heat.
    _draw_region_outlines(card, grid_x, grid_y, cell, bounds)
    _draw_cell_values(
        card, displayed, cell_colors, grid_x, grid_y, cell,
        linear_fitted=linear_fitted)
    if card.shape[1] >= 260:
        ui = _ui_factor(cell)
        peak_text = (f"MAX {peak:.2f} N" if linear_fitted
                     else f"MAX {peak:.0f}")
        peak_scale = 0.36 * ui
        (peak_w, _), _ = cv2.getTextSize(
            peak_text, cv2.FONT_HERSHEY_SIMPLEX, peak_scale, 1)
        cv2.putText(
            card, peak_text,
            (card.shape[1] - peak_w - int(round(11 * ui)),
             int(round(22 * ui))),
            cv2.FONT_HERSHEY_SIMPLEX, peak_scale, (199, 174, 156), 1,
            cv2.LINE_AA)
    return card, mask


def draw_bimanual_tactile_overlay(
        img: np.ndarray,
        tactile_frames: dict[str, np.ndarray | None],
        threshold: float = 0.0,
        scale: float = 0.6,
        baseline_corrected: bool | dict[str, bool] = True,
        linear_fitted: bool = False) -> np.ndarray:
    """Draw compact active-region pressure cards on a 1280x720 BGR frame.

    ``linear_fitted`` says the frame already carries force in newtons -- which
    is what the public SDK hands over -- so there is nothing left to fit and
    the newton bands are used.  NaN cells (positions no fitted region covers)
    read as zero force.  Otherwise ``baseline_corrected`` selects the fixed
    colour scale for legacy ADC recordings: the corrected (baseline-subtracted)
    bands when True, else the full raw-ADC bands.
    """

    def bands_for(side: str) -> list:
        corrected = (
            bool(baseline_corrected.get(side, False))
            if isinstance(baseline_corrected, dict)
            else bool(baseline_corrected)
        )
        if linear_fitted:
            return _PRESSURE_BANDS_FORCE_N
        return _PRESSURE_BANDS_CORRECTED if corrected else _PRESSURE_BANDS_RAW

    for side in _drawable_sides(tactile_frames):
        bands = bands_for(side)
        corrected = (
            bool(baseline_corrected.get(side, False))
            if isinstance(baseline_corrected, dict)
            else bool(baseline_corrected)
        )
        frame = np.asarray(
            tactile_frames[side], dtype=np.float32).reshape(16, 16)
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        if threshold > 0:
            frame = np.where(frame < float(threshold), 0.0, frame)

        # The cell size depends on the frame as well as on the scale, so the
        # frame shape joins the cache key: the same scale on a smaller canvas
        # would otherwise reuse a card too wide for it.
        cell = _card_cell(scale, img.shape)
        cache_key = (side, frame.tobytes(), float(threshold), cell,
                     corrected, linear_fitted)
        cached = _CARD_CACHE.get(cache_key)
        if cached is None:
            cached = _render_card(
                frame, side, cell, bands, linear_fitted=linear_fitted)
            _remember(_CARD_CACHE, _CARD_CACHE_MAX, cache_key, cached)
        card, mask = cached

        card_h, card_w = card.shape[:2]
        card_rect = _place_card(img.shape, side, card_w, card_h)
        x0, y0, _, _ = card_rect
        roi = img[y0:y0 + card_h, x0:x0 + card_w]
        # The reference uses opaque navy panels. OpenCV's masked copy keeps the
        # rounded corners while avoiding two full-frame float conversions.
        cv2.copyTo(card, mask, roi)
        _draw_resize_grip(img, side, card_rect)
    return img


__all__ = [
    "draw_bimanual_tactile_overlay", "tactile_card_rects",
    "tactile_cell_for_scale", "tactile_cell_span", "tactile_grip_rects",
    "tactile_scale_for_cell",
]
