"""Per-region linear conversion from raw tactile ADC samples to finger force.

The five 2026-09 pressure-response reports define three regions per finger
using ``response_ADC = k * force_N + b``.  The glove's 12-bit raw ADC is
midscale-biased: its empty value is approximately 2048 and full-scale is about
4095, while the reports fit the 0..2047 response span.  This module first maps
raw ADC to that fixed response span, then applies the inverse relation without
depending on the UI's optional empty-glove baseline collection.

The conversion table below is the asset this module exists to protect: it is
listed in the protection manifest's ``encrypt_python`` allowlist, so it ships
PyArmor-obfuscated in **both** editions.  The user ("client") edition builds its
newton output from here and never hands the raw ADC to the caller, which is why
the fitter lives inside the boundary rather than in :mod:`sdk`.

The region routing table is physically identical to
``gui.rendering.tactile.HAND_REGIONS_LEFT/RIGHT``; if the physical wiring ever
changes, update both places.  ``sdk/force.py`` carries a third plaintext copy
for the admin edition.  ``tools/self_test.py`` asserts all three agree
numerically -- that assertion is the only thing standing between a wiring
change and silently wrong newtons.
"""

from __future__ import annotations

import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
MATRIX_ROWS = 16
MATRIX_COLS = 16

# Firmware ADC scale: 0 pressure is at the 12-bit midpoint, and the response
# used by the PDF reports occupies the positive half of the range.  This is a
# fixed wire/data-format conversion, not the per-session baseline feature.
RAW_ADC_ZERO = 2048.0
RAW_ADC_RESPONSE_MAX = 2047.0

# Report order: thumb, index, middle, ring, pinky; three tested regions each.
LINEAR_K = np.asarray([
    (312.9, 313.1, 317.7),
    (723.4, 726.2, 711.9),
    (725.2, 724.2, 708.5),
    (429.5, 431.6, 444.2),
    (258.9, 259.6, 254.7),
], dtype=np.float32)
LINEAR_B = np.asarray([
    (511.4, 505.8, 480.0),
    (463.7, 461.3, 475.5),
    (410.3, 410.3, 430.2),
    (432.6, 447.6, 442.3),
    (389.8, 373.0, 386.2),
], dtype=np.float32)

# Routing table: which 16x16 matrix cells belong to each finger/palm region.
# Right hand: fingers form a vertical band in columns 12-15 stacked along rows
# (thumb 1-3 / index 4-6 / middle 7-9 / ring 10-12 / pinky 13-15), palm is
# rows 1-15 x cols 3-11.  Left hand: fingers form a horizontal band in rows
# 0-3 (pinky 0-2 / ring 3-5 / middle 6-8 / index 9-11 / thumb 12-14), palm is
# rows 4-12 x cols 0-14.  Mirrors gui.rendering.tactile exactly.
PRESSURE_REGIONS_RIGHT = {
    "thumb": {"name": "Thumb", "rows": [1, 2, 3], "cols": [15, 14, 13, 12]},
    "index": {"name": "Index", "rows": [4, 5, 6], "cols": [15, 14, 13, 12]},
    "middle": {"name": "Middle", "rows": [7, 8, 9], "cols": [15, 14, 13, 12]},
    "ring": {"name": "Ring", "rows": [10, 11, 12], "cols": [15, 14, 13, 12]},
    "pinky": {"name": "Pinky", "rows": [13, 14, 15], "cols": [15, 14, 13, 12]},
    "palm": {"name": "Palm", "rows": list(range(1, 16)), "cols": list(range(11, 2, -1))},
}
PRESSURE_REGIONS_LEFT = {
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


def _force_from_response(response_adc: np.ndarray, k: float, b: float) -> np.ndarray:
    """Apply one stored ``ADC = k * force + b`` pair (normalized scale)."""

    slope = float(k) * RAW_ADC_RESPONSE_MAX
    intercept = float(b) * RAW_ADC_RESPONSE_MAX
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0.0:
        return np.full_like(response_adc, np.nan, dtype=np.float32)
    return np.maximum(0.0, (response_adc - intercept) / slope)


def _compact_pair_sets(calibration: dict) -> dict[int, list[tuple[float, float]]]:
    """Extract the plain-data compact block returned by the firmware reader."""

    result: dict[int, list[tuple[float, float]]] = {}
    for item in calibration.get("sets") or []:
        try:
            set_index = int(item["set_index"])
            regions = [
                (float(region["k"]), float(region["b"]))
                for region in item.get("regions") or []
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= set_index < 6 and regions:
            result[set_index] = regions
    return result


def _fit_from_compact_block(
        adc: np.ndarray, side: str, calibration: dict) -> np.ndarray:
    """Convert a compact six-set firmware block into a 16x16 force matrix."""

    regions = PRESSURE_REGIONS_LEFT if side == "left" else PRESSURE_REGIONS_RIGHT
    pairs = _compact_pair_sets(calibration)
    force = np.full((MATRIX_ROWS, MATRIX_COLS), np.nan, dtype=np.float32)

    def apply_region(rows, columns, pair) -> None:
        row_index = np.asarray(list(rows), dtype=np.intp)
        column_index = np.asarray(list(columns), dtype=np.intp)
        if row_index.size == 0 or column_index.size == 0:
            return
        values = adc[np.ix_(row_index, column_index)]
        force[np.ix_(row_index, column_index)] = _force_from_response(
            values, pair[0], pair[1])

    def apply_set(set_index: int, name: str) -> None:
        set_pairs = pairs.get(set_index)
        if not set_pairs:
            return
        config = regions[name]
        rows = list(config["rows"])
        columns = list(config["cols"])
        if len(set_pairs) == 1:
            apply_region(rows, columns, set_pairs[0])
        elif len(set_pairs) == 3:
            if name == "palm":
                row_groups = np.array_split(np.asarray(rows, dtype=np.intp), 3)
                for group, pair in zip(row_groups, set_pairs):
                    apply_region(group.tolist(), columns, pair)
            elif len(rows) == 3:
                for row, pair in zip(rows, set_pairs):
                    apply_region([row], columns, pair)
            elif len(columns) == 3:
                for column, pair in zip(columns, set_pairs):
                    apply_region(rows, [column], pair)

    for set_index, name in enumerate(
            ("thumb", "index", "middle", "ring", "pinky", "palm")):
        apply_set(set_index, name)
    return force


def fit_finger_force_matrix(
        adc_matrix: np.ndarray, side: str,
        calibration: dict | None = None) -> np.ndarray:
    """Fit raw ADC samples to force (newtons); uncalibrated cells are NaN.

    ``adc_matrix`` is the original 0..4095 pressure matrix, irrespective of
    whether a viewer's optional baseline-subtracted pressure display is on.

    ``side`` is ``"left"`` or ``"right"``.

    When ``calibration`` is supplied it must be the compact decoded block
    returned by the calibration tooling's compact-block reader (a dict with a
    ``"sets"`` list of ``{"set_index", "regions": [{"k", "b"}, ...]}``).  The
    stored k/b pairs then replace the built-in report constants; they are
    fitted against the raw ADC column, so the 2048 zero level is not subtracted
    again.
    """

    adc = np.asarray(adc_matrix, dtype=np.float32)
    if adc.shape != (16, 16):
        raise ValueError(f"pressure matrix must be 16x16, got {adc.shape}")
    hand = str(side).lower()
    if hand not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if calibration is not None:
        return _fit_from_compact_block(adc, hand, calibration)

    response_adc = np.clip(
        adc - RAW_ADC_ZERO, 0.0, RAW_ADC_RESPONSE_MAX)

    regions = PRESSURE_REGIONS_LEFT if hand == "left" else PRESSURE_REGIONS_RIGHT
    force = np.full((16, 16), np.nan, dtype=np.float32)
    for finger, name in enumerate(FINGER_NAMES):
        config = regions[name]
        rows = list(config["rows"])
        columns = list(config["cols"])
        if len(rows) == 3:
            for region, row in enumerate(rows):
                force[row, columns] = np.maximum(
                    0.0,
                    (response_adc[row, columns] - LINEAR_B[finger, region])
                    / LINEAR_K[finger, region],
                )
        elif len(columns) == 3:
            for region, column in enumerate(columns):
                force[rows, column] = np.maximum(
                    0.0,
                    (response_adc[rows, column] - LINEAR_B[finger, region])
                    / LINEAR_K[finger, region],
                )
        else:
            raise ValueError(
                f"{hand} {name} must expose exactly three fitted regions")
    return force


__all__ = [
    "FINGER_NAMES",
    "LINEAR_B",
    "LINEAR_K",
    "PRESSURE_REGIONS_LEFT",
    "PRESSURE_REGIONS_RIGHT",
    "RAW_ADC_RESPONSE_MAX",
    "RAW_ADC_ZERO",
    "fit_finger_force_matrix",
]
