"""Per-region linear conversion from raw tactile ADC samples to finger force.

The five 2026-09 pressure-response reports define three regions per finger
using ``response_ADC = k * force_N + b``.  The glove's 12-bit raw ADC is
midscale-biased: its empty value is approximately 2048 and full-scale is about
4095, while the reports fit the 0..2047 response span.  This module first maps
raw ADC to that fixed response span, then applies the inverse relation without
depending on the UI's optional empty-glove baseline collection.
"""

from __future__ import annotations

import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

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

def fit_finger_force_matrix(adc_matrix: np.ndarray, side: str) -> np.ndarray:
    """Fit raw ADC samples to force; uncalibrated cells are returned as NaN.

    The input is always the original 0..4095 ADC matrix, irrespective of
    whether the viewer's optional baseline-subtracted pressure display is on.
    """

    adc = np.asarray(adc_matrix, dtype=np.float32)
    if adc.shape != (16, 16):
        raise ValueError(f"pressure matrix must be 16x16, got {adc.shape}")
    hand = str(side).lower()
    if hand not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    response_adc = np.clip(
        adc - RAW_ADC_ZERO, 0.0, RAW_ADC_RESPONSE_MAX)

    # Import lazily so the calibration layer follows the GUI's authoritative
    # routing table without creating an import cycle at module load time.
    from gui.rendering.tactile import HAND_REGIONS_LEFT, HAND_REGIONS_RIGHT

    regions = HAND_REGIONS_LEFT if hand == "left" else HAND_REGIONS_RIGHT
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
    "RAW_ADC_RESPONSE_MAX",
    "RAW_ADC_ZERO",
    "fit_finger_force_matrix",
]
