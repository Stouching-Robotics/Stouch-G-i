"""Denoising pipeline for raw tactile frames, shared by the demo and hand GUIs.

This logic originally lived only in pc/tactile_hand_gui.py; the demo had its own
simplified version (a single-frame baseline plus one hard threshold), so the two
behaviors diverged. Extracted into one shared implementation so a change in one
place updates both, and they can no longer drift apart.

Pipeline order (line-by-line consistent with the hand implementation):

  1. Temporal smoothing  history = history*α + raw*(1-α)     removes frame-to-frame jitter
  2. Baseline subtraction  max(0, history - baseline_map)    per-cell zero level
  3. Drift removal       40th percentile > 100 → subtract 0.8x of it; keeps the noise floor down on global rises
  4. Noise gate          zero below max(base_gate, min(peak*ratio, 250))
  5. Spatial filter      zero isolated cells whose 8 neighbors are all 0 and whose value is below noise_floor

The baseline map is computed as [per-cell max over N frames + 10], which is more
conservative than the mean — the mean would fold occasional noise spikes into the signal.
"""

import numpy as np

MATRIX_ROWS = 16
MATRIX_COLS = 16

# Cap on the dynamic gate. When the peak is large, the gate must not rise unboundedly, otherwise light presses get "drowned out" by heavy ones.
DYNAMIC_GATE_CAP = 250.0
# Lower bound for isolated-cell detection: cells below this are noise; a strong isolated cell is a real signal
ISOLATED_NOISE_FLOOR = 180.0
ISOLATED_BASELINE_SCALE = 2.5


class TactilePreprocessor:
    """Process a raw (16,16) ADC frame into a usable pressure map.

    :param base_gate: fixed noise gate (in ADC counts after baseline subtraction)
    :param temporal_smooth: temporal smoothing factor α; larger is smoother but slower to react
    :param dynamic_noise_ratio: dynamic gate = current peak × this value, combined with base_gate via max()
    :param spatial_filter: whether to remove isolated cells
    :param calibration_frames: how many frames to sample during auto zeroing
    :param bypass_gates: when True, skip steps 4 and 5 (hand's --debug-view)
    """

    def __init__(self,
                 base_gate: float = 500.0,
                 temporal_smooth: float = 0.15,
                 dynamic_noise_ratio: float = 0.0,
                 spatial_filter: bool = True,
                 calibration_frames: int = 30,
                 bypass_gates: bool = False):
        self.base_gate = float(base_gate)
        self.temporal_smooth = float(temporal_smooth)
        self.dynamic_noise_ratio = float(dynamic_noise_ratio)
        self.spatial_filter = bool(spatial_filter)
        self.calibration_frames = int(calibration_frames)
        self.bypass_gates = bool(bypass_gates)

        self.baseline_map = np.zeros((MATRIX_ROWS, MATRIX_COLS), dtype=np.float32)
        self.history_buffer: np.ndarray | None = None
        self.drift_baseline_val = 0.0
        self.is_calibrating = True
        self._calibration_buffer: list[np.ndarray] = []

    # ---- Baseline ----

    def start_calibration(self, frames: int | None = None):
        """Restart zeroing. It ends automatically once frames frames have been collected; process() returns None meanwhile."""
        if frames is not None:
            self.calibration_frames = int(frames)
        self._calibration_buffer.clear()
        self.is_calibrating = True

    def set_baseline(self, baseline_map: np.ndarray):
        """Load a baseline map directly (e.g. restored from disk), skipping acquisition."""
        self.baseline_map = np.asarray(baseline_map, dtype=np.float32).reshape(
            MATRIX_ROWS, MATRIX_COLS).copy()
        self.is_calibrating = False
        self._calibration_buffer.clear()

    @property
    def calibration_progress(self) -> tuple[int, int]:
        return len(self._calibration_buffer), self.calibration_frames

    # ---- Main pipeline ----

    def process(self, raw_data: np.ndarray) -> tuple[np.ndarray | None, float]:
        """→ (processed, peak); returns (None, 0.0) while zeroing is in progress."""
        raw_data = np.asarray(raw_data, dtype=np.float32).reshape(MATRIX_ROWS, MATRIX_COLS)

        if self.is_calibrating:
            self._calibration_buffer.append(raw_data.copy())
            if len(self._calibration_buffer) >= self.calibration_frames:
                # Per-cell max + 10, more conservative than the mean; noise spikes are not counted as signal
                self.baseline_map = np.max(
                    np.stack(self._calibration_buffer), axis=0) + 10
                self.is_calibrating = False
                self._calibration_buffer.clear()
            return None, 0.0

        # 1. Temporal smoothing
        if self.history_buffer is None:
            self.history_buffer = raw_data.copy()
        self.history_buffer = (self.history_buffer * self.temporal_smooth
                               + raw_data * (1.0 - self.temporal_smooth))

        # 2. Baseline subtraction
        processed = np.maximum(0.0, self.history_buffer - self.baseline_map)

        # 3. Drift removal: when the whole frame rises together (temperature/power), the 40th percentile rises too
        self.drift_baseline_val = float(np.percentile(processed, 40))
        if self.drift_baseline_val > 100:
            processed = np.maximum(0.0, processed - self.drift_baseline_val * 0.8)

        if not self.bypass_gates:
            # 4. Noise gate
            dynamic_gate = min(processed.max() * self.dynamic_noise_ratio, DYNAMIC_GATE_CAP)
            final_gate = max(self.base_gate, dynamic_gate)
            processed[processed < final_gate] = 0.0

            # 5. Isolated-cell removal. All 8 neighbors being 0 means no adjacent cell is
            #    pressed at the same time; a real finger contact always covers more than one cell.
            if self.spatial_filter:
                mask = (processed > 0).astype(np.uint8)
                neighbors = _count_neighbors(mask)
                noise_floor = np.maximum(self.baseline_map * ISOLATED_BASELINE_SCALE,
                                         ISOLATED_NOISE_FLOOR)
                isolated = (mask > 0) & (neighbors == 0) & (processed < noise_floor)
                processed[isolated] = 0.0

        return processed, float(processed.max())


def _count_neighbors(mask: np.ndarray) -> np.ndarray:
    """Count 8-neighbors, padding the border with 0.

    Equivalent to cv2.filter2D(mask, -1, 8-neighbor kernel, BORDER_CONSTANT),
    but without adding an opencv dependency — the demo side (polyscope) doesn't need it.
    """
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.int32)
    padded[1:-1, 1:-1] = mask
    out = np.zeros_like(mask, dtype=np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out += padded[1 + dr:1 + dr + mask.shape[0], 1 + dc:1 + dc + mask.shape[1]]
    return out
