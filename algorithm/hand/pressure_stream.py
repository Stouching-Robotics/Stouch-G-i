"""Tactile stream view fed by the glove's shared STM32 USB receiver."""

from __future__ import annotations

import numpy as np

from common.ring_buffer import RingBuffer


class UsbPressureSensorStream:
    """Pressure-sensor interface without a second serial-port owner."""

    stream_id = "pressure_sensor.usb"

    def __init__(self, fps: int = 80):
        self._buffer = RingBuffer(size=1024, fps=int(fps))

    @property
    def buffer(self) -> RingBuffer:
        return self._buffer

    @property
    def fps(self) -> int:
        return self._buffer.fps

    @property
    def latest_frame(self) -> np.ndarray | None:
        frame, _index, _error = self._buffer.peek(-1)
        return frame

    def start(self):
        """The hand USB receiver already owns and starts the serial stream."""

    def stop(self):
        """The hand state machine owns and stops the serial stream."""
