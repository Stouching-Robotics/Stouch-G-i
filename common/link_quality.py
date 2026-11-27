"""Sequence-gap loss accounting for one device link.

The firmware numbers every frame it emits from a single counter shared by all
frame types -- IMU, pressure matrix, status, magnetometer telemetry -- so the
gap between two frames the host *received* is exactly the number of frames the
device sent in between and the host never got.  Reading that gap is the only way
to tell a link that is losing frames from a host that is discarding them: the
first is the glove or the radio, the second is this program, and the two have
opposite fixes.  The v2.0.6 interface document asks for exactly this reading.

The one thing that looks like loss and is not is the flush at open.  A dongle
buffers while nothing is reading it, so the first read can hand over hundreds of
frames at once: a sequence gap in the hundreds whose frames were merely late,
not lost.  A gap is therefore only charged when the wall clock says the frames
could not have arrived in time -- a real stream cannot exceed
:data:`DROP_FREE_FPS`, so a gap wider than that rate allows over the observed
arrival spacing means those frames were never delivered.  The rule is the one
the standalone Bluetooth monitors (``hands_imu.py`` / ``hands_ma.py``) use, so
the two agree frame for frame on the same hardware.

Lives in :mod:`common` for the same reason as :mod:`common.frame_pacing`: the
GUI and the SDK's offline self-test both use it, and the self-test must not drag
in the GUI's rendering dependencies.
"""

from __future__ import annotations

from common.usb_cdc import (
    USB_FULL_SEQUENCE_FROM,
    USB_SLIM_HEADER_SIZE,
    firmware_real_version,
)

# A gap cannot be a real loss when the frames that *did* arrive are closer
# together than this.  The device runs at 80 Hz at most, so an ordinary frame
# spacing is 12.5 ms and always clears the bar, while a dongle flushing its
# backlog hands over many frames within microseconds and never does.
DROP_FREE_FPS = 400.0

# The counter's width follows the frame's own header, and for the slim header it
# follows the version too: through v1.2.12 bit 15 is the magnetometer-not-ready
# flag and only 15 bits are counter.  The width has to match, because masking a
# wider sequence down invents a gap of the full range at every wrap.
FULL_SEQUENCE_MASK = 0xFFFFFFFF
SLIM_SEQUENCE_MASK = 0xFFFF
LEGACY_SLIM_SEQUENCE_MASK = 0x7FFF


def sequence_mask(header_size: int, version: str | None) -> int:
    """Return the sequence-counter mask for a frame's layout and version."""

    if int(header_size) != USB_SLIM_HEADER_SIZE:
        return FULL_SEQUENCE_MASK
    real = firmware_real_version(version)
    if real is not None and real < USB_FULL_SEQUENCE_FROM:
        return LEGACY_SLIM_SEQUENCE_MASK
    return SLIM_SEQUENCE_MASK


class SequenceDropCounter:
    """Charge a drop for every frame the device sent but the host never got."""

    def __init__(self) -> None:
        self.frames = 0
        self.dropped = 0
        self._last_sequence: int | None = None
        self._last_host_us: int | None = None

    def reset(self) -> None:
        """Forget the last frame; the next one has nothing to compare against.

        Called when the transport (re)opens the port: the counterpart of the
        flush exemption, for the case where the first frame after a reconnect
        would otherwise be charged the whole backlog of the previous session.
        """

        self._last_sequence = None
        self._last_host_us = None

    def observe(self, sequence: int, host_us: int, mask: int) -> None:
        """Feed one received frame, whatever its type.

        Every frame type draws from the same counter, so this must see every
        frame the parser produced -- feeding it only the IMU frames would charge
        the pressure, status and telemetry frames in between as loss.
        """

        self.frames += 1
        if self._last_sequence is not None:
            gap = (int(sequence) - self._last_sequence) & int(mask)
            # Seconds, to match ``DROP_FREE_FPS``: a gap of 300 frames is only
            # charged when the two frames really were 0.75 s apart.  A flush
            # hands over the same 300 frames in a few milliseconds and is
            # therefore exempt, which is the entire point of the test.
            elapsed_s = (int(host_us) - int(self._last_host_us)) / 1_000_000.0
            if gap > 0 and elapsed_s > gap / DROP_FREE_FPS:
                self.dropped += gap - 1
        self._last_sequence = int(sequence)
        self._last_host_us = int(host_us)

    def snapshot(self) -> dict[str, int]:
        """Cumulative tally; a caller wanting a rate takes deltas.

        ``frames`` counts every frame type, the same unit as ``dropped``, so
        ``dropped / (frames + dropped)`` is the share of the device's own stream
        that never reached the host.
        """

        return {"frames": self.frames, "dropped": self.dropped}
