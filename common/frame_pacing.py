"""Real-time pacing policy for frames a single poll handed over.

A transport poll does not always return one frame.  A wired USB CDC link hands
over one frame per driver read, so the backlog a consumer sees is at most one
frame; a Bluetooth dongle relays an SPP stream and therefore hands over one
*batch* per RF event, so the same poll can carry several consecutive device
samples.

Live consumers want the newest data and not a replay of history, which was
originally expressed as "keep only the last frame of the poll".  On a burst that
throws away real samples and makes the consumer's rate equal the *batch* rate
instead of the frame rate -- at a three-frame batch cadence, a silent three-fold
loss that looks exactly like a slow glove.

The policy here is the one that survives both shapes: keep the whole batch, and
trim only what is old enough to be history rather than a burst.  Keeping the
decision a pure function of the frames is deliberate -- it is the piece worth
testing offline, and it needs no hardware, no transport and no GUI.

Lives in :mod:`common` because the GUI and the SDK's offline self-test both use
it, and the self-test must not drag in the GUI's rendering dependencies.
"""

from __future__ import annotations

# How far behind the newest frame a frame may be and still count as part of the
# same burst, in microseconds.
#
# This is deliberately not expressed in pacing intervals: coupling it to the
# consumer's target rate made a *higher* target trim *more* (80 Hz would keep
# only 25 ms), which is backwards, and it made the window narrower than the
# batches a dongle actually delivers.
#
# 250 ms is the same boundary the solve loop already uses to decide a device
# timestamp has jumped rather than advanced, so the two agree on where "current"
# ends.  Everything a Bluetooth link delivers -- RF batches of a few frames, and
# a dongle's buffer flushing on open -- fits inside it; only a consumer that
# stalled for a quarter second and let the queue accumulate real history does
# not, and that is exactly the backlog worth discarding.
BURST_SPAN_US = 250_000

# The firmware timestamp is a wrapping uint32, so every difference is taken
# modulo this.
_TIMESTAMP_MODULUS = 0x100000000


def recent_batch(frames, span_us: int = BURST_SPAN_US):
    """Return the newest frames of ``frames``, dropping only the stale part.

    ``frames`` are in arrival order, so the last one is the newest.  A frame is
    kept when it is within ``span_us`` of the newest frame, using each frame's
    ``device_timestamp_us`` -- device time rather than host arrival time, so a
    batch the host received late is still measured against the glove's own
    clock.

    Always returns at least the newest frame, so a caller can never be handed an
    empty list for a non-empty poll.
    """

    if len(frames) <= 1:
        return frames
    window_us = int(span_us)
    newest = int(frames[-1].device_timestamp_us)
    keep = [
        frame for frame in frames
        if ((newest - int(frame.device_timestamp_us)) % _TIMESTAMP_MODULUS)
        <= window_us]
    return keep or frames[-1:]
