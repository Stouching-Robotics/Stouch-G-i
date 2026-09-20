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

The policy here is the one that survives both shapes: drop the part of the batch
that has fallen behind, keep the rest, and let the caller's own rate budget
decide how many of the survivors to act on.  Keeping the decision a pure
function of the frames is deliberate -- it is the piece worth testing offline,
and it needs no hardware, no transport and no GUI.

Lives in :mod:`common` because the GUI and the SDK's offline self-test both use
it, and the self-test must not drag in the GUI's rendering dependencies.
"""

from __future__ import annotations

import math

# How much of a poll batch is still worth acting on, in units of one pacing
# interval.  Two intervals keeps a batch that spans the current and previous
# interval while still discarding a genuine backlog.
BATCH_KEEP_INTERVALS = 2.0

# The firmware timestamp is a wrapping uint32, so every difference is taken
# modulo this.
_TIMESTAMP_MODULUS = 0x100000000


def recent_batch(frames, rate_hz: float):
    """Return the newest frames of ``frames``, dropping only the stale part.

    ``frames`` are in arrival order, so the last one is the newest.  A frame is
    kept when it is within :data:`BATCH_KEEP_INTERVALS` pacing intervals of the
    newest frame, using each frame's ``device_timestamp_us`` -- device time
    rather than host arrival time, so a batch the host received late is still
    measured against the glove's own clock.

    The interval is rounded *up* to whole microseconds.  Device timestamps are
    integers, so a batch's true span is several truncated intervals and comes
    out just over the exact product: three frames at 60 Hz span 33 334 us
    against an exact two intervals of 33 333.  Truncating would push a batch's
    own oldest frame outside its window and re-introduce the very loss this
    exists to prevent, decided by a single microsecond.

    Always returns at least the newest frame, so a caller can never be handed an
    empty list for a non-empty poll.
    """

    if len(frames) <= 1:
        return frames
    interval_us = math.ceil(1_000_000.0 / max(float(rate_hz), 1.0))
    window_us = int(BATCH_KEEP_INTERVALS * interval_us)
    newest = int(frames[-1].device_timestamp_us)
    keep = [
        frame for frame in frames
        if ((newest - int(frame.device_timestamp_us)) % _TIMESTAMP_MODULUS)
        <= window_us]
    return keep or frames[-1:]
