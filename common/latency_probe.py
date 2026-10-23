"""Where the pressure matrix's delay accumulates, hop by hop.

Every tactile frame carries the device's own scan timestamp, and device time and
host time share no epoch -- so no absolute latency can be read off a frame, and
"the panel is laggy" has never had a number behind it.  What *can* be measured is
how long one frame sits at each host-side hop, because every hop is stamped from
the same monotonic clock.

Three stamps ride along with a frame and each gap between neighbours is one place
the delay can grow:

======================  =====================================================
``poll - enqueue``      waiting in the transport queue, for the consumer to ask
``publish - poll``      the solver's own work: the copy, the preprocessor, and
                        the pickling that puts the arrays on the IPC queue
``paint - publish``     everything the parent owns -- the IPC queue, the main
                        loop, and the display gate -- in one number
======================  =====================================================

The split is drawn at the process boundary on purpose: the first two live in the
hand's process and the third in the parent, and knowing which side of that line
the time is on is most of the diagnosis.

``time.perf_counter`` is system-wide on both platforms this runs on (QPC on
Windows, ``CLOCK_MONOTONIC`` elsewhere), so stamps taken in different processes
are still comparable -- which is what lets the parent subtract stamps the solver
took.

Off unless asked for.  Stamping every frame is work, and a probe that changes the
timing it measures is worse than no probe: ``GLOVE_LATENCY_PROBE=1`` turns it on,
and ``--link-diag`` sets it, so the switch already in use covers this too.  While
it is off every stamp is ``0.0`` and every hop is skipped.
"""

from __future__ import annotations

import os
import time
from collections import deque
from threading import Lock

_ENV_VAR = "GLOVE_LATENCY_PROBE"

# How many frames one window holds.  300 is about five seconds at the tactile
# rate: long enough for a stable median, short enough that a stall still shows
# in the very next log line instead of being averaged into a minute of health.
WINDOW = 300

_ENABLED = os.environ.get(_ENV_VAR, "").strip().lower() not in (
    "", "0", "false", "no")


def enabled() -> bool:
    """True when hop stamps are being taken."""

    return _ENABLED


def enable() -> None:
    """Turn the probe on for this process and for any process it spawns.

    Writing the environment variable is the half that matters for the hand
    processes: they are spawned, so they re-import this module and cannot see a
    flag set in memory here.  The parent must call this before it spawns them.
    """

    global _ENABLED
    _ENABLED = True
    os.environ[_ENV_VAR] = "1"


def disable() -> None:
    """Turn the probe off, in memory and in the environment.

    The counterpart to :func:`enable`, and the reason it exists is the offline
    self-test: it has to leave the module exactly as it found it, or a later
    test in the same process would silently start stamping.
    """

    global _ENABLED
    _ENABLED = False
    os.environ.pop(_ENV_VAR, None)


def stamp() -> float:
    """One hop's host timestamp, or ``0.0`` when the probe is off."""

    return time.perf_counter() if _ENABLED else 0.0


class _Side:
    __slots__ = ("queue", "solver", "parent")

    def __init__(self) -> None:
        self.queue: deque[float] = deque(maxlen=WINDOW)
        self.solver: deque[float] = deque(maxlen=WINDOW)
        self.parent: deque[float] = deque(maxlen=WINDOW)


_sides = {"left": _Side(), "right": _Side()}
_lock = Lock()


def record(
        side: str,
        enqueue_us: float,
        poll_us: float,
        publish_us: float,
        paint_us: float,
) -> None:
    """Split one painted frame's life into the three hops, in seconds."""

    if not _ENABLED:
        return
    # A zero stamp means one of this frame's producers ran with the probe off --
    # a link that reconnected into a process that never saw the variable, say.
    # Skipping the frame is honest; subtracting a zero would report a hop of
    # decades and swamp the window.
    if not (enqueue_us and poll_us and publish_us and paint_us):
        return
    # Each clock is monotonic and the stamps are taken in that order, so a frame
    # that appears to travel backwards is a mix-up between frames (a stale stamp
    # paired with a fresh one).  Dropping it beats reporting a negative hop.
    if poll_us < enqueue_us or publish_us < poll_us or paint_us < publish_us:
        return
    with _lock:
        bucket = _sides.get(side)
        if bucket is None:
            return
        bucket.queue.append(poll_us - enqueue_us)
        bucket.solver.append(publish_us - poll_us)
        bucket.parent.append(paint_us - publish_us)


def samples(side: str) -> dict[str, list[float]]:
    """Copy the collected hop windows, in seconds.

    Separated from :func:`report` so the arithmetic can be checked as numbers
    rather than as a formatted line -- the self-test asserts on these, and a
    test that parses the sentence it is testing would pass on a wrong sentence.
    """

    with _lock:
        bucket = _sides.get(side)
        if bucket is None:
            return {"queue": [], "solver": [], "parent": []}
        return {"queue": list(bucket.queue),
                "solver": list(bucket.solver),
                "parent": list(bucket.parent)}


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    last = len(ordered) - 1

    def at(fraction: float) -> float:
        return ordered[min(last, int(fraction * len(ordered)))]

    return at(0.5), at(0.95), ordered[-1]


def report(side: str) -> str | None:
    """One line of median/p95/max per hop, in milliseconds, or ``None``.

    ``None`` when the probe is off, and also when no frame has completed the
    whole path yet -- an empty window means the hand has not painted since the
    last reset, which is different from a hop that costs nothing.
    """

    if not _ENABLED:
        return None
    with _lock:
        bucket = _sides.get(side)
        if bucket is None or not bucket.queue:
            return None
        columns = (
            ("queue", _quantiles(list(bucket.queue))),
            ("solver", _quantiles(list(bucket.solver))),
            ("panel", _quantiles(list(bucket.parent))),
        )
    parts = [
        f"{name} {p50 * 1e3:5.1f}/{p95 * 1e3:5.1f}/{worst * 1e3:6.1f}"
        for name, (p50, p95, worst) in columns
    ]
    return f"tac med/p95/max ms  " + "  ".join(parts)


def reset(side: str | None = None) -> None:
    """Drop the collected window -- used by tests, not by the live path."""

    with _lock:
        for name, bucket in _sides.items():
            if side is not None and name != side:
                continue
            bucket.queue.clear()
            bucket.solver.clear()
            bucket.parent.clear()
