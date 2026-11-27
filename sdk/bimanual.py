"""Plaintext bimanual synchronization orchestration."""

from __future__ import annotations

import logging
import threading
import time

from sdk.glove import Glove, GloveEngine
from common.errors import StreamClosedError, StreamTimeoutError
from common.types import BimanualConfig, BimanualFrame, BimanualHealth, HandFrame


_PAIR_READ_SLICE_S = 0.05


class BimanualEngine:
    def __init__(self, config: BimanualConfig):
        self.config = config
        self.left = GloveEngine(config.left)
        self.right = GloveEngine(config.right)
        self._sequence = 0
        self._started = False
        self._lifecycle_lock = threading.RLock()
        self._last_sync_error_ms: float | None = None

    def start(self) -> "BimanualEngine":
        with self._lifecycle_lock:
            if self._started:
                return self
            attempted = []
            try:
                for engine in (self.left, self.right):
                    attempted.append(engine)
                    engine.start()
            except BaseException:
                for engine in reversed(attempted):
                    try:
                        engine.close()
                    except Exception:
                        logging.getLogger(__name__).exception("Bimanual startup rollback failed")
                raise
            self._started = True
        return self

    def read(self, timeout: float | None = None) -> BimanualFrame:
        """The next synchronized pair, or raise if nothing new arrived.

        The two hands are independent links, so one of them can go quiet while
        the other keeps delivering.  Reading them in turn and letting whichever
        timeout raise used to throw away the frame the healthy hand had already
        produced, which froze *both* hands on screen until the caller's own
        staleness check took over.  A hand that produced nothing therefore
        contributes its newest frame from the current connection generation.

        The exception is still raised when *neither* hand has anything new, so
        a caller polling at its own pace keeps seeing "nothing yet" rather than
        a stream of repeats -- an idle glove must not look busy.  It is raised
        as well when the silent hand has no frame from its current connection.
        A repeated ``sequence`` marks the hand
        that was filled in, and its ``sensor_age_s`` and health show how far
        behind it is, so nothing is hidden.
        """

        with self._lifecycle_lock:
            self.start()
            engines = (self.left, self.right)
            queues = [engine._read_queue() for engine in engines]
        # One budget for the pair, not one per hand: charging each hand the
        # full timeout would make a silent link cost twice what the caller
        # asked to wait.
        deadline = None if timeout is None else time.monotonic() + timeout
        fresh = [None, None]
        while True:
            slice_end = time.monotonic() + _PAIR_READ_SLICE_S
            if deadline is not None:
                slice_end = min(slice_end, deadline)
            for index, (engine, queue) in enumerate(zip(engines, queues)):
                engine._check_read_queue(queue)
                if fresh[index] is not None:
                    continue
                try:
                    fresh[index] = engine._read_frame(
                        queue, max(0.0, slice_end - time.monotonic()))
                except StreamTimeoutError:
                    pass
            # Check both queues between waits so a failed hand wakes a pair
            # reader that's currently waiting for the healthy hand.
            if all(frame is not None for frame in fresh) or slice_end == deadline:
                break
        with self._lifecycle_lock:
            if not self._started:
                raise StreamClosedError("bimanual glove was closed")
            with (self.left._lock, self.right._lock,
                  self.left._transport_state_lock(), self.right._transport_state_lock()):
                for engine, queue in zip(engines, queues):
                    engine._check_read_queue(queue)
                frame = self._pair(*fresh)
            return frame

    def _pair(self, left: HandFrame | None, right: HandFrame | None) -> BimanualFrame:
        left_fresh = self.left._pose_is_current(left)
        right_fresh = self.right._pose_is_current(right)
        if not (left_fresh or right_fresh):
            raise StreamTimeoutError("neither hand delivered a frame")
        if not left_fresh:
            left = self.left.latest()
        if not right_fresh:
            right = self.right.latest()
        if left is None or right is None:
            silent = self.config.left.side if left is None else self.config.right.side
            raise StreamTimeoutError(f"{silent} hand has delivered no frame yet")
        # A force baseline can change while read() waits for the other hand.
        left = self.left._current_force_frame(left)
        right = self.right._current_force_frame(right)
        timestamp_us = max(left.timestamp_us, right.timestamp_us)
        frame = BimanualFrame(
            sequence=self._sequence,
            timestamp_us=timestamp_us,
            left=left,
            right=right,
            synchronization_error_ms=(
                abs(left.timestamp_us - right.timestamp_us) / 1000.0),
        )
        # Either connection can advance while we look up the other hand.
        if not (self.left._pose_is_current(left) and self.right._pose_is_current(right)):
            raise StreamTimeoutError("a hand reconnected before the pair could be published")
        self._last_sync_error_ms = frame.synchronization_error_ms
        self._sequence += 1
        return frame

    def health(self) -> BimanualHealth:
        return BimanualHealth(
            left=self.left.health(),
            right=self.right.health(),
            synchronization_error_ms=self._last_sync_error_ms,
        )

    def close(self) -> None:
        with self._lifecycle_lock:
            self._started = False
            self._sequence = 0
            self._last_sync_error_ms = None
            try:
                self.left.close()
            finally:
                self.right.close()


class BimanualGlove:
    __slots__ = ("_impl",)

    def __init__(self, config: BimanualConfig):
        self._impl = BimanualEngine(config)

    @property
    def left(self) -> Glove:
        """The left hand's own facade -- for state a frame does not carry.

        ``BimanualFrame`` carries one complete ``HandFrame`` per hand, which is
        everything a frame reader needs.  Two things live outside it: the newest
        solved mesh, and the solver backend.  Both are properties of one hand's
        engine, and the returned facade is the same public surface a
        single-hand caller already gets, so nobody has to reach into ``_impl``.
        """

        return Glove._wrap(self._impl.left)

    @property
    def right(self) -> Glove:
        """The right hand's own facade.  See :attr:`left`."""

        return Glove._wrap(self._impl.right)

    def start(self) -> "BimanualGlove":
        self._impl.start()
        return self

    def read(self, timeout: float | None = None) -> BimanualFrame:
        return self._impl.read(timeout)

    def health(self) -> BimanualHealth:
        return self._impl.health()

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "BimanualGlove":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
