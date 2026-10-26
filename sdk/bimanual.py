"""Plaintext bimanual synchronization orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from sdk.glove import SDK_ROOT, Glove, GloveEngine
from common.errors import StreamTimeoutError
from common.types import (
    BimanualConfig, BimanualFrame, BimanualHealth, HandFrame)


class BimanualEngine:
    def __init__(self, config: BimanualConfig):
        self.config = config
        self.left = GloveEngine(config.left)
        self.right = GloveEngine(config.right)
        self._sequence = 0
        self._started = False
        self._last_sync_error_ms: float | None = None

    def start(self) -> "BimanualEngine":
        if not self._started:
            self.left.start()
            self.right.start()
            self._started = True
        return self

    @staticmethod
    def _read_fresh(engine: GloveEngine, deadline: float | None) -> "HandFrame":
        """One hand's next frame, waiting no longer than ``deadline``."""

        remaining = (None if deadline is None
                     else max(0.0, deadline - time.monotonic()))
        return engine.read(remaining)

    def read(self, timeout: float | None = None) -> BimanualFrame:
        """The next synchronized pair, or raise if nothing new arrived.

        The two hands are independent links, so one of them can go quiet while
        the other keeps delivering.  Reading them in turn and letting whichever
        timeout raise used to throw away the frame the healthy hand had already
        produced, which froze *both* hands on screen until the caller's own
        staleness check took over.  A hand that produced nothing therefore
        contributes the newest frame it did deliver, and the pair survives.

        The exception is still raised when *neither* hand has anything new, so
        a caller polling at its own pace keeps seeing "nothing yet" rather than
        a stream of repeats -- an idle glove must not look busy.  It is raised
        as well when the silent hand has never delivered anything at all, since
        half a pair is not a frame.  A repeated ``sequence`` marks the hand
        that was filled in, and its ``sensor_age_s`` and health show how far
        behind it is, so nothing is hidden.
        """

        if not self._started:
            self.start()
        # One budget for the pair, not one per hand: charging each hand the
        # full timeout would make a silent link cost twice what the caller
        # asked to wait.
        deadline = None if timeout is None else time.monotonic() + timeout
        fresh = []
        for engine in (self.left, self.right):
            try:
                fresh.append(self._read_fresh(engine, deadline))
            except StreamTimeoutError:
                fresh.append(None)
        left, right = fresh
        if left is None and right is None:
            raise StreamTimeoutError("neither hand delivered a frame")
        if left is None:
            left = self.left.latest()
        if right is None:
            right = self.right.latest()
        if left is None or right is None:
            silent = self.config.left.side if left is None else self.config.right.side
            raise StreamTimeoutError(f"{silent} hand has delivered no frame yet")
        timestamp_us = max(left.timestamp_us, right.timestamp_us)
        frame = BimanualFrame(
            sequence=self._sequence,
            timestamp_us=timestamp_us,
            left=left,
            right=right,
            synchronization_error_ms=(
                abs(left.timestamp_us - right.timestamp_us) / 1000.0),
        )
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
        self.left.close()
        self.right.close()
        self._started = False


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
