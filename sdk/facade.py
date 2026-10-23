"""Client ("user") edition of the build seam: restricted gloves, not full ones.

:mod:`sdk.api` builds gloves through these two functions instead of calling
:class:`~sdk.glove.Glove` / :class:`~sdk.bimanual.BimanualGlove` directly.  This
edition wraps them in the facades from :mod:`sdk.user_facade`, which drop every
raw IMU and magnetometer field on the way out. Pressure is published as both
the original ADC scan and the newtons fitted from that same scan inside the
boundary (see sdk.user_facade).

The inner objects are the real engine and stay reachable as ``_inner``; that is
noted, not defended.  See the packaging manifest for what this edition does and
does not guarantee.
"""

from __future__ import annotations

from common.types import BimanualConfig, GloveConfig
from sdk.bimanual import BimanualGlove
from sdk.glove import Glove
from sdk.user_facade import UserBimanualGlove, UserGlove


def make_glove(config: GloveConfig) -> UserGlove:
    """Build the single-hand facade, restricted to joints and newtons."""

    return UserGlove(Glove(config))


def make_bimanual(config: BimanualConfig) -> UserBimanualGlove:
    """Build the two-hand facade, restricted to joints and newtons."""

    return UserBimanualGlove(BimanualGlove(config))


__all__ = ["make_bimanual", "make_glove"]
