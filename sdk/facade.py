"""Client ("user") edition of the build seam: restricted gloves, not full ones.

:mod:`sdk.api` builds gloves through these two functions instead of calling
:class:`~sdk.glove.Glove` / :class:`~sdk.bimanual.BimanualGlove` directly.  This
edition wraps them in the facades from :mod:`sdk.user_facade`, which drop every
raw IMU and magnetometer field on the way out. Pressure is published as both
the original ADC scan and the newtons fitted from that same scan inside the
boundary (see sdk.user_facade).

Engine references remain private to the restricted wrappers.
"""

from __future__ import annotations

from common.types import BimanualConfig, GloveConfig
from sdk.bimanual import BimanualGlove as _EngineBimanualGlove
from sdk.glove import Glove as _EngineGlove
from sdk.user_facade import UserBimanualGlove, UserGlove


def make_glove(config: GloveConfig) -> UserGlove:
    """Build the single-hand facade, restricted to joints and newtons."""

    return UserGlove(_EngineGlove(config))


def make_bimanual(config: BimanualConfig) -> UserBimanualGlove:
    """Build the two-hand facade, restricted to joints and newtons."""

    return UserBimanualGlove(_EngineBimanualGlove(config))


__all__ = ["make_bimanual", "make_glove"]
