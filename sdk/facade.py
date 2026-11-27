"""Construct the user edition facades for Friston poses and tactile scans."""

from __future__ import annotations

from common.types import BimanualConfig, GloveConfig
from sdk.bimanual import BimanualGlove
from sdk.glove import Glove
from sdk.user_facade import UserBimanualGlove, UserGlove


def make_glove(config: GloveConfig) -> UserGlove:
    """Build the single-hand facade, for pose and tactile data."""

    return UserGlove(Glove(config))


def make_bimanual(config: BimanualConfig) -> UserBimanualGlove:
    """Build the two-hand facade, for pose and tactile data."""

    return UserBimanualGlove(BimanualGlove(config))


__all__ = ["make_bimanual", "make_glove"]
