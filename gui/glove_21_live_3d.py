"""Compatibility forwarding module for the plaintext live application."""

from gui import live_3d as _application
from gui.live_3d import *  # noqa: F401,F403
from gui.live_3d import (  # noqa: F401
    _draw_hand,
    _parse_args,
)


def render_frame_live(*args, **kwargs):
    """Forward while preserving legacy monkey-patching of ``_draw_hand``."""
    original = _application._draw_hand
    _application._draw_hand = _draw_hand
    try:
        return _application.render_frame_live(*args, **kwargs)
    finally:
        _application._draw_hand = original

if __name__ == "__main__":
    raise SystemExit(main())
