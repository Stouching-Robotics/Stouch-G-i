"""Lazy GPU selection with a latched, manually overridable CPU fallback."""
from __future__ import annotations

import logging
import os

from PySide6.QtGui import QImage


class CalibrationRenderBackend:
    def __init__(self, mode=None, *, factory=None):
        self._factory = factory
        self._renderer = None
        self._details = {}
        self.set_mode(mode if mode is not None else os.getenv('STOUCH_CALIBRATION_RENDERER', 'auto'))

    @property
    def mode(self):
        return self._mode

    @property
    def active(self):
        return self._active

    @property
    def info(self):
        return dict(self._details, backend=self._active, reason=self._reason)

    def _release(self):
        renderer, self._renderer = self._renderer, None
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                logging.getLogger(__name__).debug('Calibration GL cleanup failed', exc_info=True)

    def set_mode(self, mode):
        self._release()
        self._details = {}
        self._mode = str(mode).strip().lower()
        invalid = self._mode not in ('auto', 'software')
        if invalid:
            self._mode = 'software'
        self._active = 'pending' if self._mode == 'auto' else 'software'
        self._reason = ('Unknown renderer preference; using CPU compatibility' if invalid else
                        'CPU compatibility requested' if self._mode == 'software' else '')

    def render(self, target_px, live_px, faces, width, height):
        """Return a rendered image, or None to use the caller's CPU renderer.

        Initialization is deliberately inside the guarded path. Forced software
        mode never imports QtOpenGL and cannot accidentally initialize a driver.
        Failure stays in software until the user explicitly retries auto mode.
        """
        if self._active == 'software':
            return None
        try:
            if self._renderer is None:
                factory = self._factory
                if factory is None:
                    from gui.calibration_gl_renderer import CalibrationGLRenderer
                    factory = CalibrationGLRenderer
                self._renderer = factory()
                self._details = dict(self._renderer.info)
            image = self._renderer.render(target_px, live_px, faces, width, height)
            if (not isinstance(image, QImage) or image.isNull()
                    or image.width() != width or image.height() != height):
                raise RuntimeError('OpenGL returned an invalid calibration image')
            self._active = 'opengl'
            self._details = dict(self._renderer.info)
            return image
        except Exception as exc:
            self._reason = str(exc) or type(exc).__name__
            self._active = 'software'
            self._release()
            logging.getLogger(__name__).warning(
                'Calibration GPU unavailable; continuing with CPU compatibility: %s', self._reason)
            return None

    def close(self, *_ignored):
        self._release()
        self._active = 'software'
