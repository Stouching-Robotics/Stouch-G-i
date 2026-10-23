"""The one entry point: open a glove and read everything from one frame.

Call :func:`open_glove` and nothing else in this package.  What arrives on
:class:`~sdk.HandFrame`:

======================  ==========================================
21-keypoint position    ``joints_raw_m`` / ``joints_smoothed_m``
finger force (newtons)  ``force_n`` (16x16, like the pressure matrix)
fitness of that force   ``force_fresh``
solver state            ``status`` -- read this before trusting a frame
firmware version        ``firmware_version`` (raw) / ``display_firmware_version``
SDK version             ``GloveInfo.sdk_version`` from :meth:`Glove.info`
link diagnostics        ``device_fps``, ``link_drops``, ``latency_ms``
======================  ==========================================

Neither link needs naming: ``link="auto"`` finds whichever of the wired USB
cable or the Bluetooth dongle is carrying this hand.  See :mod:`sdk.transport`.

``force_n`` is in newtons and laid out like the pressure matrix; the cells no
fitted region covers are NaN, and it is ``None`` on a frame that carries no
pressure scan at all.  ``force_fresh`` says whether that scan is new or one
standing over from an earlier frame -- a temporal filter that ignores it will
integrate the same reading twice.

Check ``status.details["warming_up"]`` before using a frame: while the IMUs are
still settling the joints are a neutral resting pose, not a solve of the
wearer's hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sdk.calibration import find_latest_hand2mm_calibration
from sdk.facade import make_bimanual, make_glove
from sdk.glove import SDK_ROOT
from common.types import BimanualConfig, GloveConfig, LINK_AUTO

if TYPE_CHECKING:
    # Annotations only.  Importing the concrete classes at runtime would make
    # ``sdk.api.Glove`` a second, undocumented way to reach the unrestricted
    # facade -- in the user edition that is a raw-data bypass around the whole
    # restriction.
    from sdk.bimanual import BimanualGlove
    from sdk.glove import Glove

__all__ = ["default_calibration_dir", "open_bimanual", "open_glove"]


def default_calibration_dir() -> Path:
    """Where :func:`open_glove` looks when the caller names no calibration."""

    return SDK_ROOT / "calibration"


def _resolve_calibration(side: str, calibration, calibration_dir) -> Path:
    """Explicit file wins; otherwise the newest usable file for this hand.

    Falling back to "newest" rather than a fixed filename matters because the
    calibration folder accumulates files from several tools -- see
    :func:`sdk.calibration.find_latest_hand2mm_calibration`.
    """

    if calibration is not None:
        return Path(calibration).expanduser()
    folder = (Path(calibration_dir).expanduser() if calibration_dir is not None
              else default_calibration_dir())
    return find_latest_hand2mm_calibration(folder, side).expanduser()


def open_glove(
    side: str = "right",
    *,
    calibration: Path | str | None = None,
    calibration_dir: Path | str | None = None,
    link: str = LINK_AUTO,
    serial_number: str | None = None,
    registry: Path | str | None = None,
    sample_rate_hz: int = 80,
    include_tactile: bool = True,
    smoothing_ms: float = 35.0,
    recording_root: Path | str | None = None,
) -> Glove:
    """Open one hand.  The returned glove is not started until first use.

    ``with open_glove("right") as glove:`` starts it; so does the first
    :meth:`Glove.read`, which is why ``info()`` reports the link and firmware
    only after one of those has happened.
    """

    config = GloveConfig(
        side=side,
        calibration=_resolve_calibration(side, calibration, calibration_dir),
        serial_number=serial_number,
        registry=registry,
        sample_rate_hz=sample_rate_hz,
        include_tactile=include_tactile,
        smoothing_ms=smoothing_ms,
        recording_root=recording_root,
        link=link,
    )
    return make_glove(config)


def open_bimanual(
    *,
    left_calibration: Path | str | None = None,
    right_calibration: Path | str | None = None,
    calibration_dir: Path | str | None = None,
    left_link: str = LINK_AUTO,
    right_link: str = LINK_AUTO,
    left_serial: str | None = None,
    right_serial: str | None = None,
    registry: Path | str | None = None,
    sample_rate_hz: int = 80,
    include_tactile: bool = True,
    smoothing_ms: float = 35.0,
    output_root: Path | str | None = None,
    display_separation_m: float = 0.30,
) -> BimanualGlove:
    """Open both hands synchronized, with per-hand link and serial overrides."""

    config = BimanualConfig(
        left=GloveConfig(
            side="left",
            calibration=_resolve_calibration(
                "left", left_calibration, calibration_dir),
            serial_number=left_serial,
            registry=registry,
            sample_rate_hz=sample_rate_hz,
            include_tactile=include_tactile,
            smoothing_ms=smoothing_ms,
            recording_root=output_root,
            link=left_link,
        ),
        right=GloveConfig(
            side="right",
            calibration=_resolve_calibration(
                "right", right_calibration, calibration_dir),
            serial_number=right_serial,
            registry=registry,
            sample_rate_hz=sample_rate_hz,
            include_tactile=include_tactile,
            smoothing_ms=smoothing_ms,
            recording_root=output_root,
            link=right_link,
        ),
        output_root=output_root,
        display_separation_m=display_separation_m,
    )
    return make_bimanual(config)
