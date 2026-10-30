"""Public SDK for STM32 HAND2mm gloves: solved keypoints, force, pressure ADC.

    from sdk import open_glove

    with open_glove("right") as glove:          # link="auto" finds USB or BT
        print(glove.info())                     # firmware + SDK version, link
        while True:
            frame = glove.read(timeout=1.0)     # one frame, everything in it
            frame.joints_smoothed_m             # 21 keypoints, metres
            frame.force_n                       # 16x16 finger force, newtons
            frame.tactile_adc                   # 16x16 raw pressure ADC, counts
            frame.status                        # read this before trusting one

:class:`~sdk.HandFrame` is the whole contract; it is documented on the class.

**Read ``status`` before trusting a frame.**  The IMUs need a moment to settle
after the glove is powered on, and while they are settling the solver returns a
*neutral* resting pose rather than a solve of the wearer's hand.  Those frames
look entirely plausible and are wrong.  ``frame.status.details["warming_up"]``
is ``True`` for exactly those, so:

    frame = glove.read()
    if frame.status.details.get("warming_up"):
        continue                                # placeholder, not the wearer
    use(frame.joints_smoothed_m, frame.force_n)

``frame.tactile_adc`` is the raw 12-bit pressure matrix the device sent, in
sensor counts, alongside the newtons it fits to -- the same scan either way, so
``frame.force_fresh`` says whether both are new.  A client that wants its own
force calibration, or the sensor itself, reads that; a client that wants forces
reads ``force_n``.

This is the client edition.  It builds gloves and hands over frames; the IMU
stream behind the solve, the magnetometer, the solver, the device manager, the
calibration session and the pressure-fit parameter channel are not part of its
surface.
"""

from __future__ import annotations

from common.errors import (
    CalibrationCompatibilityError,
    CalibrationError,
    CalibrationQualityError,
    DeviceBindingError,
    DeviceBusyError,
    DeviceNotFoundError,
    GloveSdkError,
    StreamClosedError,
    StreamError,
    StreamTimeoutError,
    UnsupportedPlatformError,
)
from common.usb_cdc import display_firmware_version, firmware_real_version
from common.types import (
    BimanualConfig,
    BimanualHealth,
    DeviceHealth,
    GloveConfig,
    GloveInfo,
    HandStatus,
    LINK_AUTO,
    LINK_BLUETOOTH,
    LINK_USB,
    Side,
)
from sdk.user_facade import (
    UserBimanualFrame as BimanualFrame,
    UserBimanualGlove as BimanualGlove,
    UserGlove as Glove,
    UserHandFrame as HandFrame,
)

__version__ = "2.3.3"


def get_version() -> str:
    """Return the SDK version string, currently ``"2.3.3"``.

    Not to be confused with the glove's own ``HandFrame.firmware_version``.
    The two are independent and may share a string by coincidence -- Bluetooth
    firmware is at ``"2.0.7"`` while the SDK is at ``"2.3.3"``.

    ``2.3.1`` is the release that added ``HandFrame.tactile_adc`` and the
    per-link frame dialects (Bluetooth compresses its pressure scans, wired
    sends one full picture each); ``2.2.x`` callers that only read
    ``force_n`` are unaffected by both.
    """

    return __version__


def __getattr__(name: str):
    """Serve :func:`open_glove` / :func:`open_bimanual` lazily.

    Reached through here rather than through a ``*args, **kwargs`` forwarder.
    A forwarder exists to be readable, and it is not: ``help(sdk.open_glove)``
    and an IDE would show ``(*args, **kwargs)`` for the one entry point the
    caller is told to call.  Resolving the name on access keeps ``import sdk``
    cheap and leaves the signature intact.
    """
    if name in _INTERFACE_NAMES:
        import importlib
        return getattr(importlib.import_module(_INTERFACE_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Only the two entry points resolve lazily here.  The class names are bound
# eagerly above, and everything else the per-stream interfaces used to serve --
# the raw instrumentation streams, the solver, the device manager, the
# calibration session, the pressure-fit channel -- is simply absent from this
# edition, so no name reaches it.
_INTERFACE_NAMES = ("open_glove", "open_bimanual")
_INTERFACE_MODULES = {
    "open_glove": "sdk.api",
    "open_bimanual": "sdk.api",
}

# The public surface.  Narrow on purpose: the caller has one way in, and that
# way is open_glove.  Frame-shape and config contracts are here; the machinery
# behind them is not.
__all__ = [
    "__version__",
    "get_version",
    "open_glove", "open_bimanual",
    "Glove", "BimanualGlove",
    "GloveConfig", "BimanualConfig", "GloveInfo",
    "HandFrame", "BimanualFrame", "HandStatus",
    "DeviceHealth", "BimanualHealth",
    "Side",
    "display_firmware_version", "firmware_real_version",
    "GloveSdkError", "DeviceNotFoundError", "DeviceBindingError",
    "DeviceBusyError", "StreamError", "StreamTimeoutError",
    "StreamClosedError", "CalibrationError", "CalibrationQualityError",
    "CalibrationCompatibilityError", "UnsupportedPlatformError",
]
