"""The user ("client") edition's restricted facade.

The admin edition hands a caller ``HandFrame`` with everything on it: the 16 raw
IMU quaternions in physical channel order, the 16x16 raw pressure ADC matrix,
the magnetometer registers.  This module is what the user edition builds
instead.  It narrows a full frame to what a client application is meant to
consume -- solved joint positions, finger force in newtons, and the raw pressure
ADC matrix those newtons were fitted from -- and keeps the IMU quaternions, the
per-IMU validity and age, and every magnetometer register off its surface.

The ADC matrix is handed over deliberately (2026-10-15): a client that wants to
run its own force fit, or to see the sensor rather than the fit, needs the
readings the firmware actually sent, and withholding them only forced that
client to re-derive them from a private back door.  What the edition still
protects is the *fit*: :func:`algorithm.force_fit.fit_finger_force_matrix` is
the ADC->newton table, it is what the packaging step encrypts, and this module
holds the only route to newtons inside the boundary.

What this does *not* do is make raw data physically unreachable.  ``sdk.io``
ships plaintext because the engine needs it, so the raw decoders stay reachable
to someone who reads the source, and the firmware emits raw ADC on the wire
regardless.  The edition's guarantee is a public-surface guarantee; the asset it
actually protects is the ADC->newton conversion table, which lives encrypted in
:mod:`algorithm.force_fit`.  See ``tools/protection_manifest.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from common.types import (
    BimanualFrame,
    DeviceHealth,
    GloveInfo,
    HandFrame,
    HandStatus,
    Side,
    _array,
    _side,
)

if TYPE_CHECKING:
    # Annotations only.  The user edition's ``sdk/__init__.py`` imports
    # ``HandFrame`` from this module eagerly, so a runtime import of either
    # engine class here would pull the whole of ``sdk.glove`` in on ``import
    # sdk`` and undo the lazy surface.
    from sdk.bimanual import BimanualGlove
    from sdk.glove import Glove


#: Frame fields the user edition must never carry.  ``verify()`` in
#: ``tools/build_sdk.py`` asserts each one is absent from :class:`UserHandFrame`,
#: so adding a raw field back here fails the package build instead of shipping
#: quietly.  Kept as data rather than prose so the manifest and the build can
#: both point at one list.
#:
#: The pressure ADC used to be on this list.  It is not any more: the edition
#: hands it over as :attr:`UserHandFrame.tactile_adc` (2026-10-15), so
#: ``tactile_raw`` and ``tactile_fresh`` are the *internal* names it arrives
#: under rather than data the surface withholds -- denying the names while
#: exposing the values is a check that would keep passing while meaning the
#: opposite of what it says.  What stays denied is what the edition still does
#: not publish at all: the IMU stream behind the solve, and the magnetometer.
DENIED_FIELDS = (
    "imu_xyzw",
    "imu_valid",
    "sensor_age_s",
    "raw_imu_xyzw",
    "raw_imu_present",
    "raw_imu_valid",
    "raw_device_timestamp_us",
    "mag",
    "mag_levels",
    "mag_fresh",
    "mag_frame_levels_raw",
)


def force_newtons(adc_matrix, side: str) -> np.ndarray | None:
    """Raw ADC matrix -> ``(16,16)`` float32 newtons, ``None`` if there is none.

    The import is deliberately inside the function.  :mod:`algorithm.force_fit`
    is what the packaging step encrypts, and reaching it lazily keeps
    ``import sdk`` cheap -- the same reason :mod:`sdk` resolves its interface
    names through a module ``__getattr__``.
    """

    if adc_matrix is None:
        return None
    from algorithm.force_fit import fit_finger_force_matrix

    return fit_finger_force_matrix(adc_matrix, side)


@dataclass(frozen=True)
class UserHandFrame:
    """One solved frame: joint positions, finger force, and the pressure ADC.

    ``joints_raw_m`` and ``joints_smoothed_m`` are ``(21,3)`` metres in the
    hand's canonical keypoint order; the smoothed copy has ``smoothing_ms`` of
    low-pass applied, the raw one has not.

    ``force_n`` is ``(16,16)`` newtons laid out like the pressure matrix, NaN in
    the cells no fitted region covers, ``None`` when this frame carries no
    pressure scan at all.  ``force_fresh`` says whether that scan is new or one
    standing over from an earlier frame -- a temporal filter that ignores it
    will integrate the same reading twice.

    ``tactile_adc`` is that same scan before fitting: the ``(16,16)`` raw 12-bit
    sensor counts (0..4095) exactly as the device sent them, ``None`` when there
    is no scan.  It and ``force_n`` are always the same scan, so ``force_fresh``
    is its freshness flag too.  Rows and columns are physical channels, laid out
    as the matrix is drawn and not re-ordered by hand side -- see the firmware
    interface document, section 6.2.

    ``status`` earns its place even though it is not sensor data: while the
    solver warms up it reports ``warming_up``, and until that clears the joint
    values are a neutral placeholder rather than a solve of the wearer's hand.
    Readings taken without checking it look plausible and are wrong.

    ``device_fps`` and ``tactile_fps`` are rates, not readings: attitude frames
    and pressure scans are two independent device streams at the same nominal
    rate, and each number is what arrived on its own stream over the host's
    last one-second window, counted by arrival time rather than by the device
    clock (which jumps a whole batch on a Bluetooth handover).  A scan stays
    attached to every attitude frame that runs while it is the newest, so
    ``tactile_fps`` -- not ``device_fps`` -- is the panel's own refresh rate.
    """

    side: Side
    sequence: int
    timestamp_us: int
    joints_raw_m: np.ndarray
    joints_smoothed_m: np.ndarray
    force_n: np.ndarray | None
    tactile_adc: np.ndarray | None
    status: HandStatus
    force_fresh: bool = False
    firmware_version: str | None = None
    link_kind: str = "usb"
    device_fps: float = 0.0
    tactile_fps: float = 0.0
    link_drops: dict[str, int] = field(default_factory=dict)
    link_errors: dict[str, int] = field(default_factory=dict)
    latency_ms: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "joints_raw_m", _array(
            self.joints_raw_m, (21, 3), np.float32, "joints_raw_m"))
        object.__setattr__(self, "joints_smoothed_m", _array(
            self.joints_smoothed_m, (21, 3), np.float32, "joints_smoothed_m"))
        if self.force_n is not None:
            object.__setattr__(self, "force_n", _array(
                self.force_n, (16, 16), np.float32, "force_n"))
        if self.tactile_adc is not None:
            object.__setattr__(self, "tactile_adc", _array(
                self.tactile_adc, (16, 16), np.float32, "tactile_adc"))


@dataclass(frozen=True)
class UserBimanualFrame:
    """Both hands, each already narrowed to :class:`UserHandFrame`."""

    sequence: int
    timestamp_us: int
    left: UserHandFrame
    right: UserHandFrame
    synchronization_error_ms: float


def restrict(frame: HandFrame) -> UserHandFrame:
    """Narrow one full internal frame to the user edition's public contract.

    Every field not named here is dropped on the floor: the IMU quaternions and
    their per-channel masks and ages, the magnetometer registers, the raw device
    clock.  The pressure scan is named twice on purpose -- once as the ADC the
    device sent, once as the newtons it fits to -- because they are one scan:
    the ADC is copied out through :class:`UserHandFrame`'s own validation, and
    the fit runs on the original.  ``link_drops``/``link_errors`` are copied
    rather than shared so a caller mutating its dict cannot reach back into the
    internal frame.
    """

    return UserHandFrame(
        side=frame.side,
        sequence=frame.sequence,
        timestamp_us=frame.timestamp_us,
        joints_raw_m=frame.joints_raw_m,
        joints_smoothed_m=frame.joints_smoothed_m,
        force_n=force_newtons(frame.tactile_raw, frame.side),
        tactile_adc=frame.tactile_raw,
        status=frame.status,
        force_fresh=bool(frame.tactile_fresh),
        firmware_version=frame.firmware_version,
        link_kind=frame.link_kind,
        device_fps=float(frame.device_fps),
        tactile_fps=float(frame.tactile_fps),
        link_drops=dict(frame.link_drops or {}),
        link_errors=dict(frame.link_errors or {}),
        latency_ms=frame.latency_ms,
    )


def restrict_bimanual(frame: BimanualFrame) -> UserBimanualFrame:
    """Narrow a two-hand frame; each half goes through :func:`restrict`."""

    return UserBimanualFrame(
        sequence=frame.sequence,
        timestamp_us=frame.timestamp_us,
        left=restrict(frame.left),
        right=restrict(frame.right),
        synchronization_error_ms=frame.synchronization_error_ms,
    )


class UserGlove:
    """Restricted single-hand facade: joints, newtons, and the pressure ADC.

    Deliberately absent, and not to be added back:

    * ``start_recording`` -- the recorder writes ``raw_imu_*`` (and the raw
      device clock) straight into Parquet columns, so exposing recording hands
      the IMU stream back through a documented call and the whole restriction is
      decorative.  The pressure ADC is no longer a reason to withhold it, but
      the IMU side still is.
    * ``config`` -- carries the calibration path, serial number and registry.
    * every ``pressure_fit_*`` -- the parameter channel is the admin edition's.

    Exposed on purpose, as *display* metadata rather than sensor data:
    ``mesh_vertices_m`` (the solved hand geometry), ``backend`` (the solver's
    name and layout), ``serial_number`` and ``port`` -- a viewer needs them to
    render and to label what it is looking at, and none of them carry a raw
    IMU quaternion or a magnetometer register.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Glove) -> None:
        self._inner = inner

    def start(self) -> "UserGlove":
        self._inner.start()
        return self

    def read(self, timeout: float | None = None) -> UserHandFrame:
        """The next frame: joints, newtons, and the ADC they were fitted from."""

        return restrict(self._inner.read(timeout))

    def latest(self) -> UserHandFrame | None:
        """The newest frame without blocking, or ``None`` before the first."""

        frame = self._inner.latest()
        return None if frame is None else restrict(frame)

    def info(self) -> GloveInfo:
        """Firmware, link, port, serial -- and the SDK's own version."""

        return self._inner.info()

    def health(self) -> DeviceHealth:
        return self._inner.health()

    def neutral_joints(self):
        """The resting pose, for a viewer that has nothing to display yet."""

        return self._inner.neutral_joints()

    def wait_for_firmware_version(self, timeout_s: float = 5.0) -> str | None:
        """Block until the device names its firmware, or give up (``None``)."""

        return self._inner.wait_for_firmware_version(timeout_s)

    @property
    def side(self) -> str:
        return self._inner.side

    @property
    def serial_number(self) -> str:
        """The bound serial number; ``""`` when the device is unbound."""

        return self._inner.serial_number

    @property
    def port(self) -> str | None:
        """The resolved link port (COM name / device node), or ``None``."""

        return self._inner.port

    @property
    def mesh_vertices_m(self) -> np.ndarray | None:
        """The solved hand mesh, for a viewer; ``None`` before the first solve."""

        return self._inner.mesh_vertices_m

    @property
    def backend(self) -> dict:
        """The solver backend descriptor (name, geometry); display metadata."""

        return self._inner.backend

    def set_contact_controls(self, enabled: bool, blend: float = 1.0) -> None:
        """Set contact correction strength for subsequent solved frames."""
        self._inner.set_contact_controls(enabled, blend)

    def set_root_limits_enabled(self, enabled: bool) -> None:
        """Enable or disable the default four-finger joint limits."""
        self._inner.set_root_limits_enabled(enabled)

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> "UserGlove":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


class UserBimanualGlove:
    """Restricted two-hand facade.  Same omissions as :class:`UserGlove`."""

    __slots__ = ("_inner",)

    def __init__(self, inner: BimanualGlove) -> None:
        self._inner = inner

    def start(self) -> "UserBimanualGlove":
        self._inner.start()
        return self

    def read(self, timeout: float | None = None) -> UserBimanualFrame:
        return restrict_bimanual(self._inner.read(timeout))

    def health(self):
        return self._inner.health()

    @property
    def left(self) -> UserGlove:
        return UserGlove(self._inner.left)

    @property
    def right(self) -> UserGlove:
        return UserGlove(self._inner.right)

    def set_contact_controls(self, enabled: bool, blend: float = 1.0) -> None:
        """Apply the same contact controls to both hands."""
        forward = getattr(self._inner, "set_contact_controls", None)
        if callable(forward):
            forward(enabled, blend)
        else:
            self.left.set_contact_controls(enabled, blend)
            self.right.set_contact_controls(enabled, blend)

    def set_root_limits_enabled(self, enabled: bool) -> None:
        """Apply the same four-finger limits switch to both hands."""
        forward = getattr(self._inner, "set_root_limits_enabled", None)
        if callable(forward):
            forward(enabled)
        else:
            self.left.set_root_limits_enabled(enabled)
            self.right.set_root_limits_enabled(enabled)

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> "UserBimanualGlove":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


__all__ = [
    "DENIED_FIELDS",
    "UserBimanualFrame",
    "UserBimanualGlove",
    "UserGlove",
    "UserHandFrame",
    "force_newtons",
    "restrict",
    "restrict_bimanual",
]
