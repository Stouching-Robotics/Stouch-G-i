"""Restricted Friston pose and tactile interface for the user SDK."""

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
DENIED_FIELDS = (
    "P",
    "joints_raw_m",
    "joints_smoothed_m",
    "imu_xyzw",
    "imu_valid",
    "sensor_age_s",
    "tactile_raw",
    "tactile_fresh",
    "raw_imu_xyzw",
    "raw_imu_present",
    "raw_imu_valid",
    "raw_device_timestamp_us",
    "mag",
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
    """One hand sample. joint_xyzw contains the wrist and 15 local rotations.

    Row zero is the wrist orientation in calibrated SDK space. Finger rows
    include the rest basis. Pass the array directly to the matching skeleton.
    force_fresh marks a new ADC scan, including when force_n is unavailable.
    """

    side: Side
    sequence: int
    timestamp_us: int
    joint_xyzw: np.ndarray
    tactile_adc: np.ndarray | None
    force_n: np.ndarray | None
    status: HandStatus
    force_fresh: bool = False
    firmware_version: str | None = None
    link_kind: str = "usb"
    device_fps: float = 0.0
    tactile_fps: float = 0.0
    link_drops: dict[str, int] = field(default_factory=dict)
    link_errors: dict[str, int] = field(default_factory=dict)
    latency_ms: float | None = None
    mag_ready: tuple[bool, ...] | None = None
    force_source: str = "unavailable"
    force_status: str = "unread"
    force_error: str = ""

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))
        pose = np.asarray(self.joint_xyzw, dtype=np.float32)
        if pose.shape != (16, 4) or not np.allclose(
                np.linalg.norm(pose, axis=1), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError("joint_xyzw must contain 16 unit xyzw quaternions")
        object.__setattr__(self, "joint_xyzw", pose.copy())
        if self.tactile_adc is not None:
            object.__setattr__(self, "tactile_adc", _array(
                self.tactile_adc, (16, 16), np.float32, "tactile_adc"))
        if self.force_n is not None:
            object.__setattr__(self, "force_n", _array(
                self.force_n, (16, 16), np.float32, "force_n"))


@dataclass(frozen=True)
class UserBimanualFrame:
    """Both hands, each already narrowed to :class:`UserHandFrame`."""

    sequence: int
    timestamp_us: int
    left: UserHandFrame
    right: UserHandFrame
    synchronization_error_ms: float


def restrict(frame: HandFrame, *, repeated: bool = False) -> UserHandFrame:
    """Copy one solved pose and the tactile scan attached to that frame."""
    force = (getattr(frame, "force_n", None)
             if getattr(frame, "force_source", "unavailable") == "device"
             and getattr(frame, "force_status", "unread") == "ready" else None)
    result = UserHandFrame(
        side=frame.side,
        sequence=frame.sequence,
        timestamp_us=frame.timestamp_us,
        joint_xyzw=frame.joint_xyzw,
        tactile_adc=frame.tactile_raw,
        force_n=force,
        status=frame.status,
        force_fresh=bool(frame.tactile_fresh and not repeated),
        force_source=getattr(frame, "force_source", "unavailable"),
        force_status=getattr(frame, "force_status", "unread"),
        force_error=getattr(frame, "force_error", ""),
        firmware_version=frame.firmware_version,
        link_kind=frame.link_kind,
        device_fps=float(frame.device_fps),
        tactile_fps=float(frame.tactile_fps),
        link_drops=dict(frame.link_drops or {}),
        link_errors=dict(frame.link_errors or {}),
        latency_ms=frame.latency_ms,
        mag_ready=frame.mag_ready,
    )
    for name in ("_force_connection_id", "_force_baseline_id"):
        object.__setattr__(result, name, getattr(frame, name, None))
    return result


def restrict_bimanual(frame: BimanualFrame, *, previous_left=None, previous_right=None) -> UserBimanualFrame:
    """Narrow a two-hand frame; each half goes through :func:`restrict`."""

    return UserBimanualFrame(
        sequence=frame.sequence,
        timestamp_us=frame.timestamp_us,
        left=restrict(frame.left, repeated=(frame.left.sequence, frame.left.timestamp_us) == previous_left),
        right=restrict(frame.right, repeated=(frame.right.sequence, frame.right.timestamp_us) == previous_right),
        synchronization_error_ms=frame.synchronization_error_ms,
    )


class UserGlove:
    """Read solved poses and tactile scans for one hand."""

    __slots__ = ("_inner",)

    def __init__(self, inner: Glove) -> None:
        self._inner = inner

    def start(self) -> "UserGlove":
        self._inner.start()
        return self

    def read(self, timeout: float | None = None) -> UserHandFrame:
        """Return the next solved pose with its associated tactile scan."""

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

    @property
    def force_calibration_status(self):
        """Return this connection's device parameters and baseline availability."""
        return self._inner.force_calibration_status

    def read_force_calibration(self, *, timeout_s=5.0):
        """Read device force coefficients; unavailable firmware remains usable."""
        return self._inner.read_force_calibration(timeout_s=timeout_s)

    def set_force_baseline(self, baseline, *, connection_id=None, baseline_id=None):
        """Set an unloaded 16x16 ADC baseline locally for this live connection."""
        return self._inner.set_force_baseline(
            baseline, connection_id=connection_id, baseline_id=baseline_id)

    def set_root_limits_enabled(self, enabled: bool) -> None:
        """Enable the four-finger joint limits used by the desktop viewer."""
        self._inner.set_root_limits_enabled(enabled)

    def set_root_limits_blending_enabled(self, enabled: bool) -> None:
        """Relax finger limits during opposition."""
        self._inner.set_root_limits_blending_enabled(enabled)

    def set_fist_correction_enabled(self, enabled: bool) -> None:
        """Apply the calibrated fist correction near the sampled fist."""
        self._inner.set_fist_correction_enabled(enabled)

    def set_contact_controls(self, enabled: bool, blend: float = 1.0) -> None:
        """Enable contact assistance at the requested blend from zero to one."""
        self._inner.set_contact_controls(enabled, blend)

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

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> "UserGlove":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


class UserBimanualGlove:
    """Restricted two-hand facade.  Same omissions as :class:`UserGlove`."""

    __slots__ = ("_inner", "_last_left", "_last_right")

    def __init__(self, inner: BimanualGlove) -> None:
        self._inner = inner
        self._last_left = self._last_right = None

    def start(self) -> "UserBimanualGlove":
        self._inner.start()
        return self

    def read(self, timeout: float | None = None) -> UserBimanualFrame:
        frame = self._inner.read(timeout)
        pair = restrict_bimanual(frame, previous_left=self._last_left, previous_right=self._last_right)
        self._last_left = (pair.left.sequence, pair.left.timestamp_us)
        self._last_right = (pair.right.sequence, pair.right.timestamp_us)
        return pair

    def health(self):
        return self._inner.health()

    @property
    def left(self) -> UserGlove:
        return UserGlove(self._inner.left)

    @property
    def right(self) -> UserGlove:
        return UserGlove(self._inner.right)

    def close(self) -> None:
        self._inner.close()
        self._last_left = self._last_right = None

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
