"""Versioned public data contracts used by every SDK interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from common.usb_cdc import (
    MAG_LEVEL_MASK, MAG_LEVEL_SHIFT, mag_gate_met, mag_ready_mask)


Side = Literal["left", "right"]

# Physical links a glove stream can arrive over.  "auto" is a selection policy,
# not a link: it resolves to one of the two concrete kinds at open time.
LINK_AUTO = "auto"
LINK_USB = "usb"
LINK_BLUETOOTH = "bluetooth"
_LINK_KINDS = (LINK_AUTO, LINK_USB, LINK_BLUETOOTH)


def _side(value: str) -> Side:
    normalized = str(value).strip().lower()
    if normalized not in ("left", "right"):
        raise ValueError(f"side must be left or right, got {value!r}")
    return normalized  # type: ignore[return-value]


def _array(value, shape, dtype, name):
    result = np.asarray(value, dtype=dtype).reshape(shape).copy()
    if not np.all(np.isfinite(result)) and result.dtype.kind == "f":
        # NaN is meaningful for sensor age/tactile absence, so callers decide
        # whether it is acceptable. Shape and ownership are enforced here.
        pass
    return result


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    serial_number: str
    vid: int
    pid: int
    location: str = ""
    bound_side: Side | None = None
    # "usb" for a directly wired glove, "bluetooth" for a glove reached through
    # the BP101Y dongle.  The wire protocol is identical; only the USB PID and
    # the meaning of ``serial_number`` (glove vs dongle) differ.
    link_kind: str = "usb"


@dataclass(frozen=True)
class DeviceBindings:
    left_serial: str
    right_serial: str
    left_port: str | None = None
    right_port: str | None = None
    # Full remembered lists (schema v2); the *_serial fields remain the primary
    # (most recently bound) entry for backwards compatibility.
    left_serials: tuple[str, ...] = ()
    right_serials: tuple[str, ...] = ()


@dataclass(frozen=True)
class GloveConfig:
    side: Side
    calibration: Path | str
    serial_number: str | None = None
    registry: Path | str | None = None
    sample_rate_hz: int = 80
    include_tactile: bool = True
    smoothing_ms: float = 35.0
    recording_root: Path | str | None = None
    # "auto" picks whichever link currently carries this hand; "usb" / "bluetooth"
    # pin it explicitly.  Both run the same wire protocol -- see sdk.transport.
    link: str = "auto"

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "calibration", Path(self.calibration))
        if self.registry is not None:
            object.__setattr__(self, "registry", Path(self.registry))
        if self.recording_root is not None:
            object.__setattr__(self, "recording_root", Path(self.recording_root))
        if int(self.sample_rate_hz) <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if float(self.smoothing_ms) < 0.0:
            raise ValueError("smoothing_ms must not be negative")
        if str(self.link).strip().lower() not in _LINK_KINDS:
            raise ValueError(
                f"link must be one of {', '.join(_LINK_KINDS)}, got {self.link!r}")
        object.__setattr__(self, "link", str(self.link).strip().lower())


@dataclass(frozen=True)
class BimanualConfig:
    left: GloveConfig
    right: GloveConfig
    output_root: Path | str | None = None
    display_separation_m: float = 0.30

    def __post_init__(self):
        if self.left.side != "left" or self.right.side != "right":
            raise ValueError("BimanualConfig requires left and right GloveConfig objects")
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root))
        if float(self.display_separation_m) <= 0.0:
            raise ValueError("display_separation_m must be positive")


@dataclass(frozen=True)
class DeviceHealth:
    connected: bool
    ready_imu_count: int
    missing_imus: tuple[str, ...] = ()
    stream_fps: float = 0.0
    message: str = ""
    last_error: str | None = None


@dataclass(frozen=True)
class BimanualHealth:
    left: DeviceHealth
    right: DeviceHealth
    synchronization_error_ms: float | None = None

    @property
    def connected(self) -> bool:
        return self.left.connected and self.right.connected


@dataclass(frozen=True)
class HandStatus:
    safety_intervened: bool = False
    active_contact: str | None = None
    kinematics: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawImuFrame:
    """One decoded STM32 IMU packet in physical channel order.

    The quaternion order is XYZW.  No physical-to-hand remapping, axis
    conversion, filtering, calibration, or hand-pose solving has been applied.
    """

    sequence: int
    device_timestamp_us: int
    host_timestamp_us: int
    quaternions_xyzw: np.ndarray
    present_mask: np.ndarray
    valid_mask: np.ndarray
    mag_ready: bool = True

    def __post_init__(self):
        object.__setattr__(self, "quaternions_xyzw", _array(
            self.quaternions_xyzw, (16, 4), np.float64,
            "quaternions_xyzw"))
        object.__setattr__(self, "present_mask", _array(
            self.present_mask, (16,), bool, "present_mask"))
        object.__setattr__(self, "valid_mask", _array(
            self.valid_mask, (16,), bool, "valid_mask"))


@dataclass(frozen=True)
class ImuFrame:
    sequence: int
    timestamp_us: int
    quaternions_xyzw: np.ndarray
    valid_mask: np.ndarray
    sensor_age_s: np.ndarray
    rejected_mask: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "quaternions_xyzw", _array(
            self.quaternions_xyzw, (16, 4), np.float64, "quaternions_xyzw"))
        object.__setattr__(self, "valid_mask", _array(
            self.valid_mask, (16,), bool, "valid_mask"))
        object.__setattr__(self, "sensor_age_s", _array(
            self.sensor_age_s, (16,), np.float32, "sensor_age_s"))
        object.__setattr__(self, "rejected_mask", _array(
            self.rejected_mask, (16,), bool, "rejected_mask"))


@dataclass(frozen=True)
class TactileFrame:
    sequence: int
    timestamp_us: int
    samples: np.ndarray
    processed: bool = False
    # Host ``perf_counter`` when the reader thread queued this frame, or ``0.0``
    # when the latency probe is off.  Unlike ``timestamp_us``, this is host
    # time and therefore shares a clock with the solver and the parent, which is
    # what makes it subtractable across processes; see :mod:`common.latency_probe`.
    host_enqueue_us: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "samples", _array(
            self.samples, (16, 16), np.float32, "samples"))

    @property
    def scan_time_us(self) -> int:
        """Firmware scan-time field retained as a descriptive alias."""

        return self.timestamp_us


@dataclass(frozen=True)
class SolvedHandFrame:
    side: Side
    joints_m: np.ndarray
    status: HandStatus

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "joints_m", _array(
            self.joints_m, (21, 3), np.float32, "joints_m"))


@dataclass(frozen=True)
class KeypointFrame:
    """Versioned output of the protected raw-IMU to 21-keypoint pipeline."""

    side: Side
    sequence: int
    timestamp_us: int
    joints_m: np.ndarray
    imu_xyzw: np.ndarray
    valid_mask: np.ndarray
    sensor_age_s: np.ndarray
    rejected_mask: np.ndarray
    status: HandStatus

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "joints_m", _array(
            self.joints_m, (21, 3), np.float32, "joints_m"))
        object.__setattr__(self, "imu_xyzw", _array(
            self.imu_xyzw, (16, 4), np.float64, "imu_xyzw"))
        object.__setattr__(self, "valid_mask", _array(
            self.valid_mask, (16,), bool, "valid_mask"))
        object.__setattr__(self, "sensor_age_s", _array(
            self.sensor_age_s, (16,), np.float32, "sensor_age_s"))
        object.__setattr__(self, "rejected_mask", _array(
            self.rejected_mask, (16,), bool, "rejected_mask"))

    @property
    def keypoints_21_m(self) -> np.ndarray:
        """Descriptive alias for :attr:`joints_m`."""

        return self.joints_m.copy()


@dataclass(frozen=True)
class MagStatus:
    """One type-0x05 magnetometer calibration report, in **physical** order.

    Index *i* is physical IMU *i* for both hands.  This is unlike the type-0x02
    joint attitude frame, which the firmware hand-remaps: 0x05 is explicitly not
    remapped, so anything comparing these levels against a joint or a finger
    must map through :func:`common.usb_cdc.remap_physical_to_hand` itself.

    The frame is a report, not a request -- the device emits it on its own at
    ~0.5 Hz from firmware v1.2.13 on, and there is no host-side command that
    triggers it.  Below that version ``sdk.Glove`` never produces one of these,
    so every ``HandFrame.mag_*`` accessor is ``None``.

    ``levels`` is the MAG field only (0..3); the byte the firmware sends is the
    BNO055 CALIB_STAT register verbatim, whose upper pairs are SYS/GYR/ACC.
    :attr:`levels` is the only thing that should read the raw byte; a caller
    asking the plain question "is this IMU initialized" wants
    :attr:`common.types.HandFrame.mag_ready`, which collapses it to 0/1 and
    reorders it into hand order.
    """

    sequence: int
    device_timestamp_us: int
    host_timestamp_us: int
    # Cache: each channel keeps its last reported level when a later frame skips
    # it, so a ``None`` here means "never reported", not "reported as nothing".
    levels_raw: tuple[int | None, ...]
    # The newest frame verbatim, paired with the mask it arrived under.  Unlike
    # ``levels_raw`` this never merges, so it is what an audit of the byte
    # layout reads; the merged cache cannot answer that question.
    frame_levels_raw: tuple[int, ...]
    fresh: tuple[bool, ...]
    seen: tuple[bool, ...]
    # Stored rather than derived: a frame carrying no channels leaves the
    # verdict where it was, which a computation over this frame alone cannot
    # reproduce.
    ready: bool = False

    @property
    def levels(self) -> tuple[int | None, ...]:
        """The MAG calibration level of each channel, 0..3, or ``None``."""

        return tuple(None if value is None
                     else (value >> MAG_LEVEL_SHIFT) & MAG_LEVEL_MASK
                     for value in self.levels_raw)

    @property
    def fresh_count(self) -> int:
        """How many channels the newest frame -- this round -- carried."""

        return sum(1 for value in self.fresh if value)

    @property
    def fresh_ready_count(self) -> int:
        """Of those, how many report MAG level 3."""

        return sum(1 for value, level in zip(self.fresh, self.levels)
                   if value and level == 3)

    @property
    def gate_counts(self) -> tuple[int, int]:
        """``(at level 3, carried)`` among the channels this round reported."""

        return self.fresh_ready_count, self.fresh_count

    @property
    def gate_met(self) -> bool:
        """Is this round's magnetometer good enough?

        The order of :attr:`gate_counts` is the order
        :func:`common.usb_cdc.mag_gate_met` takes, which is easy to get
        backwards at a call site -- and getting it backwards reads as a
        permanently uncalibrated glove rather than as an error.  Callers should
        prefer this property.
        """

        return mag_gate_met(*self.gate_counts)

    @property
    def unread_channels(self) -> tuple[int, ...]:
        """The channels this round did not report, in physical order.

        Not a fault and not a level: the device carries about twelve of the
        sixteen per frame, and which ones is not predictable.
        """

        return tuple(channel for channel, value in enumerate(self.fresh)
                     if not value)

    @property
    def ready_count(self) -> int:
        """Channels ever seen that report level 3 -- over the cache, all 16."""

        return sum(1 for present, level in zip(self.seen, self.levels)
                   if present and level == 3)

    @property
    def uncalibrated_channels(self) -> tuple[int, ...]:
        """The physical IMUs that were read and are still below level 3."""

        return tuple(channel for channel, (present, level) in enumerate(
            zip(self.seen, self.levels)) if present and level != 3)


@dataclass(frozen=True)
class GloveInfo:
    """Identity of one open glove: what is running it, and where it hangs.

    ``sdk_version`` and ``firmware_version`` are independent -- a Bluetooth
    glove at firmware ``"2.0.7"`` under SDK ``"2.2.1"`` is ordinary, and the two
    strings must never be shown as one number.

    ``firmware_version`` is the wire string verbatim, and the firmware encodes
    the hand side in the ones digit of its patch: a right glove on v2.0.7
    reports ``"2.0.72"`` and a left one ``"2.0.71"``.  It is therefore not the
    name of a released firmware.  Show
    :func:`sdk.mag.display_firmware_version` of it, and compare versions with
    :func:`sdk.mag.firmware_real_version` rather than by string.
    """

    side: Side
    sdk_version: str
    firmware_version: str | None
    link_kind: str
    port: str
    serial_number: str
    mag_available: bool
    backend: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))


@dataclass(frozen=True)
class HandFrame:
    side: Side
    sequence: int
    timestamp_us: int
    imu_xyzw: np.ndarray
    imu_valid: np.ndarray
    sensor_age_s: np.ndarray
    joints_raw_m: np.ndarray
    joints_smoothed_m: np.ndarray
    # Raw 12-bit ADC only.  The SDK does not preprocess pressure -- filtering
    # and force fitting are presentation concerns and live in gui/.  A caller
    # that wants a processed map runs gui.tactile_processing.TactilePreprocessor
    # over ``tactile_raw`` itself, and records both.
    tactile_raw: np.ndarray | None
    status: HandStatus
    # Whether ``tactile_raw`` is a scan the device sent for this frame, as
    # opposed to the newest one still standing.  Attitude and pressure are
    # separate device streams, so a pressure matrix is re-attached to every
    # attitude frame that runs while it is the newest -- several of them for a
    # caller filtering over time.  Feeding those repeats to a temporal filter
    # shortens its time constant and finishes its warm-up early, so a caller
    # that processes pressure should gate on this flag.
    tactile_fresh: bool = False
    raw_imu_xyzw: np.ndarray | None = None
    raw_imu_present: np.ndarray | None = None
    raw_imu_valid: np.ndarray | None = None
    raw_device_timestamp_us: int | None = None
    # Identity and link diagnostics.  These travel on the frame rather than
    # being read back off the stream so a recorded or replayed frame still
    # carries them, and so a consumer needs one object instead of two.
    #
    # ``firmware_version`` is the wire string verbatim, and the firmware encodes
    # the hand side in the ones digit of its patch: a right glove on v2.0.7
    # reports "2.0.72", a left one "2.0.71".  Printing it as-is therefore shows a
    # number that matches no released firmware name.  For display use
    # ``sdk.mag.display_firmware_version`` (drops the digit -> "2.0.7"); use
    # ``sdk.mag.firmware_hand_side`` (or ``sdk.mag.mag_available``) when the
    # question is a property of the firmware rather than its name.
    firmware_version: str | None = None
    link_kind: str = "usb"
    mag: MagStatus | None = None
    # Device arrival rate, not the caller's consume rate: frames the host
    # actually received per wall-clock second, over a one-second window.  The
    # two diverge on a Bluetooth link, which hands over a whole batch per RF
    # event, and a host that cannot keep up must not look like a glove that is
    # not delivering.
    device_fps: float = 0.0
    # Pressure-matrix arrival rate on the same one-second host-side window, and
    # separate from ``device_fps`` for the reason ``device_fps`` is separate from
    # the caller's consume rate: attitude and pressure are two independent device
    # streams.  A pressure scan is re-attached to every attitude frame that runs
    # while it is the newest, so a caller showing the panel's own refresh rate
    # wants this number rather than the frame rate it happens to be riding on.
    tactile_fps: float = 0.0
    link_drops: dict[str, int] = field(default_factory=dict)
    link_errors: dict[str, int] = field(default_factory=dict)
    # Host<->device round trip, refreshed in the background; ``None`` until the
    # first answer arrives, and permanently on firmware older than v1.2.11 or on
    # a Bluetooth link, where the probe is suppressed (half duplex).
    latency_ms: float | None = None

    @property
    def mag_ready(self) -> tuple[bool, ...] | None:
        """Is each IMU's magnetometer initialized -- 16 flags in **hand order**.

        Index *i* is MANO joint *i*, the same order :attr:`imu_xyzw` and
        :attr:`joints_raw_m` use (0 = wrist, 1-3 = thumb, 4-6 = index, 7-9 =
        middle, 10-12 = ring, 13-15 = little), **not** the physical channel order
        the 0x05 frame itself carries -- the reorder happens here.  ``True``
        means the channel reported MAG calibration level 3; 0, 1, 2 and a channel
        that has not reported at all are all ``False``.  There is deliberately no
        third state: see :class:`MagStatus` for *which* channels have not been
        read (``unread_channels``), if that distinction matters.

        The reorder uses the firmware's default channel map
        (:data:`common.usb_cdc.DEFAULT_CHANNEL_TO_HAND`), the same one every
        calibration file in the tree resolves to.  A hand-edited calibration
        carrying a custom ``channel_to_hand`` feeds the type-0x02 solver but
        cannot reach this property -- a frame does not carry a config -- so on
        such a device these flags and :attr:`imu_xyzw` would be indexed
        differently.

        ``None`` means no 0x05 was ever received -- firmware below v1.2.13,
        which is the only firmware that sends none -- and is not the same as
        "16 IMUs uninitialized".
        """

        return None if self.mag is None else mag_ready_mask(self.mag.levels)

    @property
    def mag_fresh(self) -> tuple[bool, ...] | None:
        """Which physical channels the newest 0x05 frame carried, or ``None``."""

        return None if self.mag is None else self.mag.fresh

    @property
    def mag_frame_levels_raw(self) -> tuple[int, ...] | None:
        """The newest 0x05 frame's CALIB_STAT bytes, verbatim, or ``None``."""

        return None if self.mag is None else self.mag.frame_levels_raw

    def __post_init__(self):
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "imu_xyzw", _array(
            self.imu_xyzw, (16, 4), np.float64, "imu_xyzw"))
        object.__setattr__(self, "imu_valid", _array(
            self.imu_valid, (16,), bool, "imu_valid"))
        object.__setattr__(self, "sensor_age_s", _array(
            self.sensor_age_s, (16,), np.float32, "sensor_age_s"))
        object.__setattr__(self, "joints_raw_m", _array(
            self.joints_raw_m, (21, 3), np.float32, "joints_raw_m"))
        object.__setattr__(self, "joints_smoothed_m", _array(
            self.joints_smoothed_m, (21, 3), np.float32, "joints_smoothed_m"))
        if self.raw_imu_xyzw is not None:
            object.__setattr__(self, "raw_imu_xyzw", _array(
                self.raw_imu_xyzw, (16, 4), np.float64, "raw_imu_xyzw"))
        if self.raw_imu_present is not None:
            object.__setattr__(self, "raw_imu_present", _array(
                self.raw_imu_present, (16,), bool, "raw_imu_present"))
        if self.raw_imu_valid is not None:
            object.__setattr__(self, "raw_imu_valid", _array(
                self.raw_imu_valid, (16,), bool, "raw_imu_valid"))
        if self.tactile_raw is not None:
            object.__setattr__(self, "tactile_raw", _array(
                self.tactile_raw, (16, 16), np.float32, "tactile_raw"))


@dataclass(frozen=True)
class BimanualFrame:
    sequence: int
    timestamp_us: int
    left: HandFrame
    right: HandFrame
    synchronization_error_ms: float


@dataclass(frozen=True)
class CalibrationStep:
    id: str
    title: str
    action: Literal["capture", "compute"]
    stage: Literal["root", "installation", "contact"]


@dataclass(frozen=True)
class CalibrationStepResult:
    step_id: str
    ok: bool
    message: str


@dataclass(frozen=True)
class CalibrationProgress:
    current_index: int
    total_steps: int
    completed_step_ids: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class RecordingStatus:
    state: Literal["disabled", "idle", "recording", "saved"]
    take_index: int
    frame_count: int
    output_path: Path | None = None


@dataclass(frozen=True)
class RecordingResult:
    parquet_path: Path | None
    session_path: Path | None
    take_index: int
    frame_count: int
    duration_s: float
    sample_fps: float


@dataclass(frozen=True)
class RecordingInfo:
    parquet_path: Path
    session_path: Path | None
    mode: str
    sides: tuple[Side, ...]
    frame_count: int
    sample_fps: float
    has_imu: bool
    has_tactile: bool
    tactile_valid_frames: int = 0
    tactile_nonzero_samples: int = 0
    has_raw_imu: bool = False
    has_raw_tactile: bool = False

    @property
    def has_tactile_signal(self) -> bool:
        return self.tactile_nonzero_samples > 0


@dataclass(frozen=True)
class ReplayResult:
    input_path: Path
    output_path: Path
    frame_count: int
    sample_fps: float
