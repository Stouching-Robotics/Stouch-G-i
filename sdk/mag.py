"""Type-0x05 magnetometer telemetry: the raw report, and the version gate.

The device sends 0x05 on its own at ~0.5 Hz from firmware v1.2.13 onward.  It is
a **report**, not a request: neither interface document defines a host command
that triggers one, so this module has nothing to send.  It maps the transport's
cached :class:`~common.usb_cdc.MagTelemetrySnapshot` onto the public
:class:`~common.types.MagStatus` and exposes the version question that decides
whether 0x05 exists at all.

What lives here is protocol fact -- which byte is which field, from which
firmware version, and whether enough of this round's channels cleared level 3 to
mean anything.  What does **not** live here is the prompt policy: when to ask the
user to draw circles, how long to watch, how long to show a verdict.  Those are
presentation decisions and belong to the caller.
"""

from __future__ import annotations

from common.types import MagStatus
from common.usb_cdc import (
    MAG_LEVEL_MASK, MAG_LEVEL_SHIFT, MAG_READY_MIN_DENOMINATOR,
    MAG_READY_MIN_NUMERATOR, USB_FULL_SEQUENCE_FROM, MagTelemetrySnapshot,
    display_firmware_version, firmware_real_version, mag_gate_met,
    mag_ready_mask, remap_physical_to_hand)

# Level a channel must reach for its magnetometer to count as calibrated.
MAG_READY_LEVEL = 3

__all__ = [
    "MAG_LEVEL_MASK", "MAG_LEVEL_SHIFT", "MAG_READY_LEVEL",
    "MAG_READY_MIN_DENOMINATOR", "MAG_READY_MIN_NUMERATOR",
    "MagStatus", "display_firmware_version", "firmware_real_version",
    "from_snapshot", "mag_available", "mag_gate_met", "mag_ready_mask",
    "remap_physical_to_hand",
]


def mag_available(firmware_version: str | None) -> bool:
    """Does this firmware emit type-0x05 frames at all?

    ``False`` below v1.2.13.  The check is against the *real* parsed version, not
    the displayed string: :func:`firmware_real_version` is what resolves the
    ``1.2.1`` vs ``1.2.10`` ambiguity that :func:`display_firmware_version`
    deliberately preserves for display.
    """

    real = firmware_real_version(firmware_version)
    return real is not None and real >= USB_FULL_SEQUENCE_FROM


def from_snapshot(snapshot: MagTelemetrySnapshot | None) -> MagStatus | None:
    """Map the transport's cached snapshot onto the public contract.

    ``None`` in, ``None`` out -- an absent 0x05 and a firmware too old to send
    one are the same thing to a caller, and both must read as "no data" rather
    than raising.
    """

    if snapshot is None:
        return None
    return MagStatus(
        sequence=snapshot.sequence,
        device_timestamp_us=snapshot.device_timestamp_us,
        host_timestamp_us=snapshot.host_timestamp_us,
        levels_raw=tuple(snapshot.levels_raw),
        frame_levels_raw=tuple(snapshot.frame_levels_raw),
        fresh=tuple(snapshot.level_fresh),
        seen=tuple(snapshot.level_seen),
        ready=bool(snapshot.ready),
    )


class MagCalibrationDropMonitor:
    """Edges in actual fresh physical-channel levels, independent of pose motion.

    A partial report does not mean a missing channel lost calibration. Repeated
    cached 0x05 reports are ignored. Recovery to 3 arms the next falling edge.
    """
    def __init__(self):
        self._levels = [None] * 16
        self._last_report = None

    def update(self, report: MagStatus | None) -> tuple[tuple[int, int], ...]:
        if report is None:
            return ()
        key = (report.host_timestamp_us, report.device_timestamp_us, report.sequence)
        if key == self._last_report:
            return ()
        self._last_report = key
        dropped = []
        for channel, fresh in enumerate(report.fresh):
            if not fresh:
                continue
            level = (report.frame_levels_raw[channel] >> MAG_LEVEL_SHIFT) & MAG_LEVEL_MASK
            if self._levels[channel] == MAG_READY_LEVEL and level < MAG_READY_LEVEL:
                dropped.append((channel, level))
            self._levels[channel] = level
        return tuple(dropped)
