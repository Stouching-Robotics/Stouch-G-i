"""BP101Y Bluetooth SPP dongle control (AT commands over its USB CDC port).

The dongle enumerates as a USB CDC serial port (VID 0483 / PID 2013) and
relays the glove's SPP stream to the host.  The glove side cannot re-establish
the SPP link on its own after a power cycle, so the host sends the dongle
``AT+REBOOT``: the dongle replies ``ok``, drops the CDC link and reboots, and
the glove reconnects.
"""

from __future__ import annotations

import time

from common.usb_cdc import BLUETOOTH_DONGLE_PID, BLUETOOTH_DONGLE_VID

DONGLE_BAUDRATE = 115200
_AT_REBOOT = b"AT+REBOOT\r\n"


def find_bluetooth_dongle_port(
        vid: int = BLUETOOTH_DONGLE_VID,
        pid: int = BLUETOOTH_DONGLE_PID,
) -> str | None:
    """Return the first Bluetooth-dongle serial port, or ``None`` when absent."""

    import serial.tools.list_ports

    for port in serial.tools.list_ports.comports():
        if port.vid == int(vid) and port.pid == int(pid):
            return port.device
    return None


def reboot_bluetooth_dongle(
        port: str | None = None,
        timeout_s: float = 3.0,
) -> str:
    """Send ``AT+REBOOT`` and return the dongle's reply (normally ``ok``).

    ``port`` defaults to the first VID 0483 / PID 2013 serial port.  Raises
    ``RuntimeError`` when the dongle is absent or cannot be opened.
    """

    import serial

    target = port or find_bluetooth_dongle_port()
    if not target:
        raise RuntimeError(
            "Bluetooth dongle (VID 0483 / PID 2013) not found on any COM port")
    try:
        handle = serial.Serial(target, baudrate=DONGLE_BAUDRATE, timeout=0.2)
    except (OSError, serial.SerialException) as exc:
        raise RuntimeError(f"cannot open dongle {target}: {exc}") from exc
    try:
        handle.reset_input_buffer()
        handle.write(_AT_REBOOT)
        reply = bytearray()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            chunk = handle.read(handle.in_waiting or 1)
            if not chunk:
                continue
            reply.extend(chunk)
            if b"ok" in bytes(reply).lower():
                break
        return bytes(reply).decode("utf-8", errors="replace").strip()
    finally:
        try:
            handle.close()
        except Exception:
            pass
