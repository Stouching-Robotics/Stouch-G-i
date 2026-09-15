#!/usr/bin/env python3
"""List or resolve the two STM32 gloves configured in glove_devices.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys._MEIPASS)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glove_io.device_registry import (  # noqa: E402
    DEFAULT_REGISTRY,
    GloveDeviceError,
    clear_glove_bindings,
    list_matching_ports,
    load_device_registry,
    resolve_both_ports,
    resolve_glove_port,
    set_glove_serial,
    swap_glove_serials,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list all STM32 CDC devices")
    group.add_argument("--resolve", choices=("left", "right"), help="print only the serial port for the given side")
    group.add_argument("--check-both", action="store_true", help="validate and print the left/right hand binding")
    group.add_argument("--set-serial", choices=("left", "right"),
                       help="bind the given side to the USB serial number from --serial")
    group.add_argument("--swap", action="store_true", help="swap the left/right USB serial number bindings")
    group.add_argument("--clear", action="store_true", help="forget every remembered glove serial")
    parser.add_argument("--serial", help="used together with --set-serial")
    args = parser.parse_args(argv)
    try:
        if args.list:
            registry = load_device_registry(args.registry)
            serial_to_side = {
                serial: side
                for side in ("left", "right")
                for serial in registry[side]["usb_serials"]
            }
            for port in list_matching_ports():
                serial_number = str(port.serial_number or "").upper()
                side = serial_to_side.get(serial_number, "unbound")
                print(
                    f"{side:7s} {port.device:12s} serial={serial_number or '-'} "
                    f"location={port.location or '-'}")
            return 0
        if args.resolve:
            print(resolve_glove_port(args.resolve, args.registry))
            return 0
        if args.set_serial:
            if not args.serial:
                raise GloveDeviceError("--set-serial requires --serial")
            path = set_glove_serial(
                args.set_serial, args.serial, args.registry)
            print(f"[OK] {args.set_serial} serial={args.serial.upper()} -> {path}")
            return 0
        if args.swap:
            path = swap_glove_serials(args.registry)
            print(f"[OK] swapped left/right serial bindings -> {path}")
            return 0
        if args.clear:
            path = clear_glove_bindings(args.registry)
            print(f"[OK] cleared all glove serial bindings -> {path}")
            return 0
        for side, port in resolve_both_ports(args.registry).items():
            print(f"{side}={port}")
        return 0
    except GloveDeviceError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
