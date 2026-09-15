#!/usr/bin/env python3
"""Unified entry point for the STM32 HAND2mm glove toolkit.

Default (no arguments) opens the calibration-file selector for the adaptive
bimanual live-3D viewer, equivalent to the old ``stouch_glove.py`` shortcut::

    python main.py

Passing a known application name dispatches to that application, equivalent to
the old ``python -m apps.entrypoint <app>``::

    python main.py devices
    python main.py calibration
    python main.py replay data/.../chunk-000.parquet

Any other leading argument is forwarded to the live-3D viewer together with
``--select-calibration``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_APPLICATIONS = {
    "devices": ("gui.cli.glove_devices", "main"),
    "imu_live": ("gui.imu_live", "main"),
    "imu_live_bimanual": ("gui.imu_live_bimanual", "main"),
    "calibration": ("gui.imu_calibration_gui", "main"),
    "live_3d": ("gui.live_3d", "main"),
    "live_3d_bimanual": ("gui.live_3d_bimanual", "main"),
    "live_3d_bimanual_auto": ("gui.live_3d_bimanual_auto", "main"),
    "tactile": ("gui.tactile_live", "main"),
    "replay": ("gui.rendering.replay", "main"),
}


def run_application(name: str, argv: list[str] | None = None) -> int:
    if name not in _APPLICATIONS:
        raise ValueError(f"unknown application: {name}")
    module_name, function_name = _APPLICATIONS[name]
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    args = list(argv or [])
    if name == "replay" and not args:
        # No-argument mode: open the Qt replay launcher (session picker +
        # render window) instead of argparse's usage error.
        from gui.replay_launcher import run
        return run()
    if name == "imu_live":
        previous = sys.argv
        try:
            sys.argv = [module_name, *args]
            result = function()
        finally:
            sys.argv = previous
    else:
        result = function(args)
    return int(result or 0)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in _APPLICATIONS:
        return run_application(args[0], args[1:])
    # Default / forward-compat path: adaptive bimanual viewer with selector.
    from gui.live_3d_bimanual_auto import main as auto_main
    return auto_main(["--select-calibration", *args])


if __name__ == "__main__":
    raise SystemExit(main())
