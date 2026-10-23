"""Entry point for the replay-only executable (no torch/manotorch).

Two modes:
  - no arguments: open the Qt replay launcher (session picker + render window)
  - with arguments: headless renderer (``Stouch_glove_replay.exe data/foo.parquet``)

This file deliberately re-implements the small windowed bootstrap from
``main.py`` instead of importing it: importing ``main`` would statically pull
in ``gui.live_3d_bimanual_auto`` and therefore torch.
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path


def _fixup_stdio() -> None:
    """Windowed builds set stdout/stderr to None; redirect them to a log file."""
    if not getattr(sys, "frozen", False):
        return
    if sys.stdout is not None and sys.stderr is not None:
        return
    import os
    try:
        log_path = Path(sys.executable).resolve().with_name(
            "stouch_replay_console.log")
        stream = open(log_path, "a", encoding="utf-8", buffering=1)
    except OSError:
        stream = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def _install_crash_log() -> None:
    """Frozen windowed builds have no console; write tracebacks to a log file."""
    if not getattr(sys, "frozen", False):
        return
    import traceback
    from datetime import datetime
    log_path = Path(sys.executable).resolve().with_name("stouch_replay.log")
    real_excepthook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write("=" * 70 + "\n")
                fh.write(f"uncaught exception at {datetime.now().isoformat()}\n")
                fh.writelines(traceback.format_exception(
                    exc_type, exc_value, exc_tb))
                fh.write("\n")
        except OSError:
            pass
        real_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def main(argv: list[str] | None = None) -> int:
    _fixup_stdio()
    _install_crash_log()
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        from gui.rendering.replay import main as replay_main
        return int(replay_main(args) or 0)
    from gui.replay_launcher import run
    return run()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
