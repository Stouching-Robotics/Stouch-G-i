"""Filesystem locations shared by protected implementation modules."""

from pathlib import Path
import sys

def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")

SDK_ROOT = (Path(sys._MEIPASS) if getattr(sys, "frozen", False)
            else _sdk_root(Path(__file__).resolve().parent))
