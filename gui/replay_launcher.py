"""Qt (PySide6) replay launcher window for the replay command.

Run without arguments (``python -m gui.entrypoint replay`` or
``./glove.sh replay``) opens this window instead of printing usage: list the
recorded sessions newest-first, pick view/fps/output, render in a background
thread with the log streamed live into the window, then open the output folder
or play the finished video.  With CLI arguments the replay command keeps its
original headless renderer behavior unchanged.

Importing this module never creates a QApplication (PySide6 classes are inert
until a QApplication exists), so the headless CLI path stays safe and fast.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path


def _sdk_root(start):
    for parent in [start, *start.parents]:
        if (parent / "algorithm").is_dir():
            return parent
    raise RuntimeError("cannot locate SDK root")


# Direct execution (python gui/gui/replay_launcher.py) puts the script's own
# directory on sys.path instead of the project root; fix that before the
# ``gui`` imports below.
if getattr(sys, "frozen", False):
    BUNDLE_ROOT = Path(sys._MEIPASS)
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    BUNDLE_ROOT = PROJECT_ROOT = _sdk_root(Path(__file__).resolve().parent)
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from PySide6.QtCore import QObject, Signal  # noqa: E402

from gui.rendering.replay import main as replay_main  # noqa: E402

_VIDEO_RE = re.compile(r"^Video: (.+?) \(\d+ frames at ")


def _default_data_root() -> Path:
    # PROJECT_ROOT is the exe directory when frozen and the SDK root otherwise;
    # recordings live under <root>/data in both cases.
    return PROJECT_ROOT / "data"


def discover_sessions(data_root: Path) -> list[dict]:
    """Newest-first recorded sessions (every *.parquet under the data root)."""
    root = Path(data_root)
    found: list[dict] = []
    if root.is_dir():
        for parquet in root.rglob("*.parquet"):
            if not parquet.is_file():
                continue
            try:
                mtime = parquet.stat().st_mtime
            except OSError:
                continue
            try:
                rel = parquet.relative_to(root)
            except ValueError:
                rel = parquet
            try:
                ts_label = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except (OSError, ValueError, OverflowError):
                ts_label = "?"
            found.append({
                "parquet": parquet,
                "folder": parquet.parent,
                "mtime": mtime,
                "label": f"{rel}    [{ts_label}]",
            })
    found.sort(key=lambda session: session["mtime"], reverse=True)
    return found


class _Bridge(QObject):
    """Signal relay living in the GUI thread (parented to the dialog).

    The render itself runs in a plain Python thread (no QThread): emitting
    these signals from there is delivered queued to the GUI thread, and there
    is no QThread object whose teardown can crash a frozen exe.
    """

    _log = Signal(str)
    _done = Signal(int)  # exit code (0 = success)


class _Capture:
    """Plain holder for the render's stdout lines (kept off the bridge so the
    bridge's lifetime stays entirely Qt-managed)."""

    def __init__(self):
        self.lines: list[str] = []


def _video_path(lines: list[str]) -> str | None:
    """Output path printed by replay on success (last ``Video:`` line)."""
    for line in reversed(lines):
        match = _VIDEO_RE.search(line.strip())
        if match:
            return match.group(1)
    return None


class _LogSink(io.TextIOBase):
    """stdout/stderr adapter that forwards prints to the bridge's log signal."""

    def __init__(self, bridge: _Bridge, capture: _Capture):
        super().__init__()
        self._bridge = bridge
        self._capture = capture

    def write(self, text: str) -> int:
        for line in text.splitlines():
            try:
                self._bridge._log.emit(line)
            except RuntimeError:
                pass  # dialog closed mid-render; bridge already deleted
        self._capture.lines.append(text)
        return len(text)

    def flush(self) -> None:
        pass


def _render_in_thread(bridge: _Bridge, argv: list[str],
                      capture: _Capture) -> None:
    """Run replay.main in a plain thread, streaming prints via the bridge."""
    sink = _LogSink(bridge, capture)
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        code = 0
        try:
            replay_main(argv)
        except SystemExit as exc:
            code = 0 if exc.code in (0, None) else int(exc.code or 1)
        except Exception:  # noqa: BLE001  surface every failure in the log
            import traceback
            traceback.print_exc(file=sink)
            code = 1
    try:
        bridge._done.emit(code)
    except RuntimeError:
        pass  # dialog closed mid-render


def run(data_root: Path | None = None) -> int:
    """Open the blocking replay launcher window; returns the app exit code."""
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QDialog, QFileDialog, QGridLayout,
        QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
        QPushButton, QVBoxLayout)

    root = Path(data_root) if data_root is not None else _default_data_root()
    app = QApplication.instance() or QApplication([])

    dialog = QDialog()
    dialog.setWindowTitle("Glove Replay 回放渲染")
    # The log already absorbs spare space; permit a standard laptop window and
    # let the layouts reflow instead of placing the lower action/status area
    # outside the visible work area.
    dialog.setMinimumSize(640, 480)

    layout = QVBoxLayout(dialog)
    intro = QLabel("选择录制会话,点击“开始渲染”生成 MP4 视频。")
    layout.addWidget(intro)
    folder_label = QLabel(f"数据目录: {root.resolve()}")
    folder_label.setWordWrap(True)
    folder_label.setStyleSheet("color: rgb(120,120,120)")
    layout.addWidget(folder_label)

    sessions = discover_sessions(root)

    # --- session picker: combo (discovered sessions) + browse (any parquet) ---
    pick_row = QHBoxLayout()
    combo = QComboBox()
    for session in sessions:
        combo.addItem(session["label"], session)
    combo.setEnabled(bool(sessions))
    pick_row.addWidget(combo, 1)
    browse_button = QPushButton("浏览…")
    pick_row.addWidget(browse_button)
    layout.addLayout(pick_row)
    if not sessions:
        hint = QLabel("未在该目录下找到录制数据,请用“浏览…”选择 Parquet 文件。")
        hint.setStyleSheet("color: rgb(240,150,40)")
        layout.addWidget(hint)

    picked_path: Path | None = None
    picked_label = QLabel("自定义文件: (未选择)")
    picked_label.setStyleSheet("color: rgb(120,120,120)")
    layout.addWidget(picked_label)

    # --- options ---
    options = QGridLayout()
    options.addWidget(QLabel("视角"), 0, 0)
    view_combo = QComboBox()
    view_combo.addItem("自动(跟随录制)", None)
    view_combo.addItem("静态", "static")
    view_combo.addItem("旋转", "rotating")
    options.addWidget(view_combo, 0, 1)
    options.addWidget(QLabel("帧率"), 1, 0)
    fps_edit = QLineEdit()
    fps_edit.setPlaceholderText("默认(跟随录制)")
    options.addWidget(fps_edit, 1, 1)
    options.addWidget(QLabel("输出 MP4"), 2, 0)
    out_edit = QLineEdit()
    out_edit.setPlaceholderText("默认: 录制文件旁的 preview.mp4")
    options.addWidget(out_edit, 2, 1)
    out_browse = QPushButton("浏览…")
    options.addWidget(out_browse, 2, 2)
    layout.addLayout(options)

    # --- actions ---
    actions = QHBoxLayout()
    start_button = QPushButton("开始渲染")
    start_button.setDefault(True)
    actions.addWidget(start_button)
    actions.addStretch(1)
    open_folder_button = QPushButton("打开输出文件夹")
    open_folder_button.setEnabled(False)
    play_button = QPushButton("播放视频")
    play_button.setEnabled(False)
    actions.addWidget(open_folder_button)
    actions.addWidget(play_button)
    layout.addLayout(actions)

    log = QPlainTextEdit()
    log.setReadOnly(True)
    log.setMaximumBlockCount(5000)
    layout.addWidget(log, 1)

    status = QLabel("就绪")
    layout.addWidget(status)

    running: list = []  # [_Bridge, _Capture] of the in-flight render, if any
    video_path: str | None = None  # last successful output, for the two buttons

    def current_parquet() -> Path | None:
        if picked_path is not None:
            return picked_path
        data = combo.currentData()
        return data["parquet"] if data else None

    def append_log(line: str) -> None:
        log.appendPlainText(line)

    def finish_render(code: int) -> None:
        nonlocal video_path
        capture = running[1] if running else None
        start_button.setEnabled(True)
        if code == 0 and capture is not None:
            video_path = _video_path(capture.lines)
            if video_path:
                status.setText(f"完成 ✓ 输出: {video_path}")
                status.setStyleSheet("color: rgb(40,210,80)")
                open_folder_button.setEnabled(True)
                play_button.setEnabled(True)
                running.clear()
                return
        video_path = None
        status.setText("渲染失败,详见上方日志")
        status.setStyleSheet("color: rgb(235,70,70)")
        running.clear()

    def open_output_folder() -> None:
        if video_path:
            os.startfile(str(Path(video_path).parent))

    def play_video() -> None:
        if video_path:
            os.startfile(video_path)

    def start_render() -> None:
        parquet = current_parquet()
        if parquet is None:
            QMessageBox.warning(dialog, "未选择数据",
                                "请先从列表选择会话,或点击“浏览…”选择 Parquet 文件。")
            return
        if not parquet.is_file():
            QMessageBox.warning(dialog, "文件不存在", f"找不到文件:\n{parquet}")
            return
        argv = [str(parquet)]
        view = view_combo.currentData()
        if view:
            argv += ["--view", view]
        fps = fps_edit.text().strip()
        if fps:
            argv += ["--fps", fps]
        out = out_edit.text().strip()
        if out:
            argv += ["--out", out]

        log.clear()
        status.setText("渲染中…")
        status.setStyleSheet("")
        start_button.setEnabled(False)
        open_folder_button.setEnabled(False)
        play_button.setEnabled(False)

        bridge = _Bridge(dialog)
        bridge._log.connect(append_log)
        bridge._done.connect(finish_render)
        capture = _Capture()
        running[:] = [bridge, capture]
        threading.Thread(
            target=_render_in_thread, args=(bridge, argv, capture),
            daemon=True).start()

    def pick_file() -> None:
        nonlocal picked_path
        chosen, _ = QFileDialog.getOpenFileName(
            dialog, "选择录制 Parquet", str(root), "Parquet (*.parquet)")
        if not chosen:
            return
        picked_path = Path(chosen)
        picked_label.setText(f"自定义文件: {picked_path}")
        picked_label.setStyleSheet("color: rgb(40,210,80)")

    def pick_output() -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            dialog, "输出 MP4", str(root / "preview.mp4"), "MP4 (*.mp4)")
        if chosen:
            out_edit.setText(chosen)

    start_button.clicked.connect(start_render)
    browse_button.clicked.connect(pick_file)
    out_browse.clicked.connect(pick_output)
    open_folder_button.clicked.connect(open_output_folder)
    play_button.clicked.connect(play_video)

    dialog.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
