"""Low-rate, downsampled diagnostics for glove calibration packets.

The recorder keeps only the newest pending packet and its offer-time context
for each hand.  Receiver threads call :meth:`offer`; one writer thread owns all
file writes, flushes, and close operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid


SCHEMA_ID = "stouch.calibration-diagnostic"
SCHEMA_VERSION = 1
MAX_RATE_HZ = 4.0
STOP_TIMEOUT_S = 0.5
SIDES = ("left", "right")


def _json_value(value: Any) -> Any:
    """Copy a value into JSON-compatible objects without retaining arrays."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, os.PathLike):
        return os.fspath(value)

    # NumPy arrays/scalars expose these methods.  Avoid importing NumPy only
    # for this conversion so the recorder remains standalone.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    raise TypeError(f"diagnostic value is not JSON-compatible: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sample_allowed(
        now_s: float,
        last_sample_s: float | None,
        rate_hz: float,
) -> bool:
    """Return whether a global sample may start at ``now_s``."""

    if last_sample_s is None:
        return True
    return now_s - last_sample_s >= (1.0 / rate_hz) - 1e-9


class CalibrationDiagnosticRecorder:
    """Record latest raw calibration packets as a downsampled JSONL diagnostic."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._active = False
        self._path: Path | None = None
        self._error: str | None = None
        self._written_frames = 0
        self._sample_index = 0
        self._rate_hz = MAX_RATE_HZ
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._writer_ready = threading.Event()
        self._worker_finished = threading.Event()
        self._last_write_monotonic: float | None = None
        self._pending: dict[
            str, tuple[dict[str, Any], dict[str, Any] | None]
        ] = {}
        self._last_offered_key: dict[str, str] = {}
        self._contexts: dict[str, dict[str, Any] | None] = {
            side: None for side in SIDES
        }
        self._write_failed = False

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._active,
                "path": self._path,
                "error": self._error,
                "written_frames": self._written_frames,
            }

    def start(
            self,
            directory: str | os.PathLike[str],
            *,
            rate_hz: float = MAX_RATE_HZ,
            metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Start a new diagnostic writer without performing file I/O here."""

        directory_path = Path(directory)
        with self._lifecycle_lock:
            with self._lock:
                if self._active:
                    return self._path  # type: ignore[return-value]
                previous = self._thread
                if previous is not None:
                    if previous.is_alive():
                        fallback = (
                            self._path
                            or directory_path
                            / "calibration_diagnostics_unavailable.jsonl"
                        )
                        self._error = (
                            "diagnostic start blocked: previous writer is still "
                            "stopping"
                        )
                        return fallback
                    if self._stream is not None:
                        fallback = (
                            self._path
                            or directory_path
                            / "calibration_diagnostics_unavailable.jsonl"
                        )
                        self._error = (
                            "diagnostic start blocked: previous writer did not "
                            "release its stream"
                        )
                        return fallback
                    self._thread = None

            try:
                requested_rate = float(rate_hz)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rate_hz must be a finite number in (0, 4]") from exc
            if (not math.isfinite(requested_rate)
                    or requested_rate <= 0.0
                    or requested_rate > MAX_RATE_HZ):
                raise ValueError("rate_hz must be a finite number in (0, 4]")

            if metadata is None:
                metadata_copy: dict[str, Any] = {}
            elif not isinstance(metadata, Mapping):
                failed_path = (
                    directory_path
                    / "calibration_diagnostics_unavailable.jsonl"
                )
                with self._lock:
                    self._active = False
                    self._path = failed_path
                    self._error = (
                        "diagnostic start failed: metadata must be a mapping or None"
                    )
                    self._written_frames = 0
                return failed_path
            else:
                try:
                    metadata_copy = _json_value(metadata)
                except Exception as exc:
                    failed_path = (
                        directory_path
                        / "calibration_diagnostics_unavailable.jsonl"
                    )
                    with self._lock:
                        self._active = False
                        self._path = failed_path
                        self._error = f"diagnostic start failed: {exc}"
                        self._written_frames = 0
                    return failed_path

            candidate = self._new_path(directory_path)
            with self._lock:
                self._active = True
                self._path = candidate
                self._error = None
                self._written_frames = 0
                self._sample_index = 0
                self._rate_hz = requested_rate
                self._stream = None
                self._stop_event = threading.Event()
                self._writer_ready = threading.Event()
                self._worker_finished = threading.Event()
                self._last_write_monotonic = None
                self._pending = {}
                self._last_offered_key = {}
                self._contexts = {side: None for side in SIDES}
                self._write_failed = False
                thread = threading.Thread(
                    target=self._writer_loop,
                    args=(
                        candidate,
                        metadata_copy,
                        requested_rate,
                    ),
                    name="calibration-diagnostics",
                    daemon=True,
                )
                self._thread = thread

            try:
                thread.start()
            except Exception as exc:
                with self._lock:
                    self._active = False
                    self._error = f"diagnostic writer start failed: {exc}"
                    self._thread = None
            return candidate

    def set_context(self, side: str, context: dict[str, Any]) -> None:
        """Replace the latest calibration context for one hand."""

        with self._lock:
            if not self._active:
                return
        try:
            if side not in SIDES:
                raise ValueError(f"unknown diagnostic side: {side!r}")
            if not isinstance(context, Mapping):
                raise TypeError("context must be a mapping")
            context_copy = _json_value(context)
        except Exception as exc:
            self._note_error(f"diagnostic context ignored: {exc}")
            return

        with self._lock:
            if self._active:
                self._contexts[side] = context_copy

    def offer(self, side: str, packet: dict[str, Any]) -> bool:
        """Pair and store one packet with the context current at offer time."""

        with self._lock:
            if not self._active or self._write_failed:
                return False
        try:
            if side not in SIDES:
                raise ValueError(f"unknown diagnostic side: {side!r}")
            if not isinstance(packet, Mapping):
                raise TypeError("packet must be a mapping")
            packet_copy = _json_value(packet)
            packet_key = _canonical_json(packet_copy)
        except Exception as exc:
            self._note_error(f"diagnostic packet ignored: {exc}")
            return False

        with self._lock:
            if not self._active or self._write_failed:
                return False
            if self._last_offered_key.get(side) == packet_key:
                return False
            # Assign packet and context in one critical section.  Later UI tick
            # context updates cannot relabel a packet offered earlier.
            self._pending[side] = (packet_copy, self._contexts[side])
            self._last_offered_key[side] = packet_key
            return True

    def stop(self) -> None:
        """Wait at most ``STOP_TIMEOUT_S`` without touching writer-owned I/O."""

        with self._lifecycle_lock:
            with self._lock:
                thread = self._thread
                self._active = False
                self._stop_event.set()

            if thread is None:
                return
            if thread is threading.current_thread():
                with self._lock:
                    self._error = (
                        "diagnostic stop called from its writer thread; "
                        "the writer must exit cooperatively"
                    )
                return

            thread.join(timeout=STOP_TIMEOUT_S)
            if thread.is_alive():
                with self._lock:
                    self._error = (
                        f"diagnostic stop timed out after {STOP_TIMEOUT_S:g}s; "
                        "writer thread is still running"
                    )
                return

            with self._lock:
                if self._stream is not None:
                    self._error = (
                        "diagnostic writer ended but did not release its stream"
                    )
                    return
                self._thread = None

    def _note_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def _new_path(self, directory: Path) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        return directory / (
            f"calibration_diagnostics_{stamp}_{os.getpid()}_"
            f"{uuid.uuid4().hex[:8]}.jsonl"
        )

    def _open_exclusive(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, "x", encoding="utf-8", newline="\n")

    def _writer_loop(
            self,
            candidate: Path,
            metadata: dict[str, Any],
            rate_hz: float,
    ) -> None:
        stream = None
        try:
            stream = self._open_exclusive(candidate)
            with self._lock:
                if threading.current_thread() is not self._thread:
                    raise RuntimeError("diagnostic stream owner is not the writer")
                self._stream = stream
                self._path = candidate

            header = {
                "type": "header",
                "schema": SCHEMA_ID,
                "schema_version": SCHEMA_VERSION,
                "created_at_us": time.time_ns() // 1_000,
                "requested_rate_hz": rate_hz,
                "sampling": "global_latest_packet_per_hand",
                "downsampled": True,
                "full_stream": False,
                "time_basis": (
                    "writer_time_us is UTC epoch microseconds; packet "
                    "timestamp_us and sequence remain unchanged"
                ),
                "encoding": {
                    "missing_hand": None,
                    "nonfinite_number": None,
                    "bytes": "lowercase hex",
                },
                "metadata": metadata,
            }
            stream.write(_canonical_json(header) + "\n")
            stream.flush()
            self._writer_ready.set()

            period_s = 1.0 / rate_hz
            while (not self._stop_event.is_set()
                   and not self._write_failed):
                self._stop_event.wait(period_s)
                if self._stop_event.is_set():
                    break
                self._emit_pending()

            self._flush_pending_on_stop(period_s)
        except Exception as exc:
            self._fail_writer(f"diagnostic writer failed: {exc}")
        finally:
            self._writer_ready.set()
            if stream is not None:
                self._writer_flush_and_close(stream)
            self._worker_finished.set()

    def _flush_pending_on_stop(self, period_s: float) -> None:
        with self._lock:
            has_pending = bool(self._pending)
            last_sample_s = self._last_write_monotonic
        if not has_pending or self._write_failed:
            return

        now_s = time.monotonic()
        if last_sample_s is not None:
            remaining_s = period_s - (now_s - last_sample_s)
            if remaining_s > STOP_TIMEOUT_S:
                with self._lock:
                    self._error = (
                        "diagnostic stop discarded a pending packet before "
                        "the next requested sample time"
                    )
                return
            if remaining_s > 0.0:
                time.sleep(remaining_s)
        self._emit_pending()

    def _emit_pending(self) -> bool:
        if threading.current_thread() is not self._thread:
            raise RuntimeError("diagnostic sampling must run on the writer")
        record = self._take_pending()
        if record is None:
            return False
        try:
            self._append_record(record)
        except Exception as exc:
            self._fail_writer(f"diagnostic write failed: {exc}")
            return False

        with self._lock:
            self._sample_index = int(record["sample_index"])
            self._written_frames += 1
        return True

    def _take_pending(
            self,
            *,
            now_s: float | None = None,
            wall_time_us: int | None = None,
    ) -> dict[str, Any] | None:
        """Take a rate-limited record; pure state work used by tests/writer."""

        sample_time_s = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            if self._write_failed or not self._pending:
                return None
            if not _sample_allowed(
                    sample_time_s, self._last_write_monotonic, self._rate_hz):
                return None

            pending = {
                side: self._pending.pop(side)
                for side in SIDES
                if side in self._pending
            }
            sample_index = self._sample_index + 1
            writer_time_us = (
                int(wall_time_us)
                if wall_time_us is not None
                else time.time_ns() // 1_000
            )
            record = {
                "type": "sample",
                "schema": SCHEMA_ID,
                "schema_version": SCHEMA_VERSION,
                "sample_index": sample_index,
                "writer_time_us": writer_time_us,
                "hands": {
                    side: (
                        {
                            "packet": pending[side][0],
                            "context": pending[side][1],
                        }
                        if side in pending else None
                    )
                    for side in SIDES
                },
            }
            self._last_write_monotonic = sample_time_s
            return record

    def _append_record(self, record: dict[str, Any]) -> None:
        if threading.current_thread() is not self._thread:
            raise RuntimeError("diagnostic file writes must run on the writer")
        with self._lock:
            stream = self._stream
        if stream is None:
            raise OSError("diagnostic stream is closed")
        stream.write(_canonical_json(record) + "\n")
        stream.flush()

    def _fail_writer(self, message: str) -> None:
        with self._lock:
            self._error = message
            self._write_failed = True
            self._active = False

    def _writer_flush_and_close(self, stream) -> None:
        if threading.current_thread() is not self._thread:
            raise RuntimeError("diagnostic flush/close must run on the writer")
        try:
            stream.flush()
        except Exception as exc:
            self._note_error(f"diagnostic flush failed: {exc}")
        try:
            stream.close()
        except Exception as exc:
            self._note_error(f"diagnostic close failed: {exc}")
        finally:
            with self._lock:
                if self._stream is stream:
                    self._stream = None
