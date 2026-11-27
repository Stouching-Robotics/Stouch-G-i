"""Recording-session metadata shared by live capture and offline replay."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SESSION_FILENAME = "session.json"
SESSION_SCHEMA = "stm32-imu-usb-keypoints"
SESSION_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def session_metadata_path(parquet_path: str | os.PathLike) -> Path:
    return Path(parquet_path).resolve().parent / SESSION_FILENAME


def load_session_metadata(parquet_path: str | os.PathLike) -> dict[str, Any]:
    path = session_metadata_path(parquet_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read session metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Session metadata must be a JSON object: {path}")
    return payload


def _merge(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(target)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_session_metadata(
        parquet_path: str | os.PathLike,
        metadata: dict[str, Any]) -> Path:
    parquet = Path(parquet_path).resolve()
    path = session_metadata_path(parquet)
    payload = dict(metadata)
    payload.setdefault("schema", SESSION_SCHEMA)
    payload.setdefault("schema_version", SESSION_SCHEMA_VERSION)
    payload.setdefault("created_at", datetime.now().astimezone().isoformat())
    payload["parquet_file"] = parquet.name
    payload = _jsonable(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)
    return path


def update_session_metadata(
        parquet_path: str | os.PathLike,
        updates: dict[str, Any]) -> Path:
    current = load_session_metadata(parquet_path)
    return write_session_metadata(parquet_path, _merge(current, updates))


def view_state_metadata(view_state) -> dict[str, Any]:
    return {
        "yaw": float(view_state.yaw),
        "elev": float(view_state.elev),
        "roll": float(view_state.roll),
        "dist": float(view_state.dist),
        "pan_px": [float(value) for value in view_state.pan_px],
    }
