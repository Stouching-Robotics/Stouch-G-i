"""Compile a human-editable pressure calibration CSV into glove parameter sets.

The test reports use the equation ``response_adc = slope * force_N + offset``.
The compact host format stores the three fitted regions for each of six sets
as Q12 k/b pairs.  Firmware v1.2.16(2) treats the resulting block as opaque;
it only stores and returns it, so the format is owned by this application.

The expanded 256-channel arrays are still retained in ``CompiledSet`` for
reports and compatibility, but they are not sent to the current opaque-block
firmware format.  Only the three measured regions are serialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np

from sdk.pressure_fit import (
    PRESSURE_FIT_CHANNEL_COUNT,
    PRESSURE_FIT_Q12_MAX,
    PRESSURE_FIT_Q12_MIN,
    encode_pressure_fit_blob,
    pressure_fit_crc32,
    pressure_fit_quantize,
)


FORMAT_NAME = "stouch_glove_pressure_fit"
FORMAT_VERSION = 1
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky", "palm")
FINGER_LABELS = {
    "thumb": "拇指",
    "index": "食指",
    "middle": "中指",
    "ring": "无名指",
    "pinky": "小指",
    "palm": "手掌",
}
FINGER_ALIASES = {
    "thumb": "thumb", "拇指": "thumb",
    "index": "index", "食指": "index",
    "middle": "middle", "中指": "middle",
    "ring": "ring", "无名指": "ring",
    "pinky": "pinky", "little": "pinky", "小指": "pinky",
    "palm": "palm", "手掌": "palm",
}

# The firmware parameter page is an opaque host-owned block (v1.2.16(2)); it
# only enforces a maximum of 780 bytes and never interprets k/b.  This compact
# format is the application-level representation used by this tool:
#   magic(4) version(1) side(1) set_count(1) region_count(1) serial(16)
#   repeated six times: set_index(1), then three pairs of int16 Q12 k/b
# It is deliberately compact (102 bytes for three regions) and keeps the six-set/three-region
# meaning explicit instead of writing six expanded 1024-byte matrices.
COMPACT_MAGIC = b"STPF"
COMPACT_FORMAT_VERSION = 2
COMPACT_SET_COUNT = 6
COMPACT_REGION_COUNT = 3
COMPACT_SERIAL_SIZE = 16
COMPACT_HEADER_SIZE = 8 + COMPACT_SERIAL_SIZE
COMPACT_BLOCK_SIZE = COMPACT_HEADER_SIZE + COMPACT_SET_COUNT * (1 + COMPACT_REGION_COUNT * 4)
COMPACT_MAX_BLOCK_SIZE = 780

# These are the same physical strips used by gui.rendering.tactile.  Right
# hand fingers expose 3 rows x 4 columns; left hand fingers expose 4 rows x 3
# columns.  The three regions are ordered along the report's 前/中/后 order.
MATRIX_FINGER_REGIONS = {
    "right": {
        "thumb": {"rows": [1, 2, 3], "cols": [15, 14, 13, 12]},
        "index": {"rows": [4, 5, 6], "cols": [15, 14, 13, 12]},
        "middle": {"rows": [7, 8, 9], "cols": [15, 14, 13, 12]},
        "ring": {"rows": [10, 11, 12], "cols": [15, 14, 13, 12]},
        "pinky": {"rows": [13, 14, 15], "cols": [15, 14, 13, 12]},
        "palm": {"rows": list(range(1, 16)), "cols": list(range(11, 2, -1))},
    },
    "left": {
        "thumb": {"rows": [3, 2, 1, 0], "cols": [12, 13, 14]},
        "index": {"rows": [3, 2, 1, 0], "cols": [9, 10, 11]},
        "middle": {"rows": [3, 2, 1, 0], "cols": [6, 7, 8]},
        "ring": {"rows": [3, 2, 1, 0], "cols": [3, 4, 5]},
        "pinky": {"rows": [3, 2, 1, 0], "cols": [0, 1, 2]},
        "palm": {"rows": list(range(12, 3, -1)), "cols": list(range(14, -1, -1))},
    },
}


@dataclass(frozen=True)
class RegionFit:
    finger: str
    set_index: int
    region: str
    channel_indices: tuple[int, ...]
    slope: float
    intercept: float
    r2: float
    rmse: float
    sample_count: int

    @property
    def wire_slope(self) -> float:
        return self.slope

    @property
    def wire_intercept(self) -> float:
        return self.intercept


@dataclass(frozen=True)
class CompiledSet:
    finger: str
    set_index: int
    k: np.ndarray
    b: np.ndarray
    blob: bytes
    regions: tuple[RegionFit, ...]

    @property
    def crc32(self) -> int:
        return pressure_fit_crc32(self.blob)


@dataclass(frozen=True)
class CalibrationPlan:
    path: Path
    side: str
    write_sets: tuple[int, ...]
    sets: tuple[CompiledSet, ...]
    response_zero: float
    response_scale: float


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _side(value: Any) -> str:
    value = str(value or "right").strip().lower()
    if value not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    return value


def _finger(value: Any) -> str:
    key = str(value or "").strip().lower()
    if key not in FINGER_ALIASES:
        raise ValueError(
            f"unknown finger {value!r}; use thumb/index/middle/ring/pinky")
    return FINGER_ALIASES[key]


def _channels_from_region(side: str, finger: str, region_index: int,
                          region: dict[str, Any]) -> tuple[int, ...]:
    explicit = region.get("channels")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError(f"{finger} region {region_index}: channels must be a non-empty list")
        indices = tuple(int(value) for value in explicit)
    else:
        layout = MATRIX_FINGER_REGIONS[side][finger]
        rows = region.get("rows")
        cols = region.get("cols")
        rows = list(layout["rows"] if rows is None else rows)
        cols = list(layout["cols"] if cols is None else cols)
        # Default finger region i is row i on the right and column i on the
        # left. Palm regions are three horizontal bands on both layouts.
        if region.get("rows") is None and region.get("cols") is None:
            if finger == "palm":
                split = np.array_split(np.asarray(rows, dtype=int), 3)
                rows = split[region_index].tolist()
            elif side == "right":
                if region_index >= len(rows):
                    raise ValueError(f"{finger} has more regions than matrix rows")
                rows = [rows[region_index]]
            else:
                if region_index >= len(cols):
                    raise ValueError(f"{finger} has more regions than matrix columns")
                cols = [cols[region_index]]
        indices = tuple(row * 16 + col for row in rows for col in cols)
    if not indices or any(index < 0 or index >= PRESSURE_FIT_CHANNEL_COUNT
                          for index in indices):
        raise ValueError(f"{finger} region {region_index}: channel index out of 0..255")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{finger} region {region_index}: duplicate channels")
    return indices


def _samples(region: dict[str, Any], *, response_zero: float) -> tuple[np.ndarray, np.ndarray]:
    raw = region.get("samples")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("each region needs at least two samples")
    forces: list[float] = []
    responses: list[float] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            force = item.get("force", item.get("force_N"))
            response = item.get("response", item.get("adc"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            force, response = item
        else:
            raise ValueError(f"sample {index} must be [force_N, adc] or an object")
        forces.append(_number(force, f"sample {index} force"))
        responses.append(_number(response, f"sample {index} response") - response_zero)
    x = np.asarray(forces, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if np.unique(x).size < 2:
        raise ValueError("a region needs at least two distinct force values")
    return x, y


def _fit_region(region: dict[str, Any], *, response_zero: float,
                finger: str, set_index: int, region_name: str,
                channels: tuple[int, ...]) -> RegionFit:
    fit = region.get("fit")
    if fit is not None:
        if not isinstance(fit, dict):
            raise ValueError(f"{finger} {region_name}: fit must be an object")
        slope = _number(fit.get("k", fit.get("slope")), f"{finger} {region_name} fit.k")
        intercept = _number(fit.get("b", fit.get("intercept")), f"{finger} {region_name} fit.b")
        if "samples" in region:
            x, y = _samples(region, response_zero=response_zero)
        else:
            x = y = np.asarray([], dtype=np.float64)
    else:
        x, y = _samples(region, response_zero=response_zero)
        slope, intercept = np.polyfit(x, y, 1)
    if slope <= 0:
        raise ValueError(f"{finger} {region_name}: slope must be positive")
    if x.size:
        predicted = slope * x + intercept
        residual = y - predicted
        ss_res = float(np.sum(residual * residual))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
        rmse = float(np.sqrt(np.mean(residual * residual)))
        count = int(x.size)
    else:
        r2 = float(fit.get("r2", float("nan")))
        rmse = float(fit.get("rmse", float("nan")))
        count = 0
    return RegionFit(finger, set_index, region_name, channels,
                     float(slope), float(intercept), r2, rmse, count)


def _finger_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [{**value, "name": key} for key, value in raw.items()]
    raise ValueError("fingers must be a list or object")


def load_calibration_plan(path: str | Path) -> CalibrationPlan:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("压力拟合输入文件必须是 CSV 文件（.csv）")
    return _load_csv_calibration_plan(path)


def _csv_number(value: str, label: str) -> float | None:
    value = str(value).strip()
    if not value:
        return None
    return _number(value, label)


def _load_csv_calibration_plan(path: Path) -> CalibrationPlan:
    import csv

    rows = None
    last_error = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as stream:
                rows = [[cell.strip() for cell in row]
                        for row in csv.reader(stream)
                        if any(cell.strip() for cell in row)]
            break
        except (OSError, UnicodeDecodeError) as exc:
            last_error = exc
    if rows is None:
        raise ValueError(f"cannot read {path}: {last_error}")

    side = "right"
    response_zero = 0.0
    response_scale = 2047.0
    header_index = None
    for index, row in enumerate(rows):
        key = row[0].lower().replace(" ", "") if row else ""
        value = row[1] if len(row) > 1 else ""
        if key in ("side", "手套侧别", "侧别"):
            side = _side(value)
        elif key in ("response_zero", "响应零点"):
            response_zero = _number(value, "response_zero")
        elif key in ("response_scale", "响应缩放"):
            response_scale = _number(value, "response_scale")
        normalized = {cell.lower().replace(" ", "") for cell in row}
        if ("finger" in normalized or "手指" in normalized) and ("adc" in normalized):
            header_index = index
    if response_scale <= 0:
        raise ValueError("response_scale/响应缩放 must be positive")
    if header_index is None:
        raise ValueError("CSV needs a header row containing finger, force_N and adc")
    header = [cell.lower().replace(" ", "") for cell in rows[header_index]]
    def column(*names: str) -> int:
        for name in names:
            if name in header:
                return header.index(name)
        raise ValueError(f"CSV header is missing {names[0]}")
    finger_column = column("finger", "手指")
    region_column = next(
        (index for index, name in enumerate(header)
         if name in ("region", "区域")), None)
    force_column = column("force_n", "forcen", "标准力n", "标准力(n)")
    adc_column = column("adc", "response", "响应adc", "响应")
    grouped: dict[str, dict[str, list[list[float]]]] = {
        finger: {region: [] for region in ("front", "middle", "back")}
        for finger in FINGER_ORDER
    }
    simple_grouped: dict[str, list[list[float]]] = {
        finger: [] for finger in FINGER_ORDER
    }
    region_aliases = {
        "front": "front", "前": "front", "first": "front", "1": "front",
        "middle": "middle", "中": "middle", "mid": "middle", "second": "middle", "2": "middle",
        "back": "back", "后": "back", "last": "back", "third": "back", "3": "back",
    }
    for row_number, row in enumerate(rows[header_index + 1:], start=header_index + 2):
        required_columns = [finger_column, force_column, adc_column]
        if region_column is not None:
            required_columns.append(region_column)
        if len(row) <= max(required_columns):
            raise ValueError(f"CSV row {row_number} has too few columns")
        finger = _finger(row[finger_column])
        region_key = None
        if region_column is not None:
            region_key = region_aliases.get(row[region_column].strip().lower())
            if region_key is None:
                raise ValueError(f"CSV row {row_number}: region must be 前/中/后")
        force = _csv_number(row[force_column], f"CSV row {row_number} force_N")
        adc = _csv_number(row[adc_column], f"CSV row {row_number} adc")
        if force is None and adc is None:
            continue
        if force is None or adc is None:
            raise ValueError(f"CSV row {row_number}: force_N and adc must be filled together")
        if region_key is None:
            simple_grouped[finger].append([force, adc])
        else:
            grouped[finger][region_key].append([force, adc])
    fingers = []
    for set_index, finger in enumerate(FINGER_ORDER):
        regions = ([{"name": "全部", "samples": simple_grouped[finger]}]
                   if region_column is None else [
                       {"name": region, "samples": grouped[finger][region]}
                       for region in ("front", "middle", "back")])
        fingers.append({
            "name": finger,
            "set_index": set_index,
            "regions": regions,
        })
    return _compile_calibration_data(
        path, side, response_zero, response_scale, fingers,
        tuple(range(len(FINGER_ORDER))))


def _docx_number(text: str, label: str) -> float | None:
    text = str(text).strip().replace(",", "")
    if not text:
        return None
    match = __import__("re").search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    return _number(match.group(0), label)


def _load_docx_calibration_plan(path: Path) -> CalibrationPlan:
    try:
        from docx import Document
        document = Document(str(path))
    except ImportError as exc:
        raise ValueError("当前 Python 环境缺少 python-docx，请在 gloves_sdk 中安装") from exc
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc

    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs
                  if paragraph.text.strip()]
    all_text = "\n".join(paragraphs)
    for table in document.tables:
        all_text += "\n" + "\n".join(
            cell.text.strip() for row in table.rows for cell in row.cells)
    side = "left" if ("左手" in all_text or "left" in all_text.lower()) else "right"
    response_zero = 0.0
    response_scale = 2047.0
    import re
    zero_match = re.search(r"(?:响应零点|response[_ ]?zero)\s*[:：=]\s*([-+]?\d+(?:\.\d+)?)",
                           all_text, re.IGNORECASE)
    scale_match = re.search(r"(?:响应缩放|response[_ ]?scale)\s*[:：=]\s*([-+]?\d+(?:\.\d+)?)",
                            all_text, re.IGNORECASE)
    if zero_match:
        response_zero = _number(zero_match.group(1), "响应零点")
    if scale_match:
        response_scale = _number(scale_match.group(1), "响应缩放")
    if response_scale <= 0:
        raise ValueError("响应缩放必须为正数")

    data_tables = []
    for table in document.tables:
        if not table.rows:
            continue
        header = [cell.text.strip().lower().replace(" ", "")
                  for cell in table.rows[0].cells]
        force_index = next((i for i, value in enumerate(header)
                            if "标准力" in value or "force" in value), None)
        adc_index = next((i for i, value in enumerate(header)
                          if "adc" in value or "响应" in value), None)
        if force_index is not None and adc_index is not None:
            data_tables.append((table, force_index, adc_index))
    expected = len(FINGER_ORDER) * 3
    if len(data_tables) != expected:
        raise ValueError(
            f"Word 中需要 {expected} 个采样表（5 根手指和手掌各 3 个区域），"
            f"实际找到 {len(data_tables)} 个。请使用提供的模板，不要删除区域表。")

    fingers = []
    for finger_index, finger in enumerate(FINGER_ORDER):
        regions = []
        for region_index in range(3):
            table, force_column, adc_column = data_tables[finger_index * 3 + region_index]
            samples = []
            for row_index, row in enumerate(table.rows[1:], start=1):
                cells = [cell.text.strip() for cell in row.cells]
                if max(force_column, adc_column) >= len(cells):
                    continue
                force = _docx_number(cells[force_column],
                                     f"{FINGER_LABELS[finger]} 区域{region_index + 1} 力")
                adc = _docx_number(cells[adc_column],
                                   f"{FINGER_LABELS[finger]} 区域{region_index + 1} ADC")
                if force is None and adc is None:
                    continue
                if force is None or adc is None:
                    raise ValueError(
                        f"{FINGER_LABELS[finger]} 区域{region_index + 1} 第{row_index}行"
                        "的标准力和 ADC 必须同时填写")
                samples.append([force, adc])
            regions.append({"name": ("前", "中", "后")[region_index],
                            "samples": samples})
        fingers.append({"name": finger, "set_index": finger_index,
                        "regions": regions})
    return _compile_calibration_data(
        path, side, response_zero, response_scale, fingers,
        tuple(range(len(FINGER_ORDER))))


def _compile_calibration_data(path: Path, side: str, response_zero: float,
                              response_scale: float, fingers: list[dict[str, Any]],
                              write_sets: tuple[int, ...]) -> CalibrationPlan:
    data = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "side": side,
        "write_sets": list(write_sets),
        "measurement": {"response_zero": response_zero},
        "wire": {"response_scale": response_scale},
        "defaults": {"k": 1.0, "b": 0.0},
        "fingers": fingers,
    }
    return _load_json_calibration_data(path, data)


def _load_json_calibration_data(path: Path, data: dict[str, Any]) -> CalibrationPlan:
    if not isinstance(data, dict):
        raise ValueError("the calibration file root must be an object")
    if data.get("format") not in (FORMAT_NAME, None):
        raise ValueError(f"unsupported format {data.get('format')!r}")
    if data.get("version", FORMAT_VERSION) != FORMAT_VERSION:
        raise ValueError(f"unsupported calibration file version {data.get('version')!r}")
    side = _side(data.get("side", "right"))
    measurement = data.get("measurement", {})
    if not isinstance(measurement, dict):
        raise ValueError("measurement must be an object")
    response_zero = _number(measurement.get("response_zero", 0.0), "measurement.response_zero")
    wire = data.get("wire", {})
    if not isinstance(wire, dict):
        raise ValueError("wire must be an object")
    response_scale = _number(wire.get("response_scale", 2047.0), "wire.response_scale")
    if response_scale <= 0:
        raise ValueError("wire.response_scale must be positive")
    defaults = data.get("defaults", {"k": 1.0, "b": 0.0})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")
    default_k = _number(defaults.get("k", 1.0), "defaults.k")
    default_b = _number(defaults.get("b", 0.0), "defaults.b")
    write_sets_raw = data.get("write_sets")
    write_sets = None if write_sets_raw is None else tuple(int(x) for x in write_sets_raw)
    if write_sets is not None and any(x < 0 or x >= 6 for x in write_sets):
        raise ValueError("write_sets entries must be 0..5")

    compiled: list[CompiledSet] = []
    seen_sets: set[int] = set()
    for ordinal, entry in enumerate(_finger_entries(data.get("fingers"))):
        if not isinstance(entry, dict):
            raise ValueError(f"fingers[{ordinal}] must be an object")
        finger = _finger(entry.get("name", entry.get("finger")))
        set_index = int(entry.get("set_index", FINGER_ORDER.index(finger)))
        if set_index in seen_sets:
            raise ValueError(f"set_index {set_index} is assigned more than once")
        if not 0 <= set_index < 6:
            raise ValueError(f"{finger}: set_index must be 0..5")
        seen_sets.add(set_index)
        regions_raw = entry.get("regions")
        if not isinstance(regions_raw, list) or not regions_raw:
            raise ValueError(f"{finger}: regions must be a non-empty list")
        k_values = np.full(PRESSURE_FIT_CHANNEL_COUNT, default_k, dtype=np.float64)
        b_values = np.full(PRESSURE_FIT_CHANNEL_COUNT, default_b, dtype=np.float64)
        fits: list[RegionFit] = []
        used_channels: set[int] = set()
        for region_index, region in enumerate(regions_raw):
            if not isinstance(region, dict):
                raise ValueError(f"{finger} region {region_index} must be an object")
            name = str(region.get("name", ("前", "中", "后")[region_index]
                             if region_index < 3 else f"region-{region_index + 1}"))
            channels = _channels_from_region(side, finger, region_index, region)
            overlap = used_channels.intersection(channels)
            if overlap:
                raise ValueError(f"{finger} {name}: overlapping channels {sorted(overlap)[:5]}")
            used_channels.update(channels)
            fit = _fit_region(region, response_zero=response_zero,
                              finger=finger, set_index=set_index,
                              region_name=name, channels=channels)
            wire_k = fit.slope / response_scale
            wire_b = fit.intercept / response_scale
            if abs(wire_k) > PRESSURE_FIT_Q12_MAX / 4096.0:
                raise ValueError(f"{finger} {name}: k={wire_k:g} does not fit Q12; increase wire.response_scale")
            if abs(wire_b) > PRESSURE_FIT_Q12_MAX / 4096.0:
                raise ValueError(f"{finger} {name}: b={wire_b:g} does not fit Q12; increase wire.response_scale")
            k_values[list(channels)] = wire_k
            b_values[list(channels)] = wire_b
            fits.append(RegionFit(
                finger=fit.finger, set_index=fit.set_index, region=fit.region,
                channel_indices=fit.channel_indices, slope=wire_k,
                intercept=wire_b, r2=fit.r2, rmse=fit.rmse,
                sample_count=fit.sample_count))
        k = pressure_fit_quantize(k_values, name=f"{finger} k")
        b = pressure_fit_quantize(b_values, name=f"{finger} b")
        blob = encode_pressure_fit_blob(k, b)
        compiled.append(CompiledSet(finger, set_index, k, b, blob, tuple(fits)))

    if not compiled:
        raise ValueError("fingers must contain at least one finger")
    if write_sets is None:
        write_sets = tuple(item.set_index for item in compiled)
    missing = set(write_sets) - {item.set_index for item in compiled}
    if missing:
        raise ValueError(f"write_sets refers to sets with no finger data: {sorted(missing)}")
    return CalibrationPlan(path, side, tuple(write_sets), tuple(compiled),
                           response_zero, response_scale)


def summary_rows(plan: CalibrationPlan) -> list[dict[str, Any]]:
    """Plain-data rows suitable for the Qt table and logs."""
    rows = []
    for item in sorted(plan.sets, key=lambda value: value.set_index):
        k_values = [float(pressure_fit_quantize([region.wire_slope])[0]) / 4096.0
                    for region in item.regions]
        b_values = [float(pressure_fit_quantize([region.wire_intercept])[0]) / 4096.0
                    for region in item.regions]
        rows.append({
            "finger": item.finger,
            "label": FINGER_LABELS[item.finger],
            "set_index": item.set_index,
            "regions": len(item.regions),
            "samples": sum(region.sample_count for region in item.regions),
            "k_min": min(k_values),
            "k_max": max(k_values),
            "b_min": min(b_values),
            "b_max": max(b_values),
            "crc32": item.crc32,
            "write": item.set_index in plan.write_sets,
        })
    return rows


def _serial_bytes(serial: str | None) -> bytes:
    serial = str(serial or "").strip()
    if (serial and (len(serial) > COMPACT_SERIAL_SIZE
                    or not serial.isascii() or not serial.isalnum())):
        raise ValueError(f"序列号必须是 1～{COMPACT_SERIAL_SIZE} 位字母或数字")
    return serial.encode("ascii").ljust(COMPACT_SERIAL_SIZE, b"\0")


def encode_compact_pressure_fit_pairs(
        pairs_by_set: dict[int, list[tuple[float, float]]],
        side: str,
        serial: str = "",
) -> bytes:
    """Encode six sets of one or three Q12 k/b region pairs."""

    side_code = {"left": 1, "right": 2}.get(str(side).lower())
    if side_code is None:
        raise ValueError("side must be 'left' or 'right'")
    if set(pairs_by_set) != set(range(COMPACT_SET_COUNT)):
        raise ValueError("compact pressure-fit block requires sets 0..5")
    region_count = len(pairs_by_set[0])
    if region_count not in (1, COMPACT_REGION_COUNT):
        raise ValueError("each set must contain one or three regions")
    if any(len(pairs_by_set[index]) != region_count
           for index in range(COMPACT_SET_COUNT)):
        raise ValueError("all compact sets must contain the same region count")
    output = bytearray(
        COMPACT_MAGIC
        + bytes((COMPACT_FORMAT_VERSION, side_code,
                 COMPACT_SET_COUNT, region_count))
        + _serial_bytes(serial)
    )
    for set_index in range(COMPACT_SET_COUNT):
        output.append(set_index)
        for region_index, (k_value, b_value) in enumerate(pairs_by_set[set_index]):
            k = int(pressure_fit_quantize(
                [k_value], name=f"set {set_index} region {region_index} k")[0])
            b = int(pressure_fit_quantize(
                [b_value], name=f"set {set_index} region {region_index} b")[0])
            output.extend(struct.pack("<hh", k, b))
    encoded = bytes(output)
    if len(encoded) > COMPACT_MAX_BLOCK_SIZE:
        raise AssertionError("compact pressure-fit block exceeds firmware limit")
    return encoded


def encode_compact_pressure_fit_block(
        sets: list[CompiledSet] | tuple[CompiledSet, ...], side: str,
        serial: str = "",
) -> bytes:
    """Encode six fitted CSV sets into the opaque firmware block format."""

    by_index = {int(item.set_index): item for item in sets}
    if set(by_index) != set(range(COMPACT_SET_COUNT)):
        raise ValueError("compact pressure-fit block requires sets 0..5")
    region_count = len(by_index[0].regions)
    if region_count not in (1, COMPACT_REGION_COUNT):
        raise ValueError("each set must contain one or three regions")
    pairs = {
        set_index: [(region.wire_slope, region.wire_intercept)
                    for region in by_index[set_index].regions]
        for set_index in range(COMPACT_SET_COUNT)
    }
    return encode_compact_pressure_fit_pairs(pairs, side, serial)


def decode_compact_pressure_fit_block(data: bytes) -> dict[str, Any] | None:
    """Decode this tool's compact block, or return ``None`` for other blobs."""

    data = bytes(data)
    if len(data) < 8 or data[:4] != COMPACT_MAGIC:
        return None
    version, side_code, set_count, region_count = data[4:8]
    if (version not in (1, COMPACT_FORMAT_VERSION)
            or set_count != COMPACT_SET_COUNT
            or region_count not in (1, COMPACT_REGION_COUNT)):
        return None
    side = {1: "left", 2: "right"}.get(side_code)
    if side is None:
        return None
    if version == 1:
        serial = ""
        header_size = 8
    else:
        # v2 originally used a six-byte serial.  Keep decoding those blocks
        # so parameters written by the previous tool remain readable, while
        # new blocks use the 16-byte timestamp/counter serial below.
        legacy_header_size = 8 + 6
        new_header_size = COMPACT_HEADER_SIZE
        new_expected = new_header_size + COMPACT_SET_COUNT * (1 + region_count * 4)
        legacy_expected = legacy_header_size + COMPACT_SET_COUNT * (1 + region_count * 4)
        if len(data) == new_expected:
            serial = data[8:8 + COMPACT_SERIAL_SIZE].split(b"\0", 1)[0].decode(
                "ascii", errors="replace")
            header_size = new_header_size
        elif len(data) == legacy_expected:
            serial = data[8:14].split(b"\0", 1)[0].decode(
                "ascii", errors="replace")
            header_size = legacy_header_size
        else:
            return None
    expected = header_size + COMPACT_SET_COUNT * (1 + region_count * 4)
    if len(data) != expected:
        return None
    sets = []
    offset = header_size
    for _ in range(COMPACT_SET_COUNT):
        set_index = data[offset]
        offset += 1
        regions = []
        for region_index in range(region_count):
            k, b = struct.unpack_from("<hh", data, offset)
            offset += 4
            regions.append({
                "region_index": region_index,
                "k_q12_raw": int(k),
                "b_q12_raw": int(b),
                "k": float(k) / 4096.0,
                "b": float(b) / 4096.0,
            })
        sets.append({"set_index": int(set_index), "regions": regions})
    return {"version": int(version), "side": side, "serial": serial,
            "region_count": int(region_count), "sets": sets}


def write_compiled_report(plan: CalibrationPlan, path: str | Path) -> Path:
    """Write an inspectable report without exposing binary set internals."""
    output = {
        "format": "stouch_glove_pressure_fit_compiled",
        "version": 1,
        "source": str(plan.path),
        "side": plan.side,
        "response_zero": plan.response_zero,
        "response_scale": plan.response_scale,
        "sets": [],
    }
    for item in sorted(plan.sets, key=lambda value: value.set_index):
        output["sets"].append({
            "finger": item.finger,
            "set_index": item.set_index,
            "write": item.set_index in plan.write_sets,
            "crc32": f"0x{item.crc32:08X}",
            "regions": [
                {"name": fit.region, "channels": list(fit.channel_indices),
                 "k_q12": fit.wire_slope, "b_q12": fit.wire_intercept,
                 "r2": fit.r2, "rmse": fit.rmse,
                 "samples": fit.sample_count}
                for fit in item.regions
            ],
        })
    destination = Path(path)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return destination


__all__ = [
    "CalibrationPlan", "CompiledSet", "FINGER_LABELS", "FINGER_ORDER",
    "FORMAT_NAME", "MATRIX_FINGER_REGIONS", "RegionFit",
    "COMPACT_MAGIC", "COMPACT_BLOCK_SIZE", "COMPACT_FORMAT_VERSION",
    "encode_compact_pressure_fit_pairs", "encode_compact_pressure_fit_block",
    "decode_compact_pressure_fit_block",
    "load_calibration_plan", "summary_rows", "write_compiled_report",
]
