"""Strict, I/O-free decoders for device force calibration blocks."""
import math
import struct

REGION_KEYS = tuple(f"finger_{i}" for i in range(5)) + ("palm",)
HEADER = b"STPF\x04\x00\x06\x01"

class ForceParameterError(ValueError):
    pass

def decode_force_parameters(data: bytes, *, expected_side=None) -> dict:
    data = bytes(data)
    if len(data) < 8 or data[:4] != b"STPF":
        raise ForceParameterError("unknown or empty parameter block")
    version, side_code, count, regions = data[4:8]
    if version == 4:
        if data[:8] != HEADER or len(data) != 38:
            raise ForceParameterError("invalid v4 header or length")
        pairs = {}
        for offset in range(8, 38, 5):
            index, k, b = struct.unpack_from("<Bhh", data, offset)
            if index >= 6 or index in pairs or k <= 0:
                raise ForceParameterError("invalid v4 region or slope")
            pairs[index] = {"k": k * 2047.0 / 4096, "b": b * 2047.0 / 4096}
        if set(pairs) != set(range(6)):
            raise ForceParameterError("missing v4 region")
        return {"version": 4, "model": "delta_adc", "fits": {name: pairs[i] for i, name in enumerate(REGION_KEYS)}}
    if version not in (1, 2) or side_code not in (1, 2) or count != 6 or regions not in (1, 3):
        raise ForceParameterError("unsupported compact schema")
    side = {1: "left", 2: "right"}[side_code]
    if expected_side is not None and str(expected_side).lower() != side:
        raise ForceParameterError("calibration side mismatch")
    payload_size = 6 * (1 + regions * 4)
    headers = (8,) if version == 1 else (14, 24)
    header = next((n for n in headers if len(data) == n + payload_size), None)
    if header is None:
        raise ForceParameterError("invalid compact length")
    serial_raw = data[8:header]
    try:
        serial = serial_raw.split(b"\0", 1)[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForceParameterError("invalid serial") from exc
    if serial_raw[len(serial):].strip(b"\0") or (serial and not serial.isalnum()):
        raise ForceParameterError("invalid serial")
    sets, seen, offset = [], set(), header
    for _ in range(6):
        index = data[offset]; offset += 1
        if index >= 6 or index in seen:
            raise ForceParameterError("invalid compact region ID")
        seen.add(index); pairs = []
        for i in range(regions):
            k, b = struct.unpack_from("<hh", data, offset); offset += 4
            if k <= 0:
                raise ForceParameterError("invalid compact slope")
            pairs.append({"region_index": i, "k_q12_raw": k, "b_q12_raw": b, "k": k / 4096.0, "b": b / 4096.0})
        sets.append({"set_index": index, "regions": pairs})
    return {"version": version, "model": "raw_adc_normalized", "side": side, "serial": serial, "region_count": regions, "sets": sets}
