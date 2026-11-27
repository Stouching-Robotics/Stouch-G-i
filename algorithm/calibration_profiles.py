"""Resolve an existing matching calibration profile without fitting or writing.

Old projected contact anchors cannot reliably reconstruct discarded IMU data.
Such files remain readable as legacy calibrations; they are never silently
fitted, assigned another wearing's profile, or overwritten here.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path

from common.errors import CalibrationCompatibilityError

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CalibrationProfileResolution:
    requested_path: Path
    path: Path
    status: str


def _read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Calibration must be a JSON object')
    return value


class CalibrationProfileCatalog:
    """Read and validate once per explicitly bounded list construction.

    Discard after each refresh; files are assumed stable during that refresh.
    """

    def __init__(self):
        self._payloads = {}
        self._read_errors = {}
        self._validations = {}
        self._directories = {}

    def read(self, path):
        path = Path(path).expanduser().resolve()
        if path in self._read_errors:
            error_type, message = self._read_errors[path]
            raise error_type(message)
        if path not in self._payloads:
            try:
                self._payloads[path] = _read(path)
            except (OSError, ValueError) as exc:
                error_type = OSError if isinstance(exc, OSError) else ValueError
                self._read_errors[path] = (error_type, str(exc))
                raise
        return self._payloads[path]

    def validate(self, path, side):
        path = Path(path).expanduser().resolve()
        key = (side, path)
        if key not in self._validations:
            try:
                self._validate(path, side)
            except CalibrationCompatibilityError as exc:
                self._validations[key] = str(exc)
            else:
                self._validations[key] = None
        error = self._validations[key]
        if error is not None:
            raise CalibrationCompatibilityError(error)

    def _validate(self, path, side):
        # Use the real loader so future validation changes are inherited.
        from algorithm.solver import HandSolverEngine
        HandSolverEngine(side, path)

    def _candidates(self, directories):
        candidates = set()
        for directory in directories:
            directory = Path(directory).expanduser().resolve()
            if directory not in self._directories:
                self._directories[directory] = tuple(directory.glob('*.json'))
            candidates.update(self._directories[directory])
        return sorted(candidates)


def resolve_calibration_profile(path, side, candidate_directories=None, *, catalog=None):
    """Prefer only an exactly matching already-saved full motion profile.

    Current-policy motion/gesture files and invalid explicit files are never
    replaced. Old-policy motion files may use an already-refitted companion
    with identical source fields/captures. Different root/install IDs or captures,
    shape, serial, mapping, metadata or other source fields prohibit a match.
    Multiple numerically different matches are reported, never guessed.
    """
    requested = Path(path).expanduser().resolve()
    if catalog is None:
        catalog = CalibrationProfileCatalog()
    source = catalog.read(requested)
    if str(source.get('side') or 'right').lower() != side:
        raise ValueError(f'Calibration side does not match {side}')
    shape = source.get('param_shape_calib')
    if not isinstance(shape, dict):
        return CalibrationProfileResolution(requested, requested, 'legacy')
    fallback_status = 'legacy'
    comparable_source = source
    upgrading_twist = False
    if 'sensor_motion_refit' in shape:
        from algorithm.lite.sensor_motion_refit import ROOT_TWIST_POLICY, SCHEMA
        profile = shape['sensor_motion_refit']
        if (not isinstance(profile, dict) or profile.get('schema') != SCHEMA
                or profile.get('root_twist_policy', 'legacy-chain-transfer') not in
                    ('legacy-chain-transfer', 'outgoing-tangent-unit-circle-v1')):
            return CalibrationProfileResolution(requested, requested, 'embedded')
        upgrading_twist = True
        fallback_status = 'embedded'
        comparable_source = copy.deepcopy(source)
        comparable_source['param_shape_calib'].pop('sensor_motion_refit')
    if 'gesture_retarget' in shape:
        if upgrading_twist:
            try:
                catalog.validate(requested, side)
            except CalibrationCompatibilityError:
                return CalibrationProfileResolution(requested, requested, 'embedded')
        return CalibrationProfileResolution(requested, requested, 'gesture')
    if candidate_directories is None:
        candidate_directories = (requested.parent/'experiments', ROOT/'calibration/experiments')
    matches = []
    source_validated = False
    for candidate in catalog._candidates(candidate_directories):
        try:
            payload = catalog.read(candidate)
            candidate_shape = payload.get('param_shape_calib')
            if not isinstance(candidate_shape, dict):
                continue
            profile = candidate_shape.get('sensor_motion_refit')
            if (not isinstance(profile, dict) or profile.get('schema') != 'stouch-sensor-motion-refit-v1'
                    or profile.get('side') != side
                    or profile.get('source_dependency') != shape.get('depends_on')):
                continue
            if upgrading_twist and profile.get('root_twist_policy') != ROOT_TWIST_POLICY:
                continue
            original = copy.deepcopy(payload)
            original['param_shape_calib'].pop('sensor_motion_refit')
            if original != comparable_source:
                continue
            # Validate the same basis, dependency and baked geometry as the
            # real loader. A structurally valid but broken fallback must not
            # make an otherwise readable legacy file fail at viewer startup.
            # This constructs an offline solver only; no fitting/device IO.
            serialized_profile = json.dumps(profile, sort_keys=True, allow_nan=False)
            catalog.validate(candidate, side)
            if upgrading_twist and not source_validated:
                try:
                    catalog.validate(requested, side)
                except CalibrationCompatibilityError:
                    return CalibrationProfileResolution(requested, requested, 'embedded')
                source_validated = True
            matches.append((candidate.resolve(), serialized_profile))
        except (OSError, ValueError, TypeError, CalibrationCompatibilityError):
            continue
    if not matches:
        return CalibrationProfileResolution(requested, requested, fallback_status)
    if len({profile for _, profile in matches}) > 1:
        return CalibrationProfileResolution(requested, requested, 'ambiguous')
    return CalibrationProfileResolution(requested, matches[0][0], 'matched')


def prepare_viewer_calibration(path, side):
    """Validate the explicitly selected Friston file before the viewer connects."""
    requested = Path(path).expanduser().resolve()
    CalibrationProfileCatalog().validate(requested, side)
    return requested
