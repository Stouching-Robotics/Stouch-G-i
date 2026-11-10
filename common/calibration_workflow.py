"""Calibration step identities and dependency invalidation, independent of UI order."""
from common.enums import (MoCapCalibrateRootType as Root,
                          MoCapCalibrateInstallationType as Installation,
                          MoCapCalibrateShapeType as Shape,
                          MoCapCalibrateFistType as Fist)

ROOT_CAPTURES = ('root_level', 'root_up', 'root_down', 'root_roll', 'root_yaw')
CONTACT_CAPTURES = ('contact_index', 'contact_middle', 'contact_ring', 'contact_little')
INSTALLATION_STEPS = ('installation_open', 'installation_compute')
DERIVED_FINE_STEPS = ('contact_compute', 'verify_open')


def step_id(flow, trigger):
    tables = {
        'root': {Root.HORIZONTAL: 'root_level', Root.VERTICAL_UP: 'root_up',
                 Root.VERTICAL_DOWN: 'root_down', Root.ROLL: 'root_roll',
                 Root.YAW: 'root_yaw', Root.CALC: 'root_compute'},
        'installation': {Installation.POSE_0: 'installation_open',
                         Installation.CALC: 'installation_compute'},
        'shape': {Shape.INDEX: 'contact_index', Shape.MIDDLE: 'contact_middle',
                  Shape.RING: 'contact_ring', Shape.LITTLE: 'contact_little',
                  Shape.CALC: 'contact_compute', Shape.VERIFY_OPEN: 'verify_open'},
        'fist': {Fist.CAPTURE: 'fist_capture', Fist.CALC: 'fist_capture'},
    }
    flow = 'shape' if flow == 'contact' else flow
    try:
        return tables[flow][trigger]
    except (KeyError, TypeError):
        raise ValueError(f'Unknown calibration step: {flow}/{trigger}') from None


def invalidated_step_ids(identifier, workflow='simplified'):
    if workflow not in ('full', 'simplified'):
        raise ValueError('calibration workflow must be full or simplified')
    fine = set(CONTACT_CAPTURES) | {'contact_compute', 'fist_capture', 'verify_open'}
    if identifier == 'root_level':
        return set(ROOT_CAPTURES) | {'root_compute'} | set(INSTALLATION_STEPS) | fine
    if identifier in ROOT_CAPTURES or identifier == 'root_compute':
        return {identifier, 'root_compute'} | set(INSTALLATION_STEPS) | fine
    if identifier == 'installation_open':
        return set(INSTALLATION_STEPS) | fine
    if identifier == 'installation_compute':
        return {identifier} | fine
    if identifier in CONTACT_CAPTURES:
        return {identifier, 'contact_compute', 'verify_open'}
    if identifier == 'contact_compute':
        return {'contact_compute', 'verify_open'}
    if identifier == 'fist_capture':
        return {'fist_capture', 'verify_open'}
    if identifier == 'verify_open':
        return {'verify_open'}
    raise ValueError(f'Unknown calibration step id: {identifier}')
