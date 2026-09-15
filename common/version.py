"""STM32 USB glove compatibility identifiers based on ESP Glove 2.1.

These values are persisted with calibration results.  Changing sensor routing,
the anatomical frame convention, or the MANO model must also change the
corresponding identifier so stale calibration cannot be loaded silently.
"""

PROJECT_VERSION = "stm32-usb-2.2"
CALIBRATION_SCHEMA_VERSION = 4
IMU_MAPPING_VERSION = "stm32-usb-live-segment-remap-2026-08-19-v4"
ANATOMICAL_FRAME_VERSION = "mano-anatomical-frame-2026-08-06-v1"
HAND_MODEL_VERSION = "MANO_RIGHT.pkl-v1"
