"""Public raw IMU and tactile acquisition interfaces.

The implementation is intentionally plaintext and contains only USB framing,
serial acquisition, and immutable public data contracts.
"""

from glove_io.streams import RawImuStream, SensorStream, TactileStream

__all__ = ["RawImuStream", "SensorStream", "TactileStream"]
