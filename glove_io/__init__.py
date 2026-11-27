"""Plaintext acquisition and recording support for the public glove SDK.

This package deliberately contains no hand-pose or calibration algorithms.
It remains readable when :mod:`algorithm` is compiled/encrypted.
"""

from glove_io.streams import RawImuStream, SensorStream, TactileStream

__all__ = ["RawImuStream", "SensorStream", "TactileStream"]
