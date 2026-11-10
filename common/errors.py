"""Stable public exception hierarchy for the modular glove SDK."""


class GloveSdkError(RuntimeError):
    """Base class for all supported SDK errors."""


class DeviceNotFoundError(GloveSdkError):
    pass


class DeviceBindingError(GloveSdkError):
    pass


class DeviceBusyError(GloveSdkError):
    pass


class StreamError(GloveSdkError):
    pass


class StreamTimeoutError(StreamError):
    pass


class StreamClosedError(StreamError):
    pass


class CalibrationError(GloveSdkError):
    pass


class CalibrationQualityError(CalibrationError):
    def __init__(self, message, *, retry_steps=()):
        super().__init__(message)
        self.retry_steps = tuple(retry_steps)


class CalibrationCompatibilityError(CalibrationError):
    pass


class RecordingError(GloveSdkError):
    pass


class ReplayError(GloveSdkError):
    pass


class UnsupportedPlatformError(GloveSdkError):
    pass
