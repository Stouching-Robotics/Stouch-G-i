"""Hand package with lazy exports to avoid calibration/model import cycles."""

__all__ = ["HandStateMachine"]


def __getattr__(name):
    if name == "HandStateMachine":
        from .hand import HandStateMachine
        return HandStateMachine
    raise AttributeError(name)
