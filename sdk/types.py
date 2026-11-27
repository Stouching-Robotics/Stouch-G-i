from dataclasses import dataclass
import numpy as np
from common.types import HandFrame as BaseHandFrame, _array

@dataclass(frozen=True)
class HandFrame(BaseHandFrame):
    force_n: np.ndarray | None = None
    force_source: str = "unavailable"
    force_status: str = "unread"
    force_error: str = ""
    force_fresh: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.force_n is not None:
            object.__setattr__(self, "force_n", _array(
                self.force_n, (16, 16), np.float32, "force_n"))
