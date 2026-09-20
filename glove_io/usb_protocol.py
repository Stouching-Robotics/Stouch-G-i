"""Re-export shim: the STM32 glove wire protocol now lives in :mod:`common.usb_cdc`.

There used to be two hand-maintained copies of this module -- one here and one
in :mod:`common.usb_cdc` -- and they had already drifted apart (each was missing
functions the other had).  The framing layer is deliberately dependency-free, so
it belongs in ``common`` (a leaf package); ``glove_io`` sits above it and keeps
this name as an alias for the callers that grew up importing it from here.

The star import is intentional: it forwards *every* public name including ones
added later, so a new decoder is exported here automatically instead of quietly
existing on only one of the two paths.  :data:`__all__` is re-exported too so
``from glove_io.usb_protocol import *`` matches ``from common.usb_cdc import *``.

Do **not** add logic here, and do **not** import any other ``glove_io``
submodule: :mod:`glove_io.__init__` imports :mod:`glove_io.streams`, which
imports this module, so a sibling import would re-enter a half-initialised
package.  Note also that a star import *copies bindings* -- assigning
``glove_io.usb_protocol.SOMETHING = ...`` would shadow the name for this module
only and have no effect on :mod:`common.usb_cdc`.
"""

from common.usb_cdc import *  # noqa: F401,F403
from common.usb_cdc import __all__ as __all__
