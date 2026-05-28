"""Public waveform transform API for FWI loss preparation.

Transforms operate on synthetic/observed tensor pairs after receiver selection.
Most classes preserve tensor shape and are composable through
``DataTransformPipeline``. ``select_or_mask_receivers`` is exported from this
package for convenience, but it is intentionally a standalone helper because it
may gather receiver traces and change the receiver dimension.

Names prefixed with ``Legacy`` wrap historical ADFWI preprocessing behavior for
numerical compatibility. They are not new pure-torch implementations.
"""

from .amplitude import TraceNormalize, normalize_waveform
from .base import DataTransform, DataTransformPipeline
from .filters import LegacyLowPassFilter, LowPassFilter
from .masks import DataMask, ReceiverMask
from .mutes import LegacyLateWindowMute, LegacyOffsetMute
from .receivers import select_or_mask_receivers

__all__ = [
    "DataTransform",
    "DataTransformPipeline",
    "DataMask",
    "LegacyLateWindowMute",
    "LegacyLowPassFilter",
    "LegacyOffsetMute",
    "LowPassFilter",
    "ReceiverMask",
    "TraceNormalize",
    "normalize_waveform",
    "select_or_mask_receivers",
]
