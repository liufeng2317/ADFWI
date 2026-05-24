"""Backward-compatible waveform transform exports.

The transform implementations are split by responsibility across this package:
``amplitude``, ``filters``, ``masks``, ``mutes``, and ``receivers``. This module
keeps the historical ``ADFWI.fwi.transforms.waveform`` import path stable.
"""

from __future__ import annotations

from .amplitude import TraceNormalize
from .filters import LegacyLowPassFilter, LowPassFilter
from .masks import DataMask, ReceiverMask
from .mutes import LegacyLateWindowMute, LegacyOffsetMute
from .receivers import select_or_mask_receivers

__all__ = [
    "DataMask",
    "LegacyLateWindowMute",
    "LegacyLowPassFilter",
    "LegacyOffsetMute",
    "LowPassFilter",
    "ReceiverMask",
    "TraceNormalize",
    "select_or_mask_receivers",
]
