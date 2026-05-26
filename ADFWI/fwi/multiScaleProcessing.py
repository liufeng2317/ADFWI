"""Backward-compatible legacy multiscale processing imports.

The implementation now lives in ``ADFWI.fwi.multiscale.legacy_lowpass``. This
module is kept so existing imports from ``ADFWI.fwi.multiScaleProcessing`` keep
working.
"""

from __future__ import annotations

from ADFWI.fwi.multiscale import (
    Lfilter,
    adj_lowpass,
    data2d_to_3d,
    data3d_to_2d,
    lowpass,
    lpass,
)

__all__ = [
    "Lfilter",
    "adj_lowpass",
    "data2d_to_3d",
    "data3d_to_2d",
    "lowpass",
    "lpass",
]
