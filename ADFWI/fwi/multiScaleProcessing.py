"""Backward-compatible legacy multiscale processing imports.

The implementation lives in ``ADFWI.fwi.multiscale.legacy_lowpass``. This
module intentionally contains only re-exports so existing imports from
``ADFWI.fwi.multiScaleProcessing`` keep working while new code can import from
the package-style ``ADFWI.fwi.multiscale`` namespace.
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
