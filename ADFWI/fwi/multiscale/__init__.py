"""Multiscale processing helpers for FWI workflows."""

from .legacy_lowpass import (
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
