"""Backward-compatible waveform normalization import surface.

The canonical waveform-amplitude normalization formula lives next to the
`TraceNormalize` transform in ``ADFWI.fwi.transforms.amplitude``. This module is
kept so existing imports from ``ADFWI.fwi.normalization`` continue to work.
"""

from __future__ import annotations

from ADFWI.fwi.transforms.amplitude import normalize_waveform

__all__ = ["normalize_waveform"]
