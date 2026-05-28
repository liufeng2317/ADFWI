"""Marmousi2 acoustic validation case definitions.

This module is local to ``examples/validation/marmousi2_acoustic_bv12`` so the
validation notebooks do not import definitions from another example/check case.
It contains only lightweight data-loading, survey, and source-wavelet helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ADFWI.survey import Receiver, Source, Survey
from ADFWI.utils import wavelet


def load_npz(path: Path):
    """Load a required Marmousi2 ``.npz`` file."""

    if not path.exists():
        raise FileNotFoundError(f"required Marmousi2 file does not exist: {path}")
    return np.load(path, allow_pickle=True)


def cumulative_trapezoid(values: np.ndarray, dt: float) -> np.ndarray:
    """Integrate a source wavelet with the historical cumulative trapezoid rule."""

    out = np.zeros_like(values, dtype=np.float32)
    if values.size > 1:
        out[1:] = np.cumsum((values[:-1] + values[1:]) * (0.5 * dt), dtype=np.float64).astype(np.float32)
    return out


def build_source_wavelet(nt: int, dt: float, f0: float) -> np.ndarray:
    """Return the integrated source wavelet used by the Marmousi2 case."""

    _, src_v = wavelet(nt, dt, f0, amp0=1)
    return cumulative_trapezoid(src_v.astype(np.float32), dt)


def build_survey(obs_npz: Any, f0: float) -> Survey:
    """Build the Marmousi2 acoustic survey from observation metadata."""

    nt = int(obs_npz["nt"])
    dt = float(obs_npz["dt"])
    src_loc = np.asarray(obs_npz["src_loc"], dtype=np.int64)
    rcv_loc = np.asarray(obs_npz["rcv_loc"], dtype=np.int64)
    src_type = np.asarray(obs_npz["src_type"]).astype(str)
    rcv_type = np.asarray(obs_npz["rcv_type"]).astype(str)

    src_v = build_source_wavelet(nt, dt, f0)
    source = Source(nt=nt, dt=dt, f0=f0)
    for (src_x, src_z), src_kind in zip(src_loc, src_type):
        source.add_source(int(src_x), int(src_z), src_v, src_type=str(src_kind))

    receiver = Receiver(nt=nt, dt=dt)
    for (rcv_x, rcv_z), rcv_kind in zip(rcv_loc, rcv_type):
        receiver.add_receiver(int(rcv_x), int(rcv_z), rcv_type=str(rcv_kind))
    return Survey(source, receiver)
