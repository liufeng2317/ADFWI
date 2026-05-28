"""Public waveform misfit objectives.

The names keep the historical ADFWI API. ``Misfit_waveform_L2`` is the legacy
L2-norm objective; ``Misfit_waveform_SquaredL2`` is the safer squared residual
variant used by newer smoke tests.
"""

from .base import Misfit
from .L1 import Misfit_waveform_L1
from .L2 import Misfit_waveform_L2, Misfit_waveform_SquaredL2
from .SmoothL1 import Misfit_waveform_smoothL1
from .Weighted_L1_L2 import Misfit_weighted_L1_and_L2
from .StudentT import Misfit_waveform_studentT
from .Envelope import Misfit_envelope
from .GlobalCorrelation import Misfit_global_correlation
from .Weci import Misfit_weighted_ECI
from .TravelTime import Misfit_traveltime
from .SoftDTW import Misfit_sdtw
from .WDGC import Misfit_weighted_DTW_GC
from .Wasserstein_sinkhorn import Misfit_wasserstein_sinkhorn
from .Normalized_Integration_method import Misfit_NIM

__all__ = [
    "Misfit",
    "Misfit_NIM",
    "Misfit_envelope",
    "Misfit_global_correlation",
    "Misfit_sdtw",
    "Misfit_traveltime",
    "Misfit_wasserstein_sinkhorn",
    "Misfit_waveform_L1",
    "Misfit_waveform_L2",
    "Misfit_waveform_SquaredL2",
    "Misfit_waveform_smoothL1",
    "Misfit_waveform_studentT",
    "Misfit_weighted_DTW_GC",
    "Misfit_weighted_ECI",
    "Misfit_weighted_L1_and_L2",
]
