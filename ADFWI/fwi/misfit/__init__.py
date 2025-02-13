from .base  import Misfit
from .L1    import Misfit_waveform_L1
from .L2    import Misfit_waveform_L2
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
# from .Wasserstein_1 import Misfit_Wasserstein1
# from .Wasserstein_1d import Misfit_wasserstein_1d

from .Normalized_Integration_method import Misfit_NIM