"""Shared utility helpers used by examples and framework glue.

`ADFWI.utils` is a broad compatibility namespace. Keep package-level imports
stable for existing scripts, and move numerical behavior only with focused
before/after validation.
"""

from .utils import gpu2cpu, list2numpy, numpy2list, numpy2tensor, tensor2numpy
from .wavelets import wavelet
from .velocityDemo import (
    build_anomaly_background_model,
    build_layer_model,
    get_anomaly_model,
    get_linear_hess_model,
    get_linear_marmousi2_model,
    get_linear_vel_model,
    get_smooth_hess_model,
    get_smooth_layer_model,
    get_smooth_marmousi_model,
    get_smooth_valhall_model,
    load_hess_model,
    load_marmousi_model,
    load_overthrust_initial_model,
    load_overthrust_model,
    load_valhall_model,
    resample_marmousi_model,
    resample_overthrust_model,
)

# Keep the historical broad frequency-helper surface for existing notebooks.
from .frequency_domin_process import *  # noqa: F401,F403
