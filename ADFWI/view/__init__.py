"""Plotting helpers for models, surveys, waveforms, and inversion summaries.

This package intentionally does not select a Matplotlib backend. Applications,
notebooks, validation scripts, or tests own backend selection.
"""

from .boundary_condition import plot_bcx_bcz, plot_damp
from .inverted_loss_model import (
    animate_inversion_process,
    plot_initial_and_inverted,
    plot_misfit,
)
from .survey import plot_survey, plot_wavelet
from .velocity_model import (
    plot_eps_delta_gamma,
    plot_lam_mu,
    plot_model,
    plot_vp_rho,
    plot_vp_vs_rho,
)
from .waveform import plot_waveform2D, plot_waveform_trace, plot_waveform_wiggle

__all__ = [
    "animate_inversion_process",
    "plot_bcx_bcz",
    "plot_damp",
    "plot_eps_delta_gamma",
    "plot_initial_and_inverted",
    "plot_lam_mu",
    "plot_misfit",
    "plot_model",
    "plot_survey",
    "plot_vp_rho",
    "plot_vp_vs_rho",
    "plot_waveform2D",
    "plot_waveform_trace",
    "plot_waveform_wiggle",
    "plot_wavelet",
]
