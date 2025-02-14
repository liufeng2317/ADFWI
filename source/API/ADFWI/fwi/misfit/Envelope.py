from .base import Misfit
import torch
import numpy as np
import torch.nn.functional as F

@torch.jit.script
def hilbert(x: torch.Tensor) -> torch.Tensor:
    """
    Perform Hilbert transform along the last axis of x.

    This function computes the Hilbert transform of the input signal `x` along its last axis.
    It uses FFT-based approach to compute the analytic signal.

    Parameters
    ----------
    x : torch.Tensor
        Input signal data, a tensor of shape (..., N), where N is the length of the signal
        along the last axis.

    Returns
    -------
    torch.Tensor
        Analytic signal with the same shape as `x`. The result is the Hilbert transform of the
        input signal along the last axis.
    """
    device = x.device
    N = x.shape[-1] * 2  # Double the length for FFT
    Xf = torch.fft.fft(x, n=N)  # FFT on extended signal
    h = torch.zeros(N, dtype=Xf.dtype, device=device)  # Initialize multiplier array
    h[0] = 1
    h[1:(N + 1) // 2] = 2  # Set values to create analytic signal
    if N % 2 == 0:
        h[N // 2] = 1  # Special case for even-length signals
    return torch.fft.ifft(Xf * h)[..., :x.shape[-1]]  # Return Hilbert-transformed signal


@torch.jit.script
def diff(x: torch.Tensor, dim: int = -1, same_size: bool = False) -> torch.Tensor:
    """
    Compute discrete difference along the last axis.

    This function computes the discrete difference of the input tensor `x` along the specified
    axis (`dim`). If `same_size` is set to `True`, the output tensor is padded to maintain the
    same size as the input.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, with arbitrary shape. The difference will be computed along the last axis by default.
        
    dim : int, optional
        Axis along which to compute the difference (default is -1, i.e., the last axis).

    same_size : bool, optional
        If `True`, pads the output tensor to maintain the same size as the input tensor (default is `False`).

    Returns
    -------
    torch.Tensor
        The discrete difference along the specified axis, with the same shape as `x` if `same_size` is `True`, 
        or smaller along the specified axis if `same_size` is `False`.
    """
    if same_size:
        return F.pad(x[..., 1:] - x[..., :-1], (1, 0))  # Pad to match original shape
    else:
        return x[..., 1:] - x[..., :-1]
    
@torch.jit.script
def unwrap(phi: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Unwrap phase by correcting for phase discontinuities.

    This function unwraps a phase tensor `phi` by detecting and correcting phase discontinuities.
    The phase is adjusted such that the changes in phase are continuous along the specified axis.

    Parameters
    ----------
    phi : torch.Tensor
        Phase tensor to be unwrapped. The tensor is assumed to contain phase values in radians.

    dim : int, optional
        Axis along which to unwrap the phase (default is -1, i.e., the last axis).

    Returns
    -------
    torch.Tensor
        The unwrapped phase tensor, with phase discontinuities corrected along the specified axis.
    """
    dphi = diff(phi, same_size=True)  # Calculate discrete phase difference
    dphi_m = ((dphi + np.pi) % (2 * np.pi)) - np.pi  # Map phase difference to [-pi, pi]
    dphi_m[(dphi_m == -np.pi) & (dphi > 0)] = np.pi  # Correct for edge cases
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < np.pi] = 0  # Adjust only where discontinuities exist
    return phi + phi_adj.cumsum(dim)  # Accumulate adjustments

class Misfit_envelope(Misfit):
    """
    Compute the envelope misfit for initial velocity model estimation.

    This class computes the misfit between observed and synthetic waveforms based on their envelope or instantaneous phase.
    It is used for estimating the initial velocity model in seismic inversion.

    References:
        Wu et al., 2014; Yuan et al., 2015

    Parameters
    ----------
    dt : float, optional
        Time sampling interval (default is 1).
        
    p : float, optional
        Norm order for envelope difference (default is 1.5).
        
    instaneous_phase : bool, optional
        If True, use instantaneous phase for misfit (default is False).
        
    norm : str, optional
        Norm type ("L1" or "L2") for final loss calculation (default is "L2").
    """
    
    def __init__(self, dt: float = 1, p: float = 1.5, instaneous_phase: bool = False, norm: str = "L2") -> None:
        """
        Initialize the Misfit_envelope.
        """
        super().__init__()
        self.p = p
        self.instaneous_phase = instaneous_phase
        self.dt = dt
        self.norm = norm
        
    def forward(self, obs: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
        """
        Compute the misfit between observed and synthetic waveforms.

        This function calculates the misfit between the observed and synthetic waveforms either using their envelopes 
        or instantaneous phases, depending on the chosen option.

        Parameters
        ----------
        obs : torch.Tensor
            Observed waveform [batch, trace, time].
        
        syn : torch.Tensor
            Synthetic waveform [batch, trace, time].
        
        Returns
        -------
        torch.Tensor
            Envelope or phase difference loss computed between the observed and synthetic waveforms.
        """
        mask1    = torch.sum(torch.abs(obs),axis=1) == 0
        mask2    = torch.sum(torch.abs(syn),axis=1) == 0
        mask     = ~(mask1 * mask2)
        
        device = obs.device
        rsd = torch.zeros((obs.shape[0], obs.shape[2], obs.shape[1]), device=device)  # Residual storage
        
        for ishot in range(obs.shape[0]):
            trace_idx = torch.argwhere(mask[ishot]).reshape(-1)
            obs_shot = obs[ishot,:,trace_idx].squeeze(axis=0).T  # Transpose to [trace, time series]
            syn_shot = syn[ishot,:,trace_idx].squeeze(axis=0).T
            
            # Hilbert transform to get analytic signal
            analytic_signal_obs = hilbert(obs_shot)
            analytic_signal_syn = hilbert(syn_shot)
            
            # Compute envelopes (magnitude of analytic signals)
            envelopes_obs = torch.abs(analytic_signal_obs)
            envelopes_syn = torch.abs(analytic_signal_syn)
            
            if self.instaneous_phase:
                # Use instantaneous phase for misfit
                phase_obs = unwrap(torch.angle(analytic_signal_obs))
                phase_syn = unwrap(torch.angle(analytic_signal_syn))
                rsd[ishot,trace_idx,:] = (phase_obs - phase_syn).unsqueeze(0)
            else:
                # Compute envelope difference with norm p
                rsd[ishot,trace_idx,:] = (envelopes_syn**self.p - envelopes_obs**self.p).unsqueeze(0)

        # Compute final loss based on the selected norm
        if self.norm == "L1":
            loss = torch.sum(torch.abs(rsd))
        else:
            loss = 0.5 * torch.sum(rsd * rsd * self.dt)  # L2 loss with time weighting
        
        return loss