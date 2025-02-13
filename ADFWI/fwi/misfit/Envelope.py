'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: LiuFeng
* LastEditTime: 2024-05-15 19:30:38
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

from .base import Misfit
import torch
import numpy as np
import torch.nn.functional as F

@torch.jit.script
def hilbert(x: torch.Tensor) -> torch.Tensor:
    '''Perform Hilbert transform along the last axis of x.
    
    Args:
        x (Tensor): Input signal data.
        
    Returns:
        Tensor: Analytic signal with the same shape as `x`.
    '''
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
    '''Compute discrete difference along the last axis.
    
    Args:
        x (Tensor): Input tensor.
        dim (int): Axis along which to compute the difference (default: -1).
        same_size (bool): If True, pad the output to maintain the same size.
        
    Returns:
        Tensor: Discrete difference along specified axis.
    '''
    if same_size:
        return F.pad(x[..., 1:] - x[..., :-1], (1, 0))  # Pad to match original shape
    else:
        return x[..., 1:] - x[..., :-1]
    
@torch.jit.script
def unwrap(phi: torch.Tensor, dim: int = -1) -> torch.Tensor:
    '''Unwrap phase by correcting for phase discontinuities.
    
    Args:
        phi (Tensor): Phase tensor.
        dim (int): Axis along which to unwrap the phase (default: -1).
        
    Returns:
        Tensor: Unwrapped phase tensor.
    '''
    dphi = diff(phi, same_size=True)  # Calculate discrete phase difference
    dphi_m = ((dphi + np.pi) % (2 * np.pi)) - np.pi  # Map phase difference to [-pi, pi]
    dphi_m[(dphi_m == -np.pi) & (dphi > 0)] = np.pi  # Correct for edge cases
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < np.pi] = 0  # Adjust only where discontinuities exist
    return phi + phi_adj.cumsum(dim)  # Accumulate adjustments

class Misfit_envelope(Misfit):
    '''Compute the envelope misfit for initial velocity model estimation.
    
    References:
        Wu et al., 2014; Yuan et al., 2015
    
    Args:
        dt (float): Time sampling interval.
        p (float): Norm order for envelope difference.
        instaneous_phase (bool): If True, use instantaneous phase for misfit.
        norm (str): Norm type ("L1" or "L2") for final loss calculation.
    '''
    
    def __init__(self, dt: float = 1, p: float = 1.5, instaneous_phase: bool = False, norm: str = "L2") -> None:
        super().__init__()
        self.p = p
        self.instaneous_phase = instaneous_phase
        self.dt = dt
        self.norm = norm
        
    def forward(self, obs: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
        '''Compute the misfit between observed and synthetic waveforms.
        
        Args:
            obs (Tensor): Observed waveform [batch, trace, time].
            syn (Tensor): Synthetic waveform [batch, trace, time].
        
        Returns:
            Tensor: Envelope or phase difference loss.
        '''
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