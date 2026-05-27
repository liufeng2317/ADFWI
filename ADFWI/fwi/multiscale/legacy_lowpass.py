"""Legacy multiscale low-pass filtering helpers.

This module is the implementation owner for the historical
``ADFWI.fwi.multiScaleProcessing`` low-pass API. It preserves the SciPy
``filtfilt`` forward path and matching adjoint-style backward path used by old
FWI scripts and by ``LegacyLowPassFilter``.

The implementation intentionally keeps the CPU NumPy/SciPy round trip. Previous
bv1.2 comparisons showed that the pure torch FIR ``LowPassFilter`` is useful and
differentiable, but it is not a numerical drop-in replacement for inversion
cases that rely on this legacy Butterworth path.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, filtfilt

def lowpass(x, highcut, fn, order=1, axis=1):
    """Apply the legacy Butterworth low-pass filter with SciPy ``filtfilt``.

    Parameters
    ----------
    x:
        NumPy array shaped ``[shot, time, receiver]``.
    highcut:
        Cutoff frequency in Hz.
    fn:
        Sampling frequency in Hz.
    order:
        Butterworth filter order.
    axis:
        Time axis. Historical ADFWI callers use ``axis=1``.
    """
    # Nyquist frequency
    nyquist = 0.5 * fn
    # Normalized cutoff frequency
    normal_cutoff = highcut / nyquist
    # Butterworth filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter using filtfilt along the time axis (axis=1)
    # Apply the filter for all sources and receivers simultaneously (vectorized operation)
    y = np.empty_like(x)
    for i in range(x.shape[2]):  # Loop over receivers (nrcv)
        y[:, :, i] = filtfilt(b, a, x[:, :, i], axis=axis)
    
    return y


def adj_lowpass(x, highcut, fn, order=1, axis=1):
    """Apply the legacy adjoint-style low-pass used by ``Lfilter.backward``."""
    # Nyquist frequency
    nyquist = 0.5 * fn
    # Normalized cutoff frequency
    normal_cutoff = highcut / nyquist
    # Butterworth filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Reverse the time axis (nt), apply the filter, and reverse back (vectorized operation)
    adj_filtered = np.empty_like(x)
    for i in range(x.shape[2]):  # Loop over receivers (nrcv)
        # Apply the filter to the reversed signal and reverse it back
        adj_filtered[:, :, i] = np.flip(filtfilt(b, a, np.flip(x[:, :, i], axis=axis), axis=axis), axis=axis)
    return adj_filtered

def data2d_to_3d(data1_2d, data2_2d, ns, nr):
    """Restore flattened ``[time, shot * receiver]`` data to 3D torch tensors."""

    nt = data1_2d.shape[0]
    
    data1_3d = torch.empty((ns, nt, nr))
    data2_3d = torch.empty((ns, nt, nr))
    
    for i in range(ns):
        data1_3d[i, :, :] = data1_2d[:, i*nr:(i+1)*nr]
        data2_3d[i, :, :] = data2_2d[:, i*nr:(i+1)*nr]
    return data1_3d, data2_3d


def data3d_to_2d(data1_3d, data2_3d):
    """Flatten ``[shot, time, receiver]`` torch tensors for legacy SciPy calls."""

    ns, nt, nr = data2_3d.shape
    x1_2d = torch.empty((nt, ns*nr))
    x2_2d = torch.empty((nt, ns*nr))
    for i in range(ns):
        x1_2d[:, i*nr:(i+1)*nr] = data1_3d[i, ...]
        x2_2d[:, i*nr:(i+1)*nr] = data2_3d[i, ...]
        
    return x1_2d, x2_2d

            
def lpass(x1, x2, highcut, fn):
    """Apply the legacy autograd low-pass filter to a synthetic/observed pair."""

    x1_filtered, x2_filtered = Lfilter.apply(x1, x2, highcut, fn)
    return x1_filtered, x2_filtered



class Lfilter(torch.autograd.Function):
    """Autograd wrapper around the legacy CPU NumPy/SciPy low-pass path.

    Inputs are detached, filtered through SciPy on CPU, then restored to the
    original device. The backward pass applies ``adj_lowpass`` to preserve the
    historical custom-gradient behavior.
    """
    
    @staticmethod
    def forward(ctx, x1, x2, highcut, fn):
        ctx.lpass_highcut = highcut
        ctx.lpass_fn = fn
        
        ns, nt, nr = x1.shape
        device = x1.device
        
        x1,x2 = x1.detach(),x2.detach()
        x1,x2 = data3d_to_2d(x1 , x2)
        x1,x2 = torch.unsqueeze(x1, 0),torch.unsqueeze(x2, 0)
        
        filtered1 = lowpass(x1.numpy(), highcut=highcut, fn=fn, order=6, axis=1)
        filtered2 = lowpass(x2.numpy(), highcut=highcut, fn=fn, order=6, axis=1)

        filtered1, filtered2 = data2d_to_3d(
                torch.Tensor(filtered1[0, ...]),
                torch.Tensor(filtered2[0, ...]),
                ns, nr
            )
        filtered1 = torch.tensor(filtered1, device=device)
        filtered2 = torch.tensor(filtered2, device=device)
        return filtered1,filtered2
    
    @staticmethod
    def backward(ctx, adj1, adj2):
        
        ns, nt, nr = adj1.shape
        device = adj1.device.type
        
        x1,x2  = adj1.detach(),adj2.detach()
        x1, x2 = data3d_to_2d(x1, x2)
        x1,x2  = torch.unsqueeze(x1, 0),torch.unsqueeze(x2, 0)
        
        filtered1 = adj_lowpass(x1.numpy(), highcut=ctx.lpass_highcut, fn=ctx.lpass_fn, order=6, axis=1)
        
        filtered2 = adj_lowpass(x2.numpy(), highcut=ctx.lpass_highcut, fn=ctx.lpass_fn, order=6, axis=1)
        
        filtered1, filtered2 = data2d_to_3d(
            torch.Tensor(filtered1[0, ...]),
            torch.Tensor(filtered2[0, ...]),
            ns, nr)
        
        filtered1 = torch.tensor(filtered1, device=device)
        filtered2 = torch.tensor(filtered2, device=device)
                   
        return filtered1.to(device=device), \
                filtered2.to(device=device),\
                    None,\
                    None
