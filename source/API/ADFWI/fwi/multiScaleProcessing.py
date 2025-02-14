import numpy as np
import torch 
import numpy as np
from scipy.signal import butter, filtfilt

def lowpass(x, highcut, fn, order=1, axis=1):
    """
    Apply low-pass filter in the time domain using filtfilt (zero-phase filtering).
    
    Parameters
    ----------
    x : np.ndarray
        Input signal (3D array: [nsrc, nt, nrcv]).
    highcut : float
        High cutoff frequency in Hz.
    fn : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the Butterworth filter. Default is 1.
    axis : int, optional
        Axis along which to apply the filter. Default is 1 (time axis).
    
    Returns
    -------
    np.ndarray
        Low-pass filtered signal.
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
    """
    Apply adjoint low-pass filter in the time domain.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal (3D array: [nsrc, nt, nrcv]).
    highcut : float
        High cutoff frequency in Hz.
    fn : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the Butterworth filter. Default is 1.
    axis : int, optional
        Axis along which to apply the filter. Default is 1 (time axis).
    
    Returns
    -------
    np.ndarray
        Adjoint low-pass filtered signal.
    """
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
    nt = data1_2d.shape[0]
    
    data1_3d = torch.empty((ns, nt, nr))
    data2_3d = torch.empty((ns, nt, nr))
    
    for i in range(ns):
        data1_3d[i, :, :] = data1_2d[:, i*nr:(i+1)*nr]
        data2_3d[i, :, :] = data2_2d[:, i*nr:(i+1)*nr]
    return data1_3d, data2_3d


def data3d_to_2d(data1_3d, data2_3d):
    ns, nt, nr = data2_3d.shape
    x1_2d = torch.empty((nt, ns*nr))
    x2_2d = torch.empty((nt, ns*nr))
    for i in range(ns):
        x1_2d[:, i*nr:(i+1)*nr] = data1_3d[i, ...]
        x2_2d[:, i*nr:(i+1)*nr] = data2_3d[i, ...]
        
    return x1_2d, x2_2d


class Lfilter(torch.autograd.Function):
    """
    A custom autograd function for applying low-pass filtering to two signals in the forward pass
    and computing the adjoint of the low-pass filter in the backward pass.
    """
    
    @staticmethod
    def forward(ctx, x1, x2, highcut, fn):
        """
        Forward pass of the low-pass filter.
        
        Parameters
        ----------
        x1 : torch.Tensor
            First input signal (3D tensor: [nsrc, nt, nrcv]).
        x2 : torch.Tensor
            Second input signal (3D tensor: [nsrc, nt, nrcv]).
        highcut : float
            High cutoff frequency for the low-pass filter in Hz.
        fn : float
            Sampling frequency in Hz.
        
        Returns
        -------
        torch.Tensor, torch.Tensor
            Filtered versions of x1 and x2.
        """
        ctx.lpass_highcut = highcut
        ctx.lpass_fn = fn
        
        ns, nt, nr = x1.shape
        device = x1.device
        
        # Detach tensors to avoid tracking gradients in the low-pass filtering
        x1, x2 = x1.detach(), x2.detach()
        x1, x2 = data3d_to_2d(x1, x2)
        x1, x2 = torch.unsqueeze(x1, 0), torch.unsqueeze(x2, 0)
        
        # Apply the low-pass filter using numpy arrays
        filtered1 = lowpass(x1.numpy(), highcut=highcut, fn=fn, order=6, axis=1)
        filtered2 = lowpass(x2.numpy(), highcut=highcut, fn=fn, order=6, axis=1)

        # Convert back to the original 3D tensor shape
        filtered1, filtered2 = data2d_to_3d(
            torch.Tensor(filtered1[0, ...]),
            torch.Tensor(filtered2[0, ...]),
            ns, nr
        )

        # Ensure the tensors are on the same device as the input
        filtered1 = torch.tensor(filtered1, device=device)
        filtered2 = torch.tensor(filtered2, device=device)
        
        return filtered1, filtered2
    
    @staticmethod
    def backward(ctx, adj1, adj2):
        """
        Backward pass of the low-pass filter for computing the gradients.
        
        Parameters
        ----------
        adj1 : torch.Tensor
            The gradient with respect to the first input signal (3D tensor: [nsrc, nt, nrcv]).
        adj2 : torch.Tensor
            The gradient with respect to the second input signal (3D tensor: [nsrc, nt, nrcv]).
        
        Returns
        -------
        torch.Tensor, torch.Tensor, None, None
            The gradients with respect to the inputs x1 and x2.
        """
        ns, nt, nr = adj1.shape
        device = adj1.device.type
        
        # Detach tensors to avoid tracking gradients during the adjoint computation
        x1, x2 = adj1.detach(), adj2.detach()
        x1, x2 = data3d_to_2d(x1, x2)
        x1, x2 = torch.unsqueeze(x1, 0), torch.unsqueeze(x2, 0)
        
        # Apply the adjoint low-pass filter using numpy arrays
        filtered1 = adj_lowpass(x1.numpy(), highcut=ctx.lpass_highcut, fn=ctx.lpass_fn, order=6, axis=1)
        filtered2 = adj_lowpass(x2.numpy(), highcut=ctx.lpass_highcut, fn=ctx.lpass_fn, order=6, axis=1)
        
        # Convert back to the original 3D tensor shape
        filtered1, filtered2 = data2d_to_3d(
            torch.Tensor(filtered1[0, ...]),
            torch.Tensor(filtered2[0, ...]),
            ns, nr
        )

        # Ensure the tensors are on the same device as the input
        filtered1 = torch.tensor(filtered1, device=device)
        filtered2 = torch.tensor(filtered2, device=device)
        
        return filtered1.to(device=device), filtered2.to(device=device), None, None

def lpass(x1, x2, highcut, fn):
    """
    Apply low-pass filtering to two signals using a Butterworth filter.
    
    Parameters
    ----------
    x1 : np.ndarray
        First input signal to be filtered.
    x2 : np.ndarray
        Second input signal to be filtered.
    highcut : float
        High cutoff frequency in Hz for the low-pass filter.
    fn : float
        Sampling frequency in Hz.
    
    Returns
    -------
    np.ndarray, np.ndarray
        The filtered versions of x1 and x2.
    """
    x1_filtered, x2_filtered = Lfilter.apply(x1, x2, highcut, fn)
    return x1_filtered, x2_filtered