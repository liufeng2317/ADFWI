'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-03-01 20:51:32
* LastEditors: LiuFeng
* LastEditTime: 2024-03-21 10:49:51
* Description: 

* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''
import numpy as np
import torch 
import numpy as np
from scipy.signal import butter, filtfilt

##################################################################################
#                   multi-frequency processing
##################################################################################
# def lowpass(x1, highcut, fn, order=1, axis=1, show=False):
#     x = copy.deepcopy(x1)
#     # Zero padding
#     padding = 512
#     x = np.hstack((x, np.zeros((x.shape[0], padding, x.shape[2]))))
#     nt = x.shape[axis]
#     # Bring the data to frequency domain
#     x_fft = fft.fft(x, n=nt, axis=axis)
#     # Calculate the highcut btween 0 to 1
#     scaled_highcut = 2*highcut/fn
#     # Generate the filter
#     b, a = butter(order, scaled_highcut, btype='lowpass', output="ba")
#     # Get the frequency response
#     w, h1 = freqz(b, a, worN=nt, whole=True)
#     h = np.diag(h1)
#     # Apply the filter in the frequency domain
#     fd = h @ x_fft
#     #Double filtering by the conjugate to make up the shift
#     h = np.diag(np.conjugate(h1))
#     fd = h @ fd
#     # Bring back to time domaine
#     f_inv = fft.ifft(fd, n=nt, axis=axis).real
#     f_inv = f_inv[:, :-padding, :]
#     return f_inv

# def adj_lowpass(x, highcut, fn, order, axis=1):
#     # Zero padding
#     padding = 512
#     x = np.hstack((x, np.zeros((x.shape[0], padding, x.shape[2]))))
#     nt = x.shape[axis]
#     # Bring the data to frequency domain
#     x_fft = np.fft.fft(x, n=nt, axis=axis)
#     # Calculate the highcut btween 0 to 1
#     scaled_highcut = 2*highcut / fn
#     # Generate the filter
#     b, a = butter(order, scaled_highcut, btype='lowpass', output="ba")
#     # Get the frequency response
#     w, h = freqz(b, a, worN=nt, whole=True)
#     # Get the conjugate of the filter
#     h_c = np.diag(np.conjugate(h))
#     # Apply the adjoint filter in the frequency domain
#     fd = h_c @ x_fft
#     # Double filtering by the conjugate to make up the shift
#     h_c = np.diag(h)
#     fd = h_c @ fd
#     # Bring back to time domaine
#     adj_f_inv = np.fft.ifft(fd, axis=axis).real
#     adj_f_inv = adj_f_inv[:, :-padding, :]
#     return adj_f_inv

def lowpass(x, highcut, fn, order=1, axis=1):
    """
    Apply low-pass filter in the time domain using filtfilt (zero-phase filtering).
    
    Parameters:
    x (np.ndarray): Input signal (3D array: [nsrc, nt, nrcv]).
    highcut (float): High cutoff frequency in Hz.
    fn (float): Sampling frequency in Hz.
    order (int): Order of the Butterworth filter.
    
    Returns:
    np.ndarray: Low-pass filtered signal.
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
    
    Parameters:
    x (np.ndarray): Input signal (3D array: [nsrc, nt, nrcv]).
    highcut (float): High cutoff frequency in Hz.
    fn (float): Sampling frequency in Hz.
    order (int): Order of the Butterworth filter.
    
    Returns:
    np.ndarray: Adjoint low-pass filtered signal.
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

            
def lpass(x1, x2, highcut, fn):
    """
        fn is the sampling frequency
    """
    x1_filtered, x2_filtered = Lfilter.apply(x1, x2, highcut, fn)
    return x1_filtered, x2_filtered



class Lfilter(torch.autograd.Function):
    
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