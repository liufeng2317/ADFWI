'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-03-01 20:51:32
* LastEditors: LiuFeng
* LastEditTime: 2024-03-21 10:49:51
* Description: 

* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
import numpy as np
import torch 
import copy
import numpy.fft as fft
from scipy.signal import butter, hilbert, freqz
import matplotlib.pyplot as plt
import matplotlib as mlp
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import interpolate as intp

##################################################################################
#                   multi-frequency processing
##################################################################################
def lowpass(x1, highcut, fn, order=1, axis=1, show=False):
    x = copy.deepcopy(x1)

    # Zero padding
    padding = 512
    x = np.hstack((x, np.zeros((x.shape[0], padding, x.shape[2]))))

    nt = x.shape[axis]

    # Bring the data to frequency domain
    x_fft = fft.fft(x, n=nt, axis=axis)

    # Calculate the highcut btween 0 to 1
    scaled_highcut = 2*highcut/fn

    # Generate the filter
    b, a = butter(order, scaled_highcut, btype='lowpass', output="ba")

    # Get the frequency response
    w, h1 = freqz(b, a, worN=nt, whole=True)
    h = np.diag(h1)

    # Apply the filter in the frequency domain
    fd = h @ x_fft

    #Double filtering by the conjugate to make up the shift
    h = np.diag(np.conjugate(h1))
    fd = h @ fd

    # Bring back to time domaine
    f_inv = fft.ifft(fd, n=nt, axis=axis).real
    f_inv = f_inv[:, :-padding, :]

    return f_inv

def adj_lowpass(x, highcut, fn, order, axis=1):

    # Zero padding
    padding = 512
    x = np.hstack((x, np.zeros((x.shape[0], padding, x.shape[2]))))

    nt = x.shape[axis]

    # Bring the data to frequency domain
    x_fft = np.fft.fft(x, n=nt, axis=axis)

    # Calculate the highcut btween 0 to 1
    scaled_highcut = 2*highcut / fn

    # Generate the filter
    b, a = butter(order, scaled_highcut, btype='lowpass', output="ba")

    # Get the frequency response
    w, h = freqz(b, a, worN=nt, whole=True)

    # Get the conjugate of the filter
    h_c = np.diag(np.conjugate(h))

    # Apply the adjoint filter in the frequency domain
    fd = h_c @ x_fft

    # Double filtering by the conjugate to make up the shift
    h_c = np.diag(h)
    fd = h_c @ fd

    # Bring back to time domaine
    adj_f_inv = np.fft.ifft(fd, axis=axis).real
    adj_f_inv = adj_f_inv[:, :-padding, :]
    return adj_f_inv

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
        
        nb, ns, nt, nr = x1.shape
        device = x1.device.type
        
        x1_np = x1.detach()
        x2_np = x2.detach()
        
        x1_np = x1_np.squeeze(dim=0) #.numpy()
        x2_np = x2_np.squeeze(dim=0) #.numpy()
        
        x1_np, x2_np = data3d_to_2d(x1_np , x2_np)
        
        x1_np = torch.unsqueeze(x1_np, 0)
        x2_np = torch.unsqueeze(x2_np, 0)
        
        filtered1 = lowpass(x1_np.numpy(), highcut=highcut, fn=fn,
                           order=3, axis=1)
        
        filtered2 = lowpass(x2_np.numpy(), highcut=highcut, fn=fn,
                           order=3, axis=1)

        filtered1_3d, filtered2_3d = data2d_to_3d(
            torch.Tensor(filtered1[0, ...]),
            torch.Tensor(filtered2[0, ...]),
            ns, nr)
        
        # filtered1 = torch.tensor(filtered1_3d, device=device)
        # filtered2 = torch.tensor(filtered2_3d, device=device)
        return filtered1_3d.unsqueeze(0).to(device=device), filtered2_3d.unsqueeze(0).to(device=device)
    
    @staticmethod
    def backward(ctx, adj1, adj2):
        
        nb, ns, nt, nr = adj1.shape
        device = adj1.device.type
        
        x1_np = adj1.detach()
        x2_np = adj2.detach()
        
        x1_np = x1_np.squeeze(dim=0) # .numpy()
        x2_np = x2_np.squeeze(dim=0) # .numpy()
        
        x1_np, x2_np = data3d_to_2d(x1_np, x2_np)
        x1_np = torch.unsqueeze(x1_np, 0)
        x2_np = torch.unsqueeze(x2_np, 0)
        
        filtered1 = adj_lowpass(x1_np.numpy(), highcut=ctx.lpass_highcut,
                                fn=ctx.lpass_fn, order=3, axis=1)
        
        filtered2 = adj_lowpass(x2_np.numpy(), highcut=ctx.lpass_highcut,
                                fn=ctx.lpass_fn, order=3, axis=1)
        
        filtered1_3d, filtered2_3d = data2d_to_3d(
            torch.Tensor(filtered1[0, ...]),
            torch.Tensor(filtered2[0, ...]),
            ns, nr)
        
        # filtered1 = torch.tensor(filtered1_3d, device=device)
        # filtered2 = torch.tensor(filtered2_3d, device=device)
                   
        return filtered1_3d.unsqueeze(0).to(device=device), \
                filtered2_3d.unsqueeze(0).to(device=device),\
                    None,\
                    None