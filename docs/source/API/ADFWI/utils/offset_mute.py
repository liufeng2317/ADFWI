import numpy as np
from ADFWI.utils.utils import tensor2numpy
import torch

def mute_offset(rcv_x,src_x,dx,waveform,distance_threshold = 400):
    """
    Mute seismic data based on source-receiver offset.
    
    Parameters
    ----------
    rcv_x : torch.Tensor or np.ndarray
        Receiver positions, shape (n_shots, n_receivers).
    src_x : torch.Tensor or np.ndarray
        Source positions, shape (n_shots,).
    dx : float
        Spatial sampling interval (unit: meters).
    waveform : torch.Tensor
        Seismic waveform data, shape (n_shots, n_time, n_receivers).
    distance_threshold : float, optional
        Offset threshold for muting (unit: meters). Default is 400 m.

    Returns
    -------
    waveform : torch.Tensor
        Muted waveform with the same shape as the input.
    """
    rcv_x = tensor2numpy(rcv_x)
    src_x = tensor2numpy(src_x)
    for ishot in range(waveform.shape[0]):
        distance_mask = np.abs(rcv_x[ishot] - src_x[ishot]) < distance_threshold/dx
        distance_mask = ~distance_mask # [rcv]
        distance_mask_temp = torch.zeros_like(waveform[ishot]).to(waveform.device)
        distance_mask_temp[:,distance_mask.tolist()] = 1
        waveform[ishot] = waveform[ishot]*distance_mask_temp
    return waveform