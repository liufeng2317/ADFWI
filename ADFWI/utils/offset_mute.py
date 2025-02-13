import numpy as np
from ADFWI.utils.utils import tensor2numpy
import torch

def mute_offset(rcv_x,src_x,dx,waveform,distance_threshold = 400):
    """ mute from the shot to each side
        unit = m
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
        
    