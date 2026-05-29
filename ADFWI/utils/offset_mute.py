import numpy as np
from ADFWI.utils.utils import tensor2numpy
import torch


def mute_offset(rcv_x,src_x,dx,waveform,distance_threshold = 400):
    """Apply the legacy offset mute in place.

    Receiver positions and source positions are grid indices. ``dx`` converts
    ``distance_threshold`` from meters into grid-index distance. Receivers with
    distance strictly smaller than ``distance_threshold / dx`` are zeroed;
    farther receivers are kept.
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
