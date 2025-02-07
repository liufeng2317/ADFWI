"""
* Modified from: https://github.com/seisfwi/SWIT
* Original Author: Haipeng Li
* Original Author Email: haipeng@stanford.edu
=========================================================
* Author: Liu Feng (SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: Liu Feng
* LastEditTime: 2024-05-15 19:30:38
* Description: 
* Copyright (c) 2024 by Liu Feng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
"""

from typing import Optional
from .receiver import Receiver
from .source import Source
from ADFWI.utils import list2numpy
from ADFWI.view import plot_survey
import numpy as np

class Survey(object):
    """Survey class describes the seismic acquisition geometry (2D). I assume 
    that all sources share the same receivers, time samples, and time interval.

    Parameters
    ----------
    source : Source 
        Source object
    receiver : Receiver
        Receiver object
    receiver_mask: The index of useful receiver at each shot
        numpy: [shot number, receiver number], by default None
        this parameter is useful for design the special observation system (line one-side)
                * v v v v v v v v
                    * v v v v v v
                        * v v v v
                            * v v 
                (* means source, v means receiver)
    device : str, optional
        Device for computation: cpu or gpu, by default 'cpu'
    cpu_num : int, optional
        Maximum number of CPU cores, if cpu, by default 1
    gpu_num : int, optional
        Maximum number of GPU cards, if cuda, by default 1
    """
    def __init__(self,source:Source,receiver:Receiver,receiver_masks=None,receiver_masks_obs=True) -> None:
        self.source         = source
        self.receiver       = receiver
        # receive mask  -> mask some of the receiver are useful while other are not
        self.receiver_masks = None
        # receiver_masks_obs -> mark if the obs waveform need to be masked or not
        self.receiver_masks_obs = receiver_masks_obs
        if receiver_masks is not None:
            self.set_receiver_masks(receiver_masks)
    
    def set_receiver_masks(self,receiver_masks):
        src_x,src_z = list2numpy(self.source.loc_x),list2numpy(self.source.loc_z)
        rcv_x,rcv_z = list2numpy(self.receiver.loc_x),list2numpy(self.receiver.loc_z)
        if receiver_masks.shape[0] == len(src_x) and receiver_masks.shape[1] == len(rcv_x):        
            self.receiver_masks = receiver_masks
        else:
            raise ValueError(
                "Receiver Mask Errror: the number of receiver/source are not equal to the Mask"
            )
        
    
    def __repr__(self):
        """ Reimplement the repr function for printing the survey information
        """
        info = f"Survey Information:\n"
        info += repr(self.source)
        info += "\n"
        info += repr(self.receiver)
        return info
    
    def plot(self,model_data,**kwargs):
        src_x = list2numpy(self.source.loc_x)
        src_z = list2numpy(self.source.loc_z)
        rcv_x = list2numpy(self.receiver.loc_x)
        rcv_z = list2numpy(self.receiver.loc_z)
        
        plot_survey(src_x,src_z,rcv_x,rcv_z,model_data,**kwargs)
    
    def plot_single_shot(self,model_data,src_idx,**kwargs):
        src_x = list2numpy(self.source.loc_x[src_idx])
        src_z = list2numpy(self.source.loc_z[src_idx])
        rcv_x = list2numpy(self.receiver.loc_x)
        rcv_z = list2numpy(self.receiver.loc_z)
        if self.receiver_masks is None:
            receiver_mask = np.ones(len(rcv_x))
        else:
            receiver_mask = self.receiver_masks[src_idx]
        rcv_x = rcv_x[np.argwhere(receiver_mask)]
        rcv_z = rcv_z[np.argwhere(receiver_mask)]
        
        plot_survey(src_x,src_z,rcv_x,rcv_z,model_data,**kwargs)
        