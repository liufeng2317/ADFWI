'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-05-05 19:03:11
* LastEditors: LiuFeng
* LastEditTime: 2024-05-05 19:23:13
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''
from .base import Misfit
import torch


class Misfit_waveform_L1(Misfit):
    '''Waveform L1-norm difference misfit (Tarantola, 1984)

    Parameters:
    -----------
        dt (float)      : Time sampling interval.
        obs (Tensor)    : Observed waveform.
        syn (Tensor)    : Synthetic waveform.
    '''
    def __init__(self, dt=1) -> None:
        super().__init__()
        self.dt = dt

    def forward(self, obs, syn):
        '''Compute the L1-norm waveform misfit between observed and synthetic data.

        Args:
            obs (Tensor): Observed waveform.
            syn (Tensor): Synthetic waveform.
        
        Returns:
            Tensor: L1-norm misfit loss.
        '''
        # Calculate residuals by subtracting synthetic from observed data
        rsd = obs - syn

        # Compute the L1-norm loss by summing the absolute value of residuals, weighted by dt
        loss = torch.sum(torch.abs(rsd * self.dt))
        
        return loss