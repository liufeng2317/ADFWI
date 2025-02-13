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

class Misfit_waveform_L2(Misfit):
    '''Waveform L2-norm difference misfit (Tarantola, 1984)
    
    Parameters:
    -----------
        dt (float)      : Time sampling interval.
        obs (Tensor)    : Observed waveform.
        syn (Tensor)    : Synthetic waveform.
    '''
    def __init__(self, dt=1) -> None:
        super().__init__()
        self.dt = dt
    
    def forward(self, obs: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
        '''Compute the L2-norm waveform misfit between observed and synthetic data.

        Args:
            obs (Tensor): Observed waveform.
            syn (Tensor): Synthetic waveform.
        
        Returns:
            Tensor: L2-norm misfit loss.
        '''
        mask1    = torch.sum(torch.abs(obs),axis=1) == 0
        mask2    = torch.sum(torch.abs(syn),axis=1) == 0
        mask     = ~(mask1 * mask2)
        
        # Calculate residuals by subtracting synthetic from observed data
        rsd = obs - syn

        # Compute the L2-norm loss as the square root of the sum of squared residuals, weighted by dt
        # Summation along axis=1 (channels) for each sample, then take square root and sum over all samples
        loss = torch.sum(torch.sqrt(torch.sum(rsd * rsd * self.dt, axis=1)[mask]))
        
        return loss
