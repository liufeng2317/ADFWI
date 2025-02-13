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

class Misfit_global_correlation(Misfit):
    """ global correlation misfit functions
    
    Paraemters:
    ------------
        obs (Tensors)   : the observed waveform 
        syn (Tensors)   : the synthetic waveform 
    """
    def __init__(self,dt=1) -> None:
        super().__init__()
        self.dt = dt
    
    # One loop for calculate the misfits (traces)
    def forward(self, obs, syn):
        """Compute the global correlation misfit between observed and synthetic waveforms.
        
        Args:
            obs (Tensor): Observed waveform, shape (batch, channels, traces).
            syn (Tensor): Synthetic waveform, shape (batch, channels, traces).
        
        Returns:
            Tensor: Correlation-based misfit loss.
        """
        mask1    = torch.sum(torch.abs(obs),axis=1) == 0
        mask2    = torch.sum(torch.abs(syn),axis=1) == 0
        mask     = ~(mask1 * mask2)
        
        # Initialize result tensor
        rsd = torch.zeros((obs.shape[0], obs.shape[2]), device=obs.device)

        # Compute correlation for each trace
        for itrace in range(obs.shape[2]):
            shot_idx  = torch.argwhere(mask[:,itrace])
            obs_trace = obs[shot_idx, :, itrace].squeeze(axis=1)  # Shape: (N, T)
            syn_trace = syn[shot_idx, :, itrace].squeeze(axis=1)  # Shape: (N, T)

            obs_trace_norm = obs_trace.norm(dim=1, keepdim=True)
            syn_trace_norm = syn_trace.norm(dim=1, keepdim=True)
            
            obs_trace = obs_trace/obs_trace_norm
            syn_trace = syn_trace/syn_trace_norm
            
            # Calculate covariance and variances
            cov     = torch.mean(obs_trace * syn_trace, dim=1)  # Shape: (N,)
            var_obs = torch.var(obs_trace, dim=1)               # Shape: (N,)
            var_syn = torch.var(syn_trace, dim=1)               # Shape: (N,)

            # Avoid division by zero by masking
            corr = cov / (torch.sqrt(var_obs * var_syn) + 1e-8)  # Adding small value to avoid div by zero

            # Handle the case where both variances are zero
            corr[torch.isnan(corr)] = 0  # If both variances are zero, set correlation to zero
            
            rsd[shot_idx, itrace] = -corr.reshape(-1,1)
        
        loss = torch.sum(rsd * self.dt)
        return loss