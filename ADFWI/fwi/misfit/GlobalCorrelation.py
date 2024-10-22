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
    
    # Two loops for calculate the misfits (shots & traces)
    # def forward(self,obs,syn):
    #     obs_norm = torch.sqrt(torch.sum(obs*obs,axis=1))
    #     syn_norm = torch.sqrt(torch.sum(syn*syn,axis=1))
    #     rsd = torch.zeros((obs.shape[0],obs.shape[2])).to(obs.device)
    #     for ishot in range(obs.shape[0]):
    #         obs_shot_norm = obs[ishot]/obs_norm[ishot]
    #         syn_shot_norm = syn[ishot]/syn_norm[ishot]
    #         for itrace in range(obs.shape[2]):
    #             obs_trace_norm = obs_shot_norm[:,itrace].reshape(1,-1)
    #             syn_trace_norm = syn_shot_norm[:,itrace].reshape(1,-1)
    #             corrcoef_input = torch.vstack((obs_trace_norm,syn_trace_norm))
    #             rsd[ishot,itrace] = -torch.corrcoef(corrcoef_input)[0][1]
    #     loss = torch.sum(rsd*self.dt)
    #     return loss
    
    # One loop for calculate the misfits (traces)
    def forward(self, obs, syn):
        # Compute norms
        obs_norm = obs.norm(dim=1, keepdim=True)  # Shape: (N, 1, M)
        syn_norm = syn.norm(dim=1, keepdim=True)  # Shape: (N, 1, M)

        # Normalize the observed and synthetic waveforms
        obs_normalized = obs / obs_norm
        syn_normalized = syn / syn_norm

        # Initialize result tensor
        rsd = torch.empty(obs.shape[0], obs.shape[2], device=obs.device)

        # Compute correlation for each trace
        for itrace in range(obs.shape[2]):
            obs_trace = obs_normalized[:, :, itrace]  # Shape: (N, T)
            syn_trace = syn_normalized[:, :, itrace]  # Shape: (N, T)

            # Calculate covariance and variances
            cov = torch.mean(obs_trace * syn_trace, dim=1)  # Shape: (N,)
            var_obs = torch.var(obs_trace, dim=1)  # Shape: (N,)
            var_syn = torch.var(syn_trace, dim=1)  # Shape: (N,)

            # Avoid division by zero by masking
            corr = cov / (torch.sqrt(var_obs * var_syn) + 1e-8)  # Adding small value to avoid div by zero

            # Handle the case where both variances are zero
            corr[torch.isnan(corr)] = 0  # If both variances are zero, set correlation to zero

            rsd[:, itrace] = -corr
        
        loss = torch.sum(rsd * self.dt)
        return loss
