'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: LiuFeng
* LastEditTime: 2024-05-15 19:30:38
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

from .base import Misfit
import pysdtw
import torch
from typing import Optional

class Misfit_sdtw(Misfit):
    """soft-dtw misfit function
        origin:https://github.com/toinsson/pysdtw
        soft-DTW divergence :
            https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.SoftDTWLossPyTorch.html
            Mathieu Blondel, Arthur Mensch & Jean-Philippe Vert. “Differentiable divergences between time series”, International Conference on Artificial Intelligence and Statistics, 2021.
    
    Paraemters:
    -----------
        gamma (float)           : Regularization parameter. It should be strictly positive. Lower is less smoothed (closer to true DTW).
        sparse_sampling (int)   : down-sampling the signal for accelerating inversion
        obs (Tensors)           : the observed waveform [shot,time sampling,receiver]
        syn (Tensors)           : the synthetic waveform [shot,time sampling,receiver]
    """
    def __init__(self,gamma:Optional[float]=1,sparse_sampling:Optional[int]=1,dt:Optional[float]=1) -> None:
        super().__init__()
        self.gamma  = gamma
        self.sparse_sampling = sparse_sampling
        self.dt = dt
    
    def forward(self, obs, syn):
        device = obs.device
        mask1    = torch.sum(torch.abs(obs),axis=1) == 0
        mask2    = torch.sum(torch.abs(syn),axis=1) == 0
        mask     = ~(mask1 * mask2)
        
        fun = pysdtw.distance.pairwise_l2_squared_exact

        # Preallocate the output tensor
        rsd = torch.zeros((obs.shape[0], obs.shape[2]), device=device)

        # Initialize SoftDTW once
        sdtw = pysdtw.SoftDTW(gamma=self.gamma, dist_func=fun, use_cuda=device != "cpu")

        for ishot in range(obs.shape[0]):
            trace_idx = torch.argwhere(mask[ishot]).reshape(-1)
            obs_shot = obs[ishot, ::self.sparse_sampling, trace_idx].squeeze().T.unsqueeze(2) # [trace, T, 1]
            syn_shot = syn[ishot, ::self.sparse_sampling, trace_idx].squeeze().T.unsqueeze(2) # [trace, T, 1]

            # Compute soft-DTW divergences
            sdtw_obs     =  sdtw(obs_shot, obs_shot)
            sdtw_syn     =  sdtw(syn_shot, syn_shot)
            sdtw_obs_syn =  sdtw(obs_shot, syn_shot)
            std =sdtw_obs_syn - 0.5 * (sdtw_obs + sdtw_syn)   
            rsd[ishot,trace_idx] = std.reshape(1,-1)

        loss = torch.sum(rsd * self.dt)
        return loss
