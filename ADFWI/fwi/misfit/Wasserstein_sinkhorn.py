'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: LiuFeng
* LastEditTime: 2024-05-15 19:30:38
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

from geomloss import SamplesLoss
import numpy as np
import torch
from .base import Misfit
from typing import Optional

class Misfit_wasserstein_sinkhorn(Misfit):
    """ Wasserstein distance (sinkhorn): 
        https://github.com/dfdazac/wassdistance
        https://www.kernel-operations.io/geomloss/_auto_examples/comparisons/plot_gradient_flows_1D.html
    
    Paraemters:
    -----------
        dt (float)              : time sampling interval
        sparse_sampling (int)   : down-sampling the signal for accelerating inversion
        obs (Tensors)           : the observed waveform 
        syn (Tensors)           : the synthetic waveform 
    """
    def __init__(self,dt:Optional[float]=1,p:Optional[float]=2,blur:Optional[float]=1,scaling:Optional[float]=0.5,sparse_sampling:Optional[int] = 1,loss_method:Optional[str]='sinkhorn') -> None:
        super().__init__()
        self.dt = dt
        self.sparse_sampling = sparse_sampling
        self.p = p
        self.blur = blur
        self.scaling = scaling
        self.loss_method = loss_method

    def forward(self,obs,syn):
        device = obs.device
        mask1    = torch.sum(torch.abs(obs),axis=1) == 0
        mask2    = torch.sum(torch.abs(syn),axis=1) == 0
        mask     = ~(mask1 * mask2)
        p=self.p
        blur=self.blur
        # define the misfit function
        rsd = torch.zeros((obs.shape[0],obs.shape[2])).to(device)
        for ishot in range(obs.shape[0]):
            trace_idx = torch.argwhere(mask[ishot]).reshape(-1)
            misfit_fun = SamplesLoss(loss=self.loss_method,p=p,blur=blur,scaling=self.scaling)
            obs_shot = obs[ishot,::self.sparse_sampling,trace_idx].squeeze(axis=0).T                  # [trace,amplitude]
            syn_shot = syn[ishot,::self.sparse_sampling,trace_idx].squeeze(axis=0).T                  # [trace,amplitude]
            # concate the time list
            tlist = torch.from_numpy(np.arange(obs_shot.shape[1])*self.dt).to(device).reshape(1,-1)
            tlist = torch.ones_like(obs_shot)*tlist
            obs_shot = torch.stack((tlist,obs_shot),dim=-1) # [trace,samples,tlist and amplitude]
            syn_shot = torch.stack((tlist,syn_shot),dim=-1) # [trace,samples,tlist and amplitude]
            # sinkhorn divergence
            std = misfit_fun(obs_shot,syn_shot)
            rsd[ishot,trace_idx] = std.reshape(1,-1)
        loss = torch.sum(rsd*rsd*self.dt)
        return loss