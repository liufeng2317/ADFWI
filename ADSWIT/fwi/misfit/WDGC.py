'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-05-12 09:09:07
* LastEditors: LiuFeng
* LastEditTime: 2024-05-12 09:09:14
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
from .base import Misfit
import torch
import math
from typing import Optional
from .GlobalCorrelation import Misfit_global_correlation
from .SoftDTW import Misfit_sdtw

class Misfit_weighted_DTW_GC(Misfit):
    """Weighted softdtw and global correlation misfit (Song Chao et al., 2023, IEEE TGRS)
    
    Paraemters:
    -----------
        dt (float)      : time sampling interval
        gamma (float)           : Regularization parameter. It should be strictly positive. Lower is less smoothed (closer to true DTW).
        sparse_sampling (int)   : down-sampling the signal for accelerating inversion
        obs (Tensors)           : the observed waveform [shot,time sampling,receiver]
        syn (Tensors)           : the synthetic waveform [shot,time sampling,receiver]

    """
    def __init__(self,max_iter=1000,gamma:Optional[float]=1,sparse_sampling:Optional[int]=1,dt:Optional[float]=1) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.iter = 0
        self.dt = dt
        self.GC_fn       = Misfit_global_correlation(dt=self.dt)
        self.Softdtw_fn  = Misfit_sdtw(gamma=gamma,sparse_sampling=sparse_sampling,dt=dt)
        
    def forward(self,obs,syn):
        N = self.max_iter
        w_i = 1/(1+math.exp(-(self.iter - N/2)))
        GCN_loss = self.GC_fn.forward(obs=obs,syn=syn)
        DTW_loss = self.Softdtw_fn.forward(obs=obs,syn=syn)
        loss = w_i*GCN_loss + (1-w_i)*DTW_loss
        self.iter += 1
        return loss