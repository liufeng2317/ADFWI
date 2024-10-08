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
from .GlobalCorrelation import Misfit_global_correlation
from .Envelope import Misfit_envelope

class Misfit_weighted_ECI(Misfit):
    """Weighted envelope misfit and global correlation misfit (Song Chao et al., 2023, IEEE TGRS)
    
    Paraemters:
    -----------
        dt (float)      : time sampling interval
        p (float)       : the norm order of the reuslt
        instaneous_phase (bool) : use instaeous phase or amplitude for the misfit
        obs (Tensors)   : the observed waveform 
        syn (Tensors)   : the synthetic waveform 
    """
    def __init__(self,max_iter=1000,dt=1,p=1.5,instaneous_phase=False) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.iter = 0
        self.dt = dt
        self.p  = p
        self.instaneous_phase   = instaneous_phase
        self.GC_fn              = Misfit_global_correlation(dt=self.dt)
        self.Envelope_fn        = Misfit_envelope(dt=self.dt,p=self.p,instaneous_phase=self.instaneous_phase)
        
    def forward(self,obs,syn):
        N = self.max_iter
        w_i = 1/(1+math.exp(-(self.iter - N/2)))
        GCN_loss = self.GC_fn.forward(obs=obs,syn=syn)
        ECI_loss = self.Envelope_fn.forward(obs=obs,syn=syn)
        loss = w_i*GCN_loss + (1-w_i)*ECI_loss
        self.iter += 1
        return loss