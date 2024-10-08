'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-05-05 19:03:11
* LastEditors: LiuFeng
* LastEditTime: 2024-05-05 19:23:13
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
from .base import Misfit
import torch


class Misfit_waveform_L1(Misfit):
    ''' Waveform difference L2-norm (Tarantola, 1984)
    
    Paraemters:
    -----------
        dt (float)      : time sampling interval
        obs (Tensors)   : the observed waveform 
        syn (Tensors)   : the synthetic waveform 
    '''
    def __init__(self,dt=1) -> None:
        super().__init__()
        self.dt = dt

    def forward(self,obs,syn):
        rsd = obs - syn 
        loss = torch.sum(torch.abs(rsd*self.dt))
        return loss