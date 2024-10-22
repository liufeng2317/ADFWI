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
    
    def forward(self,obs:torch.Tensor,syn:torch.Tensor):
        rsd = obs - syn 
        loss = torch.sum(torch.sqrt(torch.sum(rsd*rsd*self.dt,axis=1)))
        return loss