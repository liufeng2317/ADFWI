from .base import Misfit
import torch
import torch.nn as nn


class Misfit_waveform_smoothL1(Misfit): 
    """smoothed L1 misfit function (Ross Girshick, 2015, International conference on computer vision)
        https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
        
    Paraemters:
    -----------
        obs (Tensors)   : the observed waveform 
        syn (Tensors)   : the synthetic waveform 
    """
    def __init__(self,dt = 1) -> None:
        self.dt = dt
        super().__init__()
    
    def forward(self,obs,syn):
        loss_fun = nn.SmoothL1Loss(reduction='sum',beta=1.0)
        loss     = loss_fun(obs,syn)*self.dt 
        return loss