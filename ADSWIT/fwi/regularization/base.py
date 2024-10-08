from abc import abstractmethod
import torch
import numpy as np
from typing import Optional


def regular_StepLR(iter,step_size,alpha,gamma=0.8):
    n = iter//step_size
    return alpha*np.power(gamma,n)

def _l1_norm(x):
    return torch.sum(torch.abs(x))

def _l2_norm(x,eps = 1e-9):
    return torch.sqrt(torch.sum(x*x)+eps)

class Regularization():
    def __init__(self,nx:int,nz:int,dx:float,dz:float,
                 alphax:float,alphaz:float,step_size:Optional[int]=1000,gamma:Optional[int]=1) -> None:
        """
        Parameters:
        -------------
            nx (int)        : Number of grid points in x-direction
            nz (int)        : Number of grid points in z-direction
            dx (float)      : Grid size in x-direction (m)
            dz (float)      : Grid size in z-direction (m)
            alphax (float)  : the regular factor in x-direction
            alphaz (float)  : the regular factor in z-direction
            step_size (int) : the update step for alphax and alphaz 
            gamma (float)   : the update step decay factor 
        """
        self.iter       = 0
        self.step_size  = step_size
        self.gamma      = gamma
        self.alphax     = alphax
        self.alphaz     = alphaz
        self.nx         = nx
        self.nz         = nz
        self.dx         = dx
        self.dz         = dz
    
    @abstractmethod
    def forward(self):
        pass
