from .base import Misfit
from .L1 import Misfit_waveform_L1
from .L2 import Misfit_waveform_L2
import math
import torch


class Misfit_weighted_L1_and_L2(Misfit):
    """ Some detail can be find in:
        http://www.sce.carleton.ca/faculty/adler/talks/2013/rahmati-CMBES2013-weighted-L1-L2-pres.pdf

    Paraemters:
    -----------
        dt (float)      : time sampling interval
        max_iter (int)  : the maximum iteration
        obs (Tensors)   : the observed waveform 
        syn (Tensors)   : the synthetic waveform 
    """
    def __init__(self,dt=1,max_iter=1000) -> None:
        super().__init__()
        self.dt     = dt
        self.iter   = 0
        self.max_iter = max_iter
        self.L1_fn = Misfit_waveform_L1(dt=self.dt)
        self.L2_fn = Misfit_waveform_L2(dt=self.dt)
        
    def forward(self,obs,syn):
        N = self.max_iter
        w_i = 1/(1+math.exp(-(self.iter - N/2)))
        L1_loss = self.L1_fn.forward(obs=obs,syn=syn)
        L2_loss = self.L2_fn.forward(obs=obs,syn=syn)
        loss = w_i*L2_loss + (1-w_i)*L1_loss
        self.iter += 1
        return loss