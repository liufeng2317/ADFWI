from .base import Misfit
import torch


class Misfit_waveform_studentT(Misfit):
    """ Roubust T Loss (Alvaro et al., 2023; Guo et al., 2023)
            https://arxiv.org/pdf/2306.00753.pdf
            https://watermark.silverchair.com/gxac096.pdf
    
    Paraemters:
    -----------
        s (float)       : the number of degree of freedom (usually 1 or 2). 
        sigma (float)   : a scale parameter
        obs (Tensors)   : the observed waveform 
        syn (Tensors)   : the synthetic waveform 
    """
    def __init__(self,s=1,sigma=1,dt=1) -> None:
        self.s      = s
        self.sigma  = sigma
        self.dt     = dt

    def forward(self,obs,syn):
        rsd     = syn - obs
        loss    = 0.5*(self.s + 1)*torch.log(1 + rsd**2/(self.s*self.sigma**2))
        loss    = torch.sum(loss*self.dt)
        return loss