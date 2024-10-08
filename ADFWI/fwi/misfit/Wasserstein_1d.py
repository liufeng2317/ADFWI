from geomloss import SamplesLoss
import numpy as np
import torch
from .base import Misfit
from typing import Optional
from ot.lp import wasserstein_1d

class Misfit_wasserstein_1d(Misfit):
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
    def __init__(self,dt:Optional[float]=1,sparse_sampling:Optional[int] = 1,p:Optional[int]=2) -> None:
        super().__init__()
        self.dt = dt
        self.sparse_sampling = sparse_sampling
        self.p = p

    def forward(self,obs,syn):
        device = obs.device
        # enforce sum to one on the support
        x_torch = torch.tensor(np.arange(obs.shape[1])*self.dt).to(device)
        # Compute the Wasserstein 1D with torch backend
        loss = 0
        for ishot in range(obs.shape[0]):
            for ircv in range(obs.shape[-1]):
                obs_trace = obs[ishot,:,ircv]
                syn_trace = syn[ishot,:,ircv]
                obs_trace = obs_trace/torch.sum(obs_trace)
                syn_trace = syn_trace/torch.sum(syn_trace)
                loss_trace = wasserstein_1d(x_torch, x_torch, obs_trace, syn_trace, p=self.p)
                loss = loss + loss_trace
        return loss 
    

    
# !!! Too abstract, contains a quadruple loop
from scipy.optimize import linear_sum_assignment
def gsot(syn: torch.Tensor, obs: torch.Tensor, eta: float):
    y = obs.transpose(1,2)
    y_pred = obs.transpose(1,2)
    device = y.device
    loss = torch.tensor(0, dtype=torch.float).to(device)
    for s in range(y.shape[0]):
        for r in range(y.shape[1]):
            nt = y.shape[-1]
            c = np.zeros([nt, nt])
            for i in range(nt):
                for j in range(nt):
                    c[i, j] = (
                        eta * (i-j)**2 +
                        (y_pred.detach()[s, r, i]-y[s, r, j])**2
                    )
            row_ind, col_ind = linear_sum_assignment(c)
            y_sigma = y[s, r, col_ind]
            loss = (
                loss + (
                    eta * torch.tensor(row_ind-col_ind).to(device)**2 +
                    (y_pred[s, r]-y_sigma)**2
                ).sum()
            )
    return loss