from .base import Misfit
import pysdtw
import torch
from typing import Optional

class Misfit_sdtw(Misfit):
    """soft-dtw misfit function
        origin:https://github.com/toinsson/pysdtw
        soft-DTW divergence :
            https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.SoftDTWLossPyTorch.html
            Mathieu Blondel, Arthur Mensch & Jean-Philippe Vert. “Differentiable divergences between time series”, International Conference on Artificial Intelligence and Statistics, 2021.
    
    Paraemters:
    -----------
        gamma (float)           : Regularization parameter. It should be strictly positive. Lower is less smoothed (closer to true DTW).
        sparse_sampling (int)   : down-sampling the signal for accelerating inversion
        obs (Tensors)           : the observed waveform [shot,time sampling,receiver]
        syn (Tensors)           : the synthetic waveform [shot,time sampling,receiver]
    """
    def __init__(self,gamma:Optional[float]=1,sparse_sampling:Optional[int]=1,dt:Optional[float]=1) -> None:
        super().__init__()
        self.gamma  = gamma
        self.sparse_sampling = sparse_sampling
        self.dt = dt
        
    def forward(self,obs,syn):
        device = obs.device
        # optionally choose a pairwise distance function
        fun = pysdtw.distance.pairwise_l2_squared_exact

        # create the SoftDTW distance function
        rsd = torch.zeros((obs.shape[0],obs.shape[2])).to(obs.device)
        for ishot in range(obs.shape[0]):
            sdtw = pysdtw.SoftDTW(gamma=self.gamma, dist_func=fun, use_cuda= False if device=="cpu" else True)
            obs_shot = obs[ishot,::self.sparse_sampling,:].T
            syn_shot = syn[ishot,::self.sparse_sampling,:].T
            obs_shot = torch.unsqueeze(obs_shot,2).to(device=device)
            syn_shot = torch.unsqueeze(syn_shot,2).to(device=device)
            # soft-DTW divergence 
            std = sdtw(obs_shot,syn_shot) - 0.5*(sdtw(obs_shot,obs_shot) + sdtw(syn_shot,syn_shot))
            rsd[ishot] = std
        loss = torch.sum(rsd*self.dt)
        return loss
