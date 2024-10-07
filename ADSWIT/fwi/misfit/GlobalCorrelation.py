from .base import Misfit
import torch

class Misfit_global_correlation(Misfit):
    """ global correlation misfit functions
    
    Paraemters:
    ------------
        obs (Tensors)   : the observed waveform 
        syn (Tensors)   : the synthetic waveform 
    """
    def __init__(self,dt=1) -> None:
        super().__init__()
        self.dt = dt
    
    def forward(self,obs,syn):
        obs_norm = torch.sqrt(torch.sum(obs*obs,axis=1))
        syn_norm = torch.sqrt(torch.sum(syn*syn,axis=1))
        rsd = torch.zeros((obs.shape[0],obs.shape[2])).to(obs.device)
        for ishot in range(obs.shape[0]):
            obs_shot_norm = obs[ishot]/obs_norm[ishot]
            syn_shot_norm = syn[ishot]/syn_norm[ishot]
            for itrace in range(obs.shape[2]):
                obs_trace_norm = obs_shot_norm[:,itrace].reshape(1,-1)
                syn_trace_norm = syn_shot_norm[:,itrace].reshape(1,-1)
                corrcoef_input = torch.vstack((obs_trace_norm,syn_trace_norm))
                rsd[ishot,itrace] = -torch.corrcoef(corrcoef_input)[0][1]
        loss = torch.sum(rsd*self.dt)
        return loss