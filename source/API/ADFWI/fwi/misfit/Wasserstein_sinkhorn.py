from geomloss import SamplesLoss
import numpy as np
import torch
from .base import Misfit
from typing import Optional

class Misfit_wasserstein_sinkhorn(Misfit):
    """
    Compute the Wasserstein Sinkhorn misfit for observed and synthetic waveforms.

    This class computes the misfit between observed and synthetic waveforms using the Sinkhorn approximation
    of the Wasserstein distance. It is used to measure the similarity between time-series signals with regularization.

    References:
        - https://github.com/dfdazac/wassdistance
        - https://www.kernel-operations.io/geomloss/_auto_examples/comparisons/plot_gradient_flows_1D.html

    Parameters
    ----------
    dt : float, optional
        Time sampling interval (default is 1).
        
    p : float, optional
        Order of the Wasserstein distance (default is 2).
        
    blur : float, optional
        Regularization parameter for Sinkhorn divergence (default is 1).
        
    scaling : float, optional
        Scaling factor for the loss function (default is 0.5).
        
    sparse_sampling : int, optional
        Down-sampling factor for accelerating inversion (default is 1).
        
    loss_method : str, optional
        Method used for Sinkhorn loss calculation. Options are 'sinkhorn' or 'exact' (default is 'sinkhorn').
    """
    
    def __init__(self, dt: Optional[float] = 1, p: Optional[float] = 2, blur: Optional[float] = 1, 
                 scaling: Optional[float] = 0.5, sparse_sampling: Optional[int] = 1, 
                 loss_method: Optional[str] = 'sinkhorn') -> None:
        """
        Initialize the Misfit_wasserstein_sinkhorn class.
        """
        super().__init__()
        self.dt = dt
        self.sparse_sampling = sparse_sampling
        self.p = p
        self.blur = blur
        self.scaling = scaling
        self.loss_method = loss_method

    def forward(self, obs, syn):
        """
        Compute the Wasserstein Sinkhorn loss between observed and synthetic waveforms.

        Parameters
        ----------
        obs : torch.Tensor
            The observed waveform with shape [num_shots, num_time_steps, num_receivers].

        syn : torch.Tensor
            The synthetic waveform with shape [num_shots, num_time_steps, num_receivers].

        Returns
        -------
        torch.Tensor
            The computed Wasserstein Sinkhorn loss.
        """
        device = obs.device
        # Mask to ignore empty traces
        mask1 = torch.sum(torch.abs(obs), axis=1) == 0
        mask2 = torch.sum(torch.abs(syn), axis=1) == 0
        mask = ~(mask1 * mask2)
        
        # Define the Wasserstein misfit function
        rsd = torch.zeros((obs.shape[0], obs.shape[2])).to(device)
        for ishot in range(obs.shape[0]):
            trace_idx = torch.argwhere(mask[ishot]).reshape(-1)
            
            # Initialize the Sinkhorn loss function
            misfit_fun = SamplesLoss(loss=self.loss_method, p=self.p, blur=self.blur, scaling=self.scaling)
            
            # Down-sample the observed and synthetic traces
            obs_shot = obs[ishot, ::self.sparse_sampling, trace_idx].squeeze(axis=0).T  # [trace, amplitude]
            syn_shot = syn[ishot, ::self.sparse_sampling, trace_idx].squeeze(axis=0).T  # [trace, amplitude]
            
            # Create the time list for the traces
            tlist = torch.from_numpy(np.arange(obs_shot.shape[1]) * self.dt).to(device).reshape(1, -1)
            tlist = torch.ones_like(obs_shot) * tlist  # Broadcast time across all traces
            
            # Stack time and amplitude for the traces
            obs_shot = torch.stack((tlist, obs_shot), dim=-1)  # [trace, samples, time and amplitude]
            syn_shot = torch.stack((tlist, syn_shot), dim=-1)  # [trace, samples, time and amplitude]
            
            # Compute the Sinkhorn divergence
            std = misfit_fun(obs_shot, syn_shot)
            rsd[ishot, trace_idx] = std.reshape(1, -1)
        
        # Sum the loss and scale by the time sampling interval
        loss = torch.sum(rsd * rsd * self.dt)
        return loss
