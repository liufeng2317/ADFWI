from .base import Misfit
import pysdtw
import torch
from typing import Optional

class Misfit_sdtw(Misfit):
    """
    Soft-DTW misfit function, calculates the soft Dynamic Time Warping (DTW) divergence.

    Origin:
    -------
    https://github.com/toinsson/pysdtw

    Soft-DTW divergence:
    ---------------------
    https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.SoftDTWLossPyTorch.html
    Mathieu Blondel, Arthur Mensch & Jean-Philippe Vert. “Differentiable divergences between time series,” 
    International Conference on Artificial Intelligence and Statistics, 2021.

    Parameters
    ----------
    gamma : float, optional
        Regularization parameter. It should be strictly positive. Lower values lead to less smoothing (closer to true DTW). Default is 1.
    
    sparse_sampling : int, optional
        Down-sampling the signal for accelerating inversion. Default is 1 (no down-sampling).
        
    dt : float, optional
        Time interval between samples. Default is 1.
        
    obs : torch.Tensor
        The observed waveform with shape [num_shots, num_time_steps, num_receivers].
        
    syn : torch.Tensor
        The synthetic waveform with shape [num_shots, num_time_steps, num_receivers].

    Returns
    -------
    torch.Tensor
        The computed soft-DTW loss.
    """
    
    def __init__(self, gamma: Optional[float] = 1, sparse_sampling: Optional[int] = 1, dt: Optional[float] = 1) -> None:
        """
        Initialize the Misfit_sdtw class.
        """
        super().__init__()
        self.gamma = gamma
        self.sparse_sampling = sparse_sampling
        self.dt = dt
    
    def forward(self, obs, syn):
        """
        Compute the soft-DTW divergence between the observed and synthetic waveforms.

        Parameters
        ----------
        obs : torch.Tensor
            The observed waveform with shape [num_shots, num_time_steps, num_receivers].
            
        syn : torch.Tensor
            The synthetic waveform with shape [num_shots, num_time_steps, num_receivers].

        Returns
        -------
        torch.Tensor
            The computed soft-DTW loss.
        """
        device = obs.device
        # Create masks to exclude zero entries in both observed and synthetic waveforms
        mask1 = torch.sum(torch.abs(obs), axis=1) == 0
        mask2 = torch.sum(torch.abs(syn), axis=1) == 0
        mask = ~(mask1 * mask2)
        
        # Define the pairwise L2 squared distance function
        fun = pysdtw.distance.pairwise_l2_squared_exact

        # Preallocate the output tensor for the results
        rsd = torch.zeros((obs.shape[0], obs.shape[2]), device=device)

        # Initialize SoftDTW once
        sdtw = pysdtw.SoftDTW(gamma=self.gamma, dist_func=fun, use_cuda=device != "cpu")

        # Loop over each shot to compute the soft-DTW divergence for each trace
        for ishot in range(obs.shape[0]):
            trace_idx = torch.argwhere(mask[ishot]).reshape(-1)
            # Extract and down-sample the observed and synthetic shots
            obs_shot = obs[ishot, ::self.sparse_sampling, trace_idx].squeeze().T.unsqueeze(2)  # Shape: [num_traces, T, 1]
            syn_shot = syn[ishot, ::self.sparse_sampling, trace_idx].squeeze().T.unsqueeze(2)  # Shape: [num_traces, T, 1]

            # Compute soft-DTW divergences
            sdtw_obs = sdtw(obs_shot, obs_shot)
            sdtw_syn = sdtw(syn_shot, syn_shot)
            sdtw_obs_syn = sdtw(obs_shot, syn_shot)
            
            # Compute the divergence as a soft-DTW difference
            std = sdtw_obs_syn - 0.5 * (sdtw_obs + sdtw_syn)
            rsd[ishot, trace_idx] = std.reshape(1, -1)

        # Compute the total loss by summing over all shots and applying the time interval scaling
        loss = torch.sum(rsd * self.dt)
        return loss