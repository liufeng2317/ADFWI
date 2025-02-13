from .base import Misfit
import torch

class Misfit_global_correlation(Misfit):
    """
    Global correlation misfit function.

    This class computes the global correlation misfit between observed and synthetic waveforms, 
    focusing on their correlation structure across traces.

    Parameters
    ----------
    dt : float, optional
        Time sampling interval (default is 1).
    """
    def __init__(self, dt=1) -> None:
        """
        Initialize the Misfit_global_correlation.
        """
        super().__init__()
        self.dt = dt
    
    def forward(self, obs: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
        """
        Compute the global correlation misfit between observed and synthetic waveforms.

        This function calculates the correlation between each pair of observed and synthetic waveforms 
        across all traces and computes the global misfit.

        Parameters
        ----------
        obs : torch.Tensor
            Observed waveform with shape (batch, channels, traces).
        
        syn : torch.Tensor
            Synthetic waveform with shape (batch, channels, traces).
        
        Returns
        -------
        torch.Tensor
            Correlation-based misfit loss computed between the observed and synthetic waveforms.
        """
        mask1    = torch.sum(torch.abs(obs), axis=1) == 0
        mask2    = torch.sum(torch.abs(syn), axis=1) == 0
        mask     = ~(mask1 * mask2)
        
        # Initialize result tensor for storing residuals
        rsd = torch.zeros((obs.shape[0], obs.shape[2]), device=obs.device)

        # Loop through each trace and calculate the correlation misfit
        for itrace in range(obs.shape[2]):
            shot_idx  = torch.argwhere(mask[:, itrace])
            obs_trace = obs[shot_idx, :, itrace].squeeze(axis=1)  # Shape: (N, T)
            syn_trace = syn[shot_idx, :, itrace].squeeze(axis=1)  # Shape: (N, T)

            # Normalize the observed and synthetic traces
            obs_trace_norm = obs_trace.norm(dim=1, keepdim=True)
            syn_trace_norm = syn_trace.norm(dim=1, keepdim=True)
            
            obs_trace = obs_trace / obs_trace_norm
            syn_trace = syn_trace / syn_trace_norm
            
            # Calculate covariance and variances
            cov     = torch.mean(obs_trace * syn_trace, dim=1)  # Shape: (N,)
            var_obs = torch.var(obs_trace, dim=1)               # Shape: (N,)
            var_syn = torch.var(syn_trace, dim=1)               # Shape: (N,)

            # Compute correlation, with small value added to avoid division by zero
            corr = cov / (torch.sqrt(var_obs * var_syn) + 1e-8)

            # Set correlation to zero where both variances are zero
            corr[torch.isnan(corr)] = 0
            
            rsd[shot_idx, itrace] = -corr.reshape(-1, 1)
        
        # Compute the total loss, scaled by the time interval dt
        loss = torch.sum(rsd * self.dt)
        return loss
