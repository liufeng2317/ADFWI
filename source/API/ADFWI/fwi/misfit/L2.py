from .base import Misfit
import torch

class Misfit_waveform_L2(Misfit):
    """
    Waveform L2-norm difference misfit (Tarantola, 1984).

    This class computes the L2-norm misfit between the observed and synthetic waveforms.
    The L2-norm is calculated as the square root of the sum of squared residuals between
    corresponding points in the observed and synthetic waveforms.

    Parameters
    ----------
    dt : float, optional
        Time sampling interval (default is 1).

    """
    def __init__(self, dt=1) -> None:
        """
        Initialize the Misfit_waveform_L2 class.
        """
        super().__init__()
        self.dt = dt
    
    def forward(self, obs: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
        """
        Compute the L2-norm waveform misfit between observed and synthetic data.

        The L2-norm is computed as the square root of the sum of squared residuals, weighted
        by the time sampling interval `dt`.

        Parameters
        ----------
        obs : torch.Tensor
            The observed waveform, typically with shape (batch, channels, time).

        syn : torch.Tensor
            The synthetic waveform, typically with shape (batch, channels, time).

        Returns
        -------
        torch.Tensor
            The L2-norm misfit loss between the observed and synthetic waveforms.
        """
        # Create mask to handle traces where both observed and synthetic data are zero
        mask1 = torch.sum(torch.abs(obs), axis=1) == 0
        mask2 = torch.sum(torch.abs(syn), axis=1) == 0
        mask = ~(mask1 * mask2)

        # Calculate residuals by subtracting synthetic from observed data
        rsd = obs - syn

        # Compute the L2-norm loss as the square root of the sum of squared residuals, weighted by dt
        # Summation is done along the channels axis (axis=1) for each sample, then the square root
        # is taken and summed over all the valid (non-zero) samples
        loss = torch.sum(torch.sqrt(torch.sum(rsd * rsd * self.dt, axis=1)[mask]))
        
        return loss