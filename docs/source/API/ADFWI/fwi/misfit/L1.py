from .base import Misfit
import torch

class Misfit_waveform_L1(Misfit):
    """
    Waveform L1-norm difference misfit (Tarantola, 1984).

    This class computes the L1-norm misfit between the observed and synthetic waveforms.
    The L1-norm is calculated as the sum of the absolute differences between corresponding
    points in the observed and synthetic waveforms.

    Parameters
    ----------
    dt : float, optional
        Time sampling interval (default is 1).

    """
    def __init__(self, dt=1) -> None:
        """
        Initialize the Misfit_waveform_L1 class.
        """
        super().__init__()
        self.dt = dt

    def forward(self, obs: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
        """
        Compute the L1-norm waveform misfit between observed and synthetic data.

        The L1-norm is computed as the sum of absolute residuals, weighted by the time sampling interval `dt`.

        Parameters
        ----------
        obs : torch.Tensor
            The observed waveform, typically with shape (batch, channels, time).

        syn : torch.Tensor
            The synthetic waveform, typically with shape (batch, channels, time).

        Returns
        -------
        torch.Tensor
            The L1-norm misfit loss between the observed and synthetic waveforms.
        """
        # Calculate residuals by subtracting synthetic from observed data
        rsd = obs - syn

        # Compute the L1-norm loss by summing the absolute value of residuals, weighted by dt
        loss = torch.sum(torch.abs(rsd * self.dt))
        
        return loss
