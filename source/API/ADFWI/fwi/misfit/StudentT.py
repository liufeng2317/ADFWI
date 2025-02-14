from .base import Misfit
import torch

class Misfit_waveform_studentT(Misfit):
    """
    Student T Loss (Alvaro et al., 2023; Guo et al., 2023) for waveform misfit calculation.

    This loss function uses the Student's T-distribution to provide robustness against outliers
    in the waveform difference.

    References:
    -----------
    - Alvaro et al., 2023. "Robust Loss Functions for Seismic Inversion". (https://arxiv.org/pdf/2306.00753.pdf)
    - Guo et al., 2023. "Seismic Inversion with Student's T Loss Function". (https://watermark.silverchair.com/gxac096.pdf)

    Parameters
    ----------
    s : float, optional
        The number of degrees of freedom (usually 1 or 2). Default is 1.
        
    sigma : float, optional
        The scale parameter controlling the spread of the distribution. Default is 1.
        
    dt : float, optional
        Time interval between samples. Default is 1.
        
    obs : torch.Tensor
        The observed waveform with shape [num_shots, num_time_steps, num_receivers].
        
    syn : torch.Tensor
        The synthetic waveform with shape [num_shots, num_time_steps, num_receivers].

    Returns
    -------
    torch.Tensor
        The computed robust T loss.
    """
    
    def __init__(self, s=1, sigma=1, dt=1) -> None:
        """
        Initialize the Misfit_waveform_studentT class.
        """
        self.s = s
        self.sigma = sigma
        self.dt = dt

    def forward(self, obs, syn):
        """
        Compute the robust T loss between the observed and synthetic waveforms.

        Parameters
        ----------
        obs : torch.Tensor
            The observed waveform with shape [num_shots, num_time_steps, num_receivers].
            
        syn : torch.Tensor
            The synthetic waveform with shape [num_shots, num_time_steps, num_receivers].

        Returns
        -------
        torch.Tensor
            The computed robust T loss.
        """
        # Calculate the residual (difference between synthetic and observed)
        rsd = syn - obs
        # Compute the robust T loss
        loss = 0.5 * (self.s + 1) * torch.log(1 + rsd ** 2 / (self.s * self.sigma ** 2))
        # Sum over all the losses and scale by the time interval
        loss = torch.sum(loss * self.dt)
        return loss

