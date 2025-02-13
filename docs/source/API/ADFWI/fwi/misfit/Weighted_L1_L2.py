from .base import Misfit
from .L1 import Misfit_waveform_L1
from .L2 import Misfit_waveform_L2
import math
import torch


class Misfit_weighted_L1_and_L2(Misfit):
    """
    Weighted combination of L1 and L2 waveform misfit functions.

    This class combines the L1 and L2 misfit functions in a weighted manner, with the weight dynamically updated
    throughout the optimization process based on the current iteration.

    Some details can be found in:
    http://www.sce.carleton.ca/faculty/adler/talks/2013/rahmati-CMBES2013-weighted-L1-L2-pres.pdf

    Parameters
    ----------
    dt : float, optional
        Time sampling interval. Default is 1.

    max_iter : int, optional
        The maximum number of iterations for weight update. Default is 1000.

    Returns
    -------
    torch.Tensor
        The weighted misfit loss between the observed and synthetic waveforms.
    """
    def __init__(self, dt=1, max_iter=1000) -> None:
        """
        Initialize the Misfit_weighted_L1_and_L2 class.
        """
        super().__init__()
        self.dt = dt
        self.iter = 0
        self.max_iter = max_iter
        self.L1_fn = Misfit_waveform_L1(dt=self.dt)
        self.L2_fn = Misfit_waveform_L2(dt=self.dt)

    def forward(self, obs, syn):
        """
        Compute the weighted misfit between the observed and synthetic waveforms.

        The weight between L1 and L2 is dynamically updated during the optimization process.

        Parameters
        ----------
        obs : torch.Tensor
            The observed waveform with shape [num_shots, num_time_steps, num_receivers].

        syn : torch.Tensor
            The synthetic waveform with shape [num_shots, num_time_steps, num_receivers].

        Returns
        -------
        torch.Tensor
            The computed weighted loss combining L1 and L2 misfits.
        """
        N = self.max_iter
        w_i = 1 / (1 + math.exp(-(self.iter - N / 2)))  # Sigmoid weight function
        L1_loss = self.L1_fn.forward(obs=obs, syn=syn)
        L2_loss = self.L2_fn.forward(obs=obs, syn=syn)
        loss = w_i * L2_loss + (1 - w_i) * L1_loss
        self.iter += 1
        return loss
