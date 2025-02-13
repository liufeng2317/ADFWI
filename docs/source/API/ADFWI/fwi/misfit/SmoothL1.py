from .base import Misfit
import torch
import torch.nn as nn


class Misfit_waveform_smoothL1(Misfit): 
    """
    Smoothed L1 misfit function (Ross Girshick, 2015, International conference on computer vision)
    https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html

    Parameters:
    -----------
    dt (float): Time interval between samples, default is 1.

    Methods:
    --------
    forward(obs, syn):
        Computes the smoothed L1 loss between observed and synthetic waveforms.
    
    Args:
    -----
    obs (Tensor): The observed waveform of shape [num_time_steps, num_shots*num_receivers_per_shot].
    syn (Tensor): The synthetic waveform of shape [num_time_steps, num_shots*num_receivers_per_shot].

    Returns:
    --------
    Tensor: The computed smoothed L1 loss value.
    """
    def __init__(self, dt=1) -> None:
        """
        Initialize the Misfit_waveform_smoothL1 class.
        """
        self.dt = dt
        super().__init__()

    def forward(self, obs, syn):
        """
        Compute the smoothed L1 loss between the observed and synthetic waveforms.

        Args:
        -----
        obs (Tensor): The observed waveform.
        syn (Tensor): The synthetic waveform.

        Returns:
        --------
        Tensor: The computed smoothed L1 loss.
        """
        loss_fun = nn.SmoothL1Loss(reduction='sum', beta=1.0)
        loss = loss_fun(obs, syn) * self.dt  # Apply time interval scaling
        return loss
