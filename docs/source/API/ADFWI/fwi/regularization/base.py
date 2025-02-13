from abc import abstractmethod
import torch
import numpy as np
from typing import Optional


def regular_StepLR(iter, step_size, alpha, gamma=0.8):
    """
    Regular learning rate scheduler using exponential decay.

    Parameters
    ----------
    iter : int
        The current iteration.
    step_size : int
        The step size for updating the learning rate.
    alpha : float
        The initial learning rate.
    gamma : float, optional
        The decay factor for the learning rate. Default is 0.8.

    Returns
    -------
    float
        The updated learning rate.
    """
    n = iter // step_size
    return alpha * np.power(gamma, n)


class Regularization():
    """
    A base class for applying regularization on a grid with grid spacing in the x and z directions.

    Parameters
    ----------
    nx : int
        Number of grid points in the x-direction.
    nz : int
        Number of grid points in the z-direction.
    dx : float
        Grid size in the x-direction (m).
    dz : float
        Grid size in the z-direction (m).
    alphax : float
        The regularization factor in the x-direction.
    alphaz : float
        The regularization factor in the z-direction.
    step_size : int, optional
        The update step for alphax and alphaz. Default is 1000.
    gamma : float, optional
        The update step decay factor. Default is 1.
    device : str, optional
        The device to use ('cpu' or 'cuda'). Default is 'cpu'.
    dtype : torch.dtype, optional
        The data type for the tensors. Default is `torch.float32`.
    """
    def __init__(self, nx: int, nz: int, dx: float, dz: float,
                 alphax: float, alphaz: float,
                 step_size: Optional[int] = 1000, gamma: Optional[int] = 1,
                 device="cpu", dtype=torch.float32) -> None:
        self.iter = 0
        self.step_size = step_size
        self.gamma = gamma
        self.alphax = alphax
        self.alphaz = alphaz
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.device = device
        self.dtype = dtype
    
    @abstractmethod
    def forward(self):
        """
        Abstract method to compute the forward pass for regularization.

        This method must be implemented in subclasses.

        Returns
        -------
        torch.Tensor
            The regularized output.
        """
        pass