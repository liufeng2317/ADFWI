from .base import Misfit
import torch
import math
from typing import Optional
from .GlobalCorrelation import Misfit_global_correlation
from .SoftDTW import Misfit_sdtw

class Misfit_weighted_DTW_GC(Misfit):
    """
    Weighted soft-DTW and global correlation misfit (Song Chao et al., 2023, IEEE TGRS)

    This class computes the weighted combination of the soft-DTW and global correlation misfit functions,
    where the weight is updated iteratively during optimization. The weight increases as the number of iterations progresses,
    allowing the inversion to progressively shift focus from the global correlation misfit to the soft-DTW misfit.

    Parameters
    ----------
    max_iter : int, optional
        The maximum number of iterations for the weight update. Default is 1000.
        
    gamma : float, optional
        Regularization parameter for soft-DTW. It should be strictly positive. Lower values lead to less smoothing (closer to true DTW).
        Default is 1.
        
    sparse_sampling : int, optional
        Down-sampling factor for the signals to accelerate inversion. Default is 1 (no down-sampling).
        
    dt : float, optional
        Time interval between samples. Default is 1.

    Returns
    -------
    torch.Tensor
        The computed weighted loss, combining soft-DTW and global correlation misfits.
    """
    
    def __init__(self, max_iter=1000, gamma: Optional[float] = 1, sparse_sampling: Optional[int] = 1, dt: Optional[float] = 1) -> None:
        """
        Initialize the Misfit_weighted_DTW_GC class.
        """
        super().__init__()
        self.max_iter = max_iter
        self.iter = 0
        self.dt = dt
        self.GC_fn = Misfit_global_correlation(dt=self.dt)
        self.Softdtw_fn = Misfit_sdtw(gamma=gamma, sparse_sampling=sparse_sampling, dt=dt)

    def forward(self, obs, syn):
        """
        Compute the weighted misfit between the observed and synthetic waveforms.

        The weight is updated in each iteration based on the progress of the optimization.
        
        Parameters
        ----------
        obs : torch.Tensor
            The observed waveform with shape [num_shots, num_time_steps, num_receivers].
            
        syn : torch.Tensor
            The synthetic waveform with shape [num_shots, num_time_steps, num_receivers].

        Returns
        -------
        torch.Tensor
            The computed weighted loss, combining soft-DTW and global correlation misfits.
        """
        N = self.max_iter
        w_i = 1 / (1 + math.exp(-(self.iter - N / 2)))  # Sigmoid weight function
        GCN_loss = self.GC_fn.forward(obs=obs, syn=syn)
        DTW_loss = self.Softdtw_fn.forward(obs=obs, syn=syn)
        loss = w_i * GCN_loss + (1 - w_i) * DTW_loss
        self.iter += 1
        return loss