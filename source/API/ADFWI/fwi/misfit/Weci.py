from .base import Misfit
import torch
import math
from .GlobalCorrelation import Misfit_global_correlation
from .Envelope import Misfit_envelope

class Misfit_weighted_ECI(Misfit):
    """
    Weighted envelope misfit and global correlation misfit (Song Chao et al., 2023, IEEE TGRS)
    
    This class computes the weighted combination of the envelope misfit and global correlation misfit functions,
    where the weight is updated iteratively during optimization. The weight increases as the number of iterations progresses,
    allowing the inversion to progressively shift focus from the global correlation misfit to the envelope misfit.

    Parameters
    ----------
    max_iter : int, optional
        The maximum number of iterations for the weight update. Default is 1000.
        
    dt : float, optional
        Time interval between samples. Default is 1.
        
    p : float, optional
        The norm order for the envelope difference. Default is 1.5.
        
    instaneous_phase : bool, optional
        If True, use instantaneous phase for misfit instead of amplitude. Default is False.

    Returns
    -------
    torch.Tensor
        The computed weighted loss, combining envelope and global correlation misfits.
    """
    
    def __init__(self, max_iter=1000, dt=1, p=1.5, instaneous_phase=False) -> None:
        """
        Initialize the Misfit_weighted_ECI class.
        """
        super().__init__()
        self.max_iter = max_iter
        self.iter = 0
        self.dt = dt
        self.p = p
        self.instaneous_phase = instaneous_phase
        self.GC_fn = Misfit_global_correlation(dt=self.dt)
        self.Envelope_fn = Misfit_envelope(dt=self.dt, p=self.p, instaneous_phase=self.instaneous_phase)

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
            The computed weighted loss, combining envelope and global correlation misfits.
        """
        N = self.max_iter
        w_i = 1 / (1 + math.exp(-(self.iter - N / 2)))  # Sigmoid weight function
        GCN_loss = self.GC_fn.forward(obs=obs, syn=syn)
        ECI_loss = self.Envelope_fn.forward(obs=obs, syn=syn)
        loss = w_i * GCN_loss + (1 - w_i) * ECI_loss
        self.iter += 1
        return loss
