
'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: LiuFeng
* LastEditTime: 2024-05-15 19:30:38
* Description: Thanks to Dr. Deng Bao for modifying this objective function
* Bao Deng (University of Science and Technology of China)
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

from .base import Misfit
import torch
import torch.nn.functional as F

""" 
NOTE!!!

This program implements an experimental traveltime objective function for waveform data
analysis. The following are key considerations and methods used in this approach:

1. **Cross-Correlation Calculation**:
   - The function `cross_correlation` computes the cross-correlation between observed and synthetic waveforms.
   - Cross-correlation is truncated to retain only the center portion of the result to match the length of the original waveform, avoiding the extended size caused by padding.

2. **Non-Differentiability of `argmax`**:
   - Traditional traveltime misfit calculations often rely on finding the peak location of the cross-correlation using `argmax`, which is non-differentiable.
   - This makes it unsuitable for gradient-based optimization methods, as backpropagation requires differentiable operations.

3. **Differentiable Approximation with Softmax**:
   - To make the process compatible with gradient-based methods, we use a softmax-weighted sum over indices in `trvaletime_difference`.
   - Specifically, `tt_loss` approximates the peak location by calculating `sum(softmax * indices)`, where `beta` controls the softness of the weighting. A high beta value sharpens the softmax, mimicking the `argmax` behavior more closely.
   - This alternative, while differentiable, has yet to be fully validated for robustness in noisy or amplitude-varying environments.

4. **Experimental Status**:
   - This objective function remains under development and testing, particularly in challenging scenarios like high noise or amplitude discrepancies.
   - The value of `beta` is adjustable and may need tuning based on specific data properties.

5. **Misfit Calculation**:
   - The `Misfit_traveltime` class normalizes both observed and synthetic waveforms before calculating the residuals.
   - For each source-receiver pair, it computes a traveltime difference and accumulates the absolute values as the total loss.

This code is designed to test the performance of a differentiable approximation for traveltime misfit calculation, though improvements may be needed after further testing and evaluation.
"""

def calculate_time_shift(wave1,wave2,beta=100):
    """Calculate the travel time difference based on cross-correlation."""
    cross_corr = F.conv1d(
        wave1.view(1, 1, -1),
        wave2.view(1, 1, -1),
        padding=wave1.numel()-1
    ).view(-1)
    weights = F.softmax(beta*cross_corr,dim=0)
    tt_lags = torch.arange(-wave1.numel() + 1, wave1.numel(), device=wave1.device, dtype=torch.float32)
    time_shift = torch.sum(weights*tt_lags)
    return time_shift

class Misfit_traveltime(Misfit):
    '''Waveform L2-norm difference misfit (Tarantola, 1984)
    
    Parameters:
    -----------
        dt (float)      : Time sampling interval.
        obs (Tensor)    : Observed waveform.
        syn (Tensor)    : Synthetic waveform.
    '''
    def __init__(self, dt=1, beta=10) -> None:
        super().__init__()
        self.dt = dt
        self.beta = beta
    
    def forward(self, obs: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
        srcn, nt, rcvn = obs.shape
        padding = nt - 1
        device = obs.device

        # normalization
        obs = obs / torch.max(torch.abs(obs), dim=1, keepdim=True)[0]  # Normalize observed data
        syn = syn / torch.max(torch.abs(syn), dim=1, keepdim=True)[0]  # Normalize synthetic data
        
        rsd = torch.zeros((srcn, rcvn), device=device)  # Reset residual tensor
        for ishot in range(srcn):
            for ircv in range(rcvn):
                # cross_corr = cross_correlation(obs[ishot, :, ircv], syn[ishot, :, ircv], padding=padding)
                # tt_loss = trvaletime_difference(cross_corr=cross_corr, beta=self.beta)*self.dt
                tt_loss = calculate_time_shift(obs[ishot,:,ircv],syn[ishot,:,ircv],beta=self.beta)*self.dt
                rsd[ishot, ircv] = torch.abs(tt_loss)
        
        loss = torch.sum(rsd)  # Compute the total loss
        return loss

# def cross_correlation(wave1, wave2, padding):
#     """Calculate the cross-correlation between two waveforms."""
#     cross_corr = F.conv1d(
#         wave1.view(1, 1, -1),
#         wave2.view(1, 1, -1),
#         padding=padding
#     ).squeeze().abs()
#     center_idx = padding
#     cross_corr = cross_corr[center_idx : center_idx + wave1.size(-1)]
#     return cross_corr

# def trvaletime_difference(cross_corr, beta=100):
#     """Calculate the travel time difference based on cross-correlation."""
#     *_, n = cross_corr.shape
#     input = F.softmax(beta * cross_corr, dim=-1)            # Fix variable reference
#     indices = torch.linspace(0, 1, n, device=input.device)  # Move to the right device
#     tt_loss = torch.sum(n * input * indices, dim=-1)
#     return tt_loss 

# class Misfit_traveltime(torch.autograd.Function):
#     """Implementation of the cross-correlation misfit function
#         s = 0.5*\delta \tau**2
#         where \delta \tau is the time shift between synthetic and observed data
        
#         Luo, Y., & Schuster, G. T. (1991). Wave-equation traveltime inversion.
#             Geophysics, 56(5), 645-653.
#     """
#     @staticmethod
#     def forward(ctx,syn:torch.Tensor,obs:torch.Tensor):
#         """forward pass of the cross-correlation misfit function"""
#         nsrc,nt,nrec    = syn.shape
#         padding         = nt -1 
#         max_time_lags   = torch.zeros((nsrc,nrec),device=syn.device,dtype=syn.dtype)
        
#         # compute the time shift by cross-correlation
#         for isrc in range(nsrc):
#             for irec in range(nrec):
#                 # avoid zero traces
#                 if syn[isrc,:,irec].sum() == 0 or obs[isrc,:,irec].sum() == 0:
#                     continue
#                 cross_corr = F.conv1d(
#                     syn[isrc,:,irec].view(1,1,-1),
#                     obs[isrc,:,irec].view(1,1,-1)
#                 ).squeeze().abs()
#                 max_time_lags[isrc,irec] = torch.argmax(cross_corr) - padding
#         loss = 0.5*torch.sum(max_time_lags**2)
#         # save necessary values for backward pass
#         ctx.save_for_backward(syn,max_time_lags)
#         return loss
    
#     @staticmethod
#     def backward(ctx,grad_output):
#         """backward pass of the cross-correlation travetime misfit function
#             grad_output: the gradient of last layer
            
#             ds/dv = d(delta tau)/dv * delta tau
#                   = 1/(dp_syn/dt) * dp_syn/dv * delta tau
#         """
#         syn,max_time_lags = ctx.saved_tensors
#         # compute the gradient
#         adj = torch.zeros_like(syn)
#         adj[:,1:-1,:] = (syn[:,2:,:] - syn[:,0:-2,:])/2.0
#         adj_sum_squared = torch.sum(adj**2,dim=1,keepdim=True)
        
#         # avoid division by zero
#         adj_sum_squared[adj_sum_squared == 0] = 1.0
        
#         adj = adj/adj_sum_squared
#         adj = adj * (max_time_lags).unsqueeze(1)*grad_output
        
#         # check for NaNs
#         if torch.isnan(adj).any():
#             raise ValueError("NaNs in the gradient")
#         return adj,None