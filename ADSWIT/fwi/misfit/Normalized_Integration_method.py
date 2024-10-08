import numpy as np
import torch
from .base import Misfit
from typing import Optional

"""
    Wasserstein distance (W1 Loss)
        https://github.com/YangFangShu/FWIGAN/blob/main/FWIGAN/utils/misfit.py
"""

def transform(f, g, trans_type, theta):
    """
        do the transform for the signal f and g
        Args:
            f, g: seismic data with the shape [num_time_steps,num_shots*num_receivers_per_shot]
            trans_type: type for transform
            theta:the scalar variable for transform            
        return:
            output with transfomation
    """ 
    c = 0.0 
    device = f.device
    if trans_type == 'linear':
        min_value = torch.min(f.detach().min(), g.detach().min())
        mu, nu = f, g
        c = -min_value if min_value<0 else 0
        c = c * theta
        d = torch.ones(f.shape).to(device)
    elif trans_type == 'abs':
        mu, nu = torch.abs(f), torch.abs(g)
        d = torch.sign(f).to(device)
    elif trans_type == 'square':
        mu = f * f
        nu = g * g
        d = 2 * f
    elif trans_type == 'exp':
        mu = torch.exp(theta*f)
        nu = torch.exp(theta*g)
        d = theta * mu
    elif trans_type == 'softplus':
        mu = torch.log(torch.exp(theta*f) + 1)
        nu = torch.log(torch.exp(theta*g) + 1)
        d = theta / torch.exp(-theta*f)
    else:
        mu, nu = f, g
        d = torch.ones(f.shape).to(device)
    mu = mu + c + 1e-18 
    nu = nu + c + 1e-18
    return mu, nu, d

def trace_sum_normalize(x,time_dim=1):
    """
        normalization with the summation of each trace
        note that the channel should be 1
    """
    x = x / (x.sum(dim=time_dim,keepdim=True)+1e-18)
    return x

def trace_max_normalize(x):
    """
        normalization with the maximum value of each trace (the value of each trace is in [-1,1] after the processing)
        note that the channel should be 1
    """
    x_max,_ = torch.max(x.abs(),dim=0,keepdim=True)
    x = x / (x_max+1e-18)
    return x

# The following gradient are not assistant with Automatic Differentiation
# class Misfit_NIM(torch.autograd.Function):
#     """
#         Normalized Integration Method, Liu et al., 2012: the objective function measures the misfit between the integral of the absolute value, or of the square, or of the envelope of the signal.
#             F_i = \frac{\sum_{j=1}^i P(f_j)}{\sum_{j=1}^n P(f_j)}, 
#             G_i = \frac{\sum_{j=1}^i P(g_j)}{\sum_{j=1}^n P(g_j)}, 
#             \ell(f, g) = \sum_{i=1}^n |F_i - G_i|^p, 
#         where function :`P` is choosed to make the vector nonnegative, 
#             e.g. :`|x|`, `|x|^2`.
#         parameters
#         -----------
#             p (int)         : the norm degree. Default: 2 
#             trans_type (str): the nonnegative transform. Default: 'linear'
#             theta (int)     : the parameter used in nonnegtive transform. Default: 1
#         Note:
#             NIM is equivalent to Wasserstein-1 distance (Earth Mover's distance) when p = 1
#     """
#     def __init__(self,p=2,trans_type='linear',theta=1.):
#         self.p = p
#         self.trans_type = trans_type
#         self.theta = theta
    
#     @staticmethod
#     def forward(ctx, syn:torch.Tensor, obs:torch.Tensor, p=2, trans_type='linear', theta=1.):
#         assert p >= 1
#         assert syn.shape == obs.shape
#         device = syn.device
        
#         # transform for the signal syn and obs
#         p = torch.tensor(p).to(device)
#         mu, nu, d = transform(syn, obs, trans_type, theta)
        
#         # normalization with the summation of each trace
#         mu = trace_sum_normalize(mu,time_dim=1)
#         nu = trace_sum_normalize(nu,time_dim=1)
        
#         F = torch.cumsum(mu, dim=1)
#         G = torch.cumsum(nu, dim=1)
        
#         ctx.save_for_backward(F-G, mu, p, d)
        
#         return (torch.abs(F - G) ** p).sum()

#     @staticmethod
#     def backward(ctx, grad_output):
#         residual, mu, p, d = ctx.saved_tensors
#         if p == 1:
#             df = torch.sign(residual) * mu *d
#         else:
#             df = (residual) ** (p - 1) * mu * d
#         return df, None, None, None, None

class Misfit_NIM(Misfit):
    """
        Normalized Integration Method, Liu et al., 2012: the objective function measures the misfit between the integral of the absolute value, or of the square, or of the envelope of the signal.
            F_i = \frac{\sum_{j=1}^i P(f_j)}{\sum_{j=1}^n P(f_j)}, 
            G_i = \frac{\sum_{j=1}^i P(g_j)}{\sum_{j=1}^n P(g_j)}, 
            \ell(f, g) = \sum_{i=1}^n |F_i - G_i|^p, 
        where function :`P` is choosed to make the vector nonnegative, 
            e.g. :`|x|`, `|x|^2`.
        parameters
        -----------
            p (int)         : the norm degree. Default: 2 
            trans_type (str): the nonnegative transform. Default: 'linear'
            theta (int)     : the parameter used in nonnegtive transform. Default: 1
        Note:
            NIM is equivalent to Wasserstein-1 distance (Earth Mover's distance) when p = 1
    """
    def __init__(self,p=2,trans_type='linear',theta=1.,dt=1):
        self.p = p
        self.trans_type = trans_type
        self.theta = theta
        self.dt = dt
    def forward(self, syn:torch.Tensor, obs:torch.Tensor):
        assert self.p >= 1
        assert syn.shape == obs.shape
        
        # transform for the signal syn and obs
        mu, nu, d = transform(syn, obs, self.trans_type, self.theta)
        
        # normalization with the summation of each trace
        mu = trace_sum_normalize(mu,time_dim=1)
        nu = trace_sum_normalize(nu,time_dim=1)
        
        F   = torch.cumsum(mu, dim=1)
        G   = torch.cumsum(nu, dim=1)
        rsd = F-G
        result = (torch.abs(rsd*self.dt) ** self.p).sum()
        return result