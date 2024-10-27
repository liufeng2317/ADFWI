'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: LiuFeng
* LastEditTime: 2024-05-15 19:30:38
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''
import torch
from torch.autograd import Function

"""
    NIM misfit
        https://github.com/YangFangShu/FWIGAN/blob/main/FWIGAN/utils/misfit.py
"""

def transform(f, g, trans_type, theta):
    """
    Apply a transformation to make f and g non-negative.
    
    Args:
        f, g (Tensor): Seismic data, shape [num_time_steps, num_shots*num_receivers_per_shot].
        trans_type (str): Transformation type ('linear', 'abs', 'square', 'exp', 'softplus').
        theta (float): Scalar parameter for transformation.
        
    Returns:
        mu (Tensor): Transformed f.
        nu (Tensor): Transformed g.
        d (Tensor): Derivative of transformed f for potential use in gradient-based methods.
    """
    c = 0.0  # Initialize the offset constant
    device = f.device
    if trans_type == 'linear':
        # Linear transformation with offset
        min_value = torch.min(f.detach().min(), g.detach().min())
        mu, nu = f, g
        c = -min_value if min_value < 0 else 0
        c = c * theta  # Scale by theta for flexibility
        d = torch.ones(f.shape).to(device)
    elif trans_type == 'abs':
        # Absolute value transformation
        mu, nu = torch.abs(f), torch.abs(g)
        d = torch.sign(f).to(device)
    elif trans_type == 'square':
        # Squaring transformation
        mu = f * f
        nu = g * g
        d = 2 * f
    elif trans_type == 'exp':
        # Exponential transformation, scaled by theta
        mu = torch.exp(theta * f)
        nu = torch.exp(theta * g)
        d = theta * mu
    elif trans_type == 'softplus':
        # Softplus transformation for smooth non-negativity
        mu = torch.log(torch.exp(theta * f) + 1)
        nu = torch.log(torch.exp(theta * g) + 1)
        d = theta / (torch.exp(-theta * f) + 1e-18)  # Avoid division by zero
    else:
        mu, nu = f, g
        d = torch.ones(f.shape).to(device)
    # Ensure positive values for mu and nu by adding a small constant
    mu = mu + c + 1e-18
    nu = nu + c + 1e-18
    return mu, nu, d

def trace_sum_normalize(x, time_dim=0):
    """
    Normalize each trace by its sum along the specified dimension.
    
    Args:
        x (Tensor): Input tensor.
        time_dim (int): Dimension for time steps.
        
    Returns:
        Tensor: Normalized tensor.
    """
    x = x / (x.sum(dim=time_dim, keepdim=True) + 1e-18)  # Avoid division by zero
    return x

def trace_max_normalize(x, time_dim=0):
    """
    normalization with the maximum value of each trace (the value of each trace is in [-1,1] after the processing)
    note that the channel should be 1
    """
    x_max,_ = torch.max(x.abs(), dim=time_dim, keepdim=True)
    x = x / (x_max+1e-18)
    return x

class Misfit_NIM(Function):
    """
    Normalized Integration Method (NIM), computes misfit between cumulative distributions of transformed signals.
    
    Parameters:
        p (int): Norm degree, default is 2.
        trans_type (str): Type of non-negative transform, default is 'linear'.
        theta (float): Parameter for non-negative transform, default is 1.
        dt (float): Time sampling interval.
    
    Note:
        NIM is equivalent to the Wasserstein-1 distance when p=1.
    """
    def __init__(self, p=1,trans_type='linear',theta=1,dt=1):
        self.p = p
        self.trans_type = trans_type
        self.theta = theta
        self.dt = dt
    
    @staticmethod
    def forward(ctx, syn, obs, p=2, trans_type='linear', theta=1.):
        assert p >= 1, "Norm degree must be >= 1"
        assert syn.shape == obs.shape, "Shape mismatch between synthetic and observed data"
        
        # Flatten the input tensors for transformation (shape: [num_shots, num_time_steps * num_receivers])
        num_shots, num_time_steps, num_receivers = syn.shape
        
        # [num_shots, num_time_steps, num_receivers] --> [num_shots, num_receivers, num_time_steps]
        syn_transposed = syn.permute(1, 0, 2)
        obs_transposed = obs.permute(1, 0, 2) 

        # Reshape input tensors for transformation
        syn_flat = syn_transposed.reshape(num_time_steps, num_shots * num_receivers)
        obs_flat = obs_transposed.reshape(num_time_steps, num_shots * num_receivers)
        
        # Transform signals to ensure non-negativity
        mu, nu, d = transform(syn_flat, obs_flat, trans_type, theta)
        
        # Normalize each trace by its sum
        mu = trace_sum_normalize(mu, time_dim=0)
        nu = trace_sum_normalize(nu, time_dim=0)
        
        # Compute cumulative sums over the time dimension
        F = torch.cumsum(mu, dim=0)  # Keep the cumulative sum along the flattened time dimension
        G = torch.cumsum(nu, dim=0)  # Keep the cumulative sum along the flattened time dimension
        
        # Save the necessary tensors for backward computation
        ctx.save_for_backward(F - G, mu,  d)
        ctx.p = p
        ctx.num_shots = num_shots  # Save as an attribute
        ctx.num_time_steps = num_time_steps  # Save as an attribute
        ctx.num_receivers = num_receivers  # Save as an attribute
        
        return (torch.abs(F - G) ** p).sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        residual, mu, d = ctx.saved_tensors
        p = ctx.p
        num_shots = ctx.num_shots
        num_time_steps = ctx.num_time_steps
        num_receivers = ctx.num_receivers
        
        if p == 1:  # Check if p is 1
            df = torch.sign(residual) * mu * d
        else:
            df = (residual) ** (p - 1) * mu * d     
        
        df = df.reshape(num_time_steps, num_shots, num_receivers).permute(1, 0, 2)
        return -df, None, None, None, None




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
