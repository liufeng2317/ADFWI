'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: LiuFeng
* LastEditTime: 2024-05-15 19:30:38
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

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

class Misfit_Wasserstein1(Misfit):
    def __init__(self,p=1,trans_type='linear',theta=1,dt=1) -> None:
        """
            trans_type: linear, abs, square, exp, softplus
        """
        super().__init__()
        self.p = 1
        self.trans_type = trans_type
        self.theta = theta
        self.dt = dt
        
    def forward(self,syn:torch.Tensor,obs:torch.Tensor):
        assert syn.shape == obs.shape
        
        # Flatten the input tensors for transformation (shape: [num_shots, num_time_steps * num_receivers])
        num_shots, num_time_steps, num_receivers = syn.shape
        
        # [num_shots, num_time_steps, num_receivers] --> [num_shots, num_receivers, num_time_steps]
        obs_transposed = obs.permute(1, 0, 2) 
        syn_transposed = syn.permute(1, 0, 2)
        
        # Reshape input tensors for transformation
        syn_flat = syn_transposed.reshape(num_time_steps, num_shots * num_receivers)
        obs_flat = obs_transposed.reshape(num_time_steps, num_shots * num_receivers)
        
        # Transform signals to ensure non-negativity
        mu, nu, d = transform(syn_flat, obs_flat, self.trans_type, self.theta)
        
        assert mu.min() > 0
        assert nu.min() > 0
        
        # Normalize each trace by its sum
        mu = trace_sum_normalize(mu, time_dim=0)
        nu = trace_sum_normalize(nu, time_dim=0)

        # Compute cumulative sums over the time dimension
        F = torch.cumsum(mu, dim=0)  # Keep the cumulative sum along the flattened time dimension
        G = torch.cumsum(nu, dim=0)  # Keep the cumulative sum along the flattened time dimension
        
        w1loss = (torch.abs(F-G) ** self.p).sum()*self.dt
        return w1loss