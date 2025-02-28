'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-05-14 09:08:12
* LastEditors: LiuFeng
* LastEditTime: 2024-05-26 10:41:48
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from FyeldGenerator import generate_field
import numpy as np

# Helper that generates power-law power spectrum
def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)
    return Pk

def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b

def generate_grf(shape,alpha,unit_length=10,device='cpu'):
    field = generate_field(distrib, Pkgen(alpha), shape, unit_length=unit_length)
    return torch.tensor(field,dtype=torch.float32,device=device)

# Double convolution block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            # nn.LeakyReLU(0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(0.1)
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Downscaling block
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Upscaling block
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2] # CHW
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Output layer
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# General UNet class for any number of layers
class UNet(nn.Module):
    def __init__(self, model_shape, 
                 n_layers, 
                 base_channel, 
                 vmin=None, 
                 vmax=None, 
                 in_channels=1, out_channels=1, 
                 bilinear=False,
                 grf_initialize=False,
                 grf_alpha = 0, # power-law power spectrum
                 unit = 1000, 
                 device="cpu"
                 ):
        super(UNet, self).__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.model_shape    = model_shape
        self.bilinear       = bilinear
        self.unit           = unit
        
        self.grf_initialize = grf_initialize
        self.grf_alpha      = grf_alpha

        self.inc    = DoubleConv(in_channels, base_channel)
        self.downs  = nn.ModuleList()
        self.ups    = nn.ModuleList()

        # Create down blocks
        channels = base_channel
        for _ in range(n_layers):
            self.downs.append(Down(channels, channels * 2))
            channels *= 2

        # Create up blocks
        factor = 2 if bilinear else 1
        for _ in range(n_layers):
            self.ups.append(Up(channels, channels // 2 // factor, bilinear))
            channels //= 2

        self.outc = OutConv(channels, out_channels)

        # Random latent variable for input
        self.device = device
        self.vmin = vmin
        self.vmax = vmax
        self.h0, self.w0 = model_shape
        if grf_initialize:
            self.random_latent_vector = self._grf_initialize()
        else:
            self.random_latent_vector = self._random_initialize()
    
    def _random_initialize(self):
        torch.manual_seed(1234)
        return torch.rand(1, 1, self.h0, self.w0).to(self.device)
    
    def _grf_initialize(self):
        return generate_grf(self.model_shape,self.grf_alpha,device=self.device).unsqueeze(0).unsqueeze(0)

    def forward(self,x=None):
        if x is None:
            x = self.random_latent_vector
        x1 = self.inc(x)
        # down-sampling
        downs_outputs = [x1]
        for down in self.downs:
            downs_outputs.append(down(downs_outputs[-1]))
        # up-sampling
        x = downs_outputs[-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x, downs_outputs[-2-i])
        # output velocity model
        out = self.outc(x)
        
        # post process
        out = torch.squeeze(out)
        if self.vmin is not None and self.vmax is not None:
            out = ((self.vmax - self.vmin) * torch.tanh(out) + (self.vmax + self.vmin)) / 2
        out = torch.squeeze(out) * self.unit
        return out


# def approximate_kv(v, x):
#     """
#     Approximate the modified Bessel function of the second kind (Kv) for PyTorch.
#     Args:
#         v (torch.Tensor): Order of the Bessel function (v > 0).
#         x (torch.Tensor): Input value (x > 0).
#     Returns:
#         torch.Tensor: Approximated Kv(v, x).
#     """
#     # Avoid division by zero
#     x = torch.clamp(x, min=1e-10)

#     # Asymptotic expansion for large x
#     large_x = x > 10  # Threshold for large x
#     kv_large = torch.exp(-x) * torch.sqrt(torch.pi / (2 * x))

#     # Small x approximation
#     kv_small = 1 / x  # Small x: Kv(v, x) ~ 1 / x when v ~ 0

#     # Combine based on thresholds
#     kv_approx = torch.where(large_x, kv_large, kv_small)
#     return kv_approx


# def matern_covariance(d, alpha, sigma2, beta):
#     """
#     Matérn covariance function implemented with PyTorch.
#     Args:
#         d (torch.Tensor): Distance matrix (NxN or grid distance matrix).
#         alpha (float): Smoothness parameter.
#         sigma2 (float): Variance (amplitude) parameter.
#         beta (float): Scale parameter (controls correlation decay).
#     Returns:
#         torch.Tensor: Covariance matrix.
#     """
#     # Convert constants to tensors
#     alpha_tensor = torch.tensor(alpha, dtype=d.dtype, device=d.device)
#     beta_tensor = torch.tensor(beta, dtype=d.dtype, device=d.device)

#     # Compute sqrt_term
#     sqrt_term = torch.sqrt(torch.tensor(2 * alpha, dtype=d.dtype, device=d.device)) * d / beta_tensor
#     sqrt_term = torch.clamp(sqrt_term, min=1e-10)  # Avoid division by zero

#     # Compute kv using approximation
#     kv_term = approximate_kv(alpha_tensor, sqrt_term)

#     # Compute gamma(alpha) using PyTorch
#     gamma_alpha = torch.exp(torch.special.gammaln(alpha_tensor))

#     # Compute Matérn covariance
#     cov = sigma2 * (2 ** (1 - alpha_tensor)) / gamma_alpha * (sqrt_term ** alpha_tensor) * kv_term
#     cov[d == 0] = sigma2  # Handle the diagonal (d=0)

#     # Add jitter for numerical stability
#     cov += torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device) * 1e-6

#     # Validate positive-definiteness
#     eigenvalues = torch.linalg.eigvalsh(cov)
#     if torch.any(eigenvalues <= 0):
#         raise ValueError("Covariance matrix is not positive-definite. Adjust parameters or add more jitter.")
    
#     return cov

# def generate_grf(model_shape, alpha=1.5, sigma2=1.0, beta=10.0, device="cpu", seed=None):
#     """
#     Generate a 2D Gaussian Random Field (GRF) using Matérn covariance function.
#     Args:
#         model_shape (tuple): Size of the output grid (h, w).
#         alpha (float): Smoothness parameter of the Matérn covariance.
#         sigma2 (float): Variance (amplitude) parameter.
#         beta (float): Scale parameter (correlation length).
#         device (str): Device for computation ("cpu" or "cuda").
#         seed (int, optional): Random seed for reproducibility.
#     Returns:
#         torch.Tensor: 2D Gaussian Random Field of size (h, w).
#     """
#     if seed is not None:
#         torch.manual_seed(seed)

#     h, w = model_shape
#     y = torch.arange(0, h, device=device)
#     x = torch.arange(0, w, device=device)
#     yy, xx = torch.meshgrid(y, x, indexing='ij')

#     # Compute pairwise distance matrix (flattened grid)
#     grid_points = torch.stack([yy.flatten(), xx.flatten()], dim=1).float()
#     distances = torch.cdist(grid_points, grid_points, p=2)
#     distances = distances.reshape(h * w, h * w)

#     # Compute covariance matrix using Matérn function
#     cov_matrix = matern_covariance(distances, alpha, sigma2, beta)

#     # Perform Cholesky decomposition to generate correlated samples
#     L = torch.linalg.cholesky(cov_matrix + 1e-6 * torch.eye(cov_matrix.size(0), device=device))  # Add jitter for stability
#     z = torch.randn(h * w, 1, device=device)  # Standard normal samples
#     grf = (L @ z).reshape(h, w)  # Correlated samples reshaped to grid

#     return grf