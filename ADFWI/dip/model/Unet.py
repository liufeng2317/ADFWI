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