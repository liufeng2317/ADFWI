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

def act(act_fun='LeakyReLU'):
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.1, inplace=True)
        elif act_fun == 'Swish':
            return nn.SiLU()  # PyTorch equivalent of Swish
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Identity()  # Use Identity instead of empty Sequential
        else:
            raise ValueError(f"Unsupported activation: {act_fun}")
    else:
        return act_fun()

def conv(in_f, out_f, kernel_size=3, stride=1, bias=False, pad='same', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            raise ValueError(f"Unsupported downsample mode: {downsample_mode}")
        stride = 1  # Reset stride after pooling

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=pad, bias=bias)

    layers = [convolver]
    if downsampler:
        layers.append(downsampler)
    return nn.Sequential(*layers)

def get_block(num_channel, norm_layer, act_fun):
    layers = [
        nn.Conv2d(num_channel, num_channel, 3, 1, 1, bias=False),
        norm_layer(num_channel, affine=True),
        act(act_fun),
        nn.Conv2d(num_channel, num_channel, 3, 1, 1, bias=False),
        norm_layer(num_channel, affine=True),
    ]
    return nn.Sequential(*layers)

class ResidualSequential(nn.Module):
    def __init__(self, *args):
        super(ResidualSequential, self).__init__()
        self.model = nn.Sequential(*args)

    def forward(self, x):
        out = self.model(x)
        # Ensure consistent dimensions or handle mismatch explicitly in ResidualSequential
        if out.size(2) != x.size(2) or out.size(3) != x.size(3):
            x = F.interpolate(x, size=out.size()[2:], mode='bilinear', align_corners=False)
        return out + x

class ResNet(nn.Module):
    def __init__(self, model_shape,
                 random_state_num=100, 
                 vmin=None, vmax=None, 
                 in_channels=1, out_channels=1, 
                 num_blocks=8, num_channel=32, 
                 need_residual=True, 
                 act_fun='LeakyReLU', 
                 norm_layer=nn.InstanceNorm2d, 
                 pad='same',
                 unit=1000,
                 device="cpu"):
        super(ResNet, self).__init__()
        self.vmin   = vmin
        self.vmax   = vmax
        self.unit   = unit
        self.device = device
        
        # neural network blocks
        self.FNN_in = nn.Sequential(
            nn.Linear(in_features=random_state_num, out_features=in_channels*model_shape[0] * model_shape[1], bias=False),
            nn.Unflatten(0,(-1, in_channels, model_shape[0], model_shape[1])),
            nn.LeakyReLU(0.1)
        )

        # residual blocks
        block_class = ResidualSequential if need_residual else nn.Sequential
        layers = [
            conv(in_channels, num_channel, 3, stride=1, bias=False, pad=pad),
            act(act_fun)
        ]
        
        for _ in range(num_blocks):
            layers.append(block_class(*get_block(num_channel, norm_layer, act_fun)))
        
        layers += [
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            norm_layer(num_channel, affine=True),
            conv(num_channel, out_channels, 3, 1, bias=False, pad=pad),
            act(act_fun)
        ]
        
        self.res_model = nn.Sequential(*layers)
        
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(random_state_num, device=self.device)

    def forward(self):
        # Pass through the main model
        out = self.FNN_in(self.random_latent_vector)
        out = self.res_model(out)
        out = torch.squeeze(out)
        if self.vmin is not None and self.vmax is not None:
            out = ((self.vmax - self.vmin) * torch.tanh(out) + (self.vmax + self.vmin)) / 2
        out = torch.squeeze(out)*self.unit
        return out

    def eval(self):
        self.model.eval()