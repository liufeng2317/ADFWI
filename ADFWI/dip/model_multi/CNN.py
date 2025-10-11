'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2025-10-11 09:08:12
* LastEditors: LiuFeng
* LastEditTime: 2024-05-26 10:41:48
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

############################################################################
#                   CNN model implement by Weiqiang Zhu
# Zhu, W., Xu, K., Darve, E. & Beroza, G.C., 2021. 
#   A general approach to seismic inversion with automatic differentiation. 
#   Computers & Geosciences, 151, 104751. 
#   doi:10.1016/j.cageo.2021.104751
# Zhu, W., Xu, K., Darve, E., Biondi, B. & Beroza, G.C., 2022. 
#   Integrating deep neural networks with full-waveform inversion: Reparameterization, regularization, and uncertainty quantification. 
#   GEOPHYSICS, 87, R93–R109. 
#   doi:10.1190/geo2020-0933.1  
############################################################################

# model
class CNNs(torch.nn.Module):
    def __init__(self,model_shape,
                 random_state_num    = 100,
                 in_channels         = [32,32],
                 out_channels_number = 1,
                 vmins               = [None], 
                 vmaxs               = [None],
                 units               = [1000],
                 dropout_prob        = 0,
                 device="cpu"):
        """
            model_shape (tuple) : the shape of velocity model
            random_state_num (int) : the number of random state
            in_channels (list)  : the input and output channels for each CNN block
            out_channels_number (int) : the number of output channels
            vmins (list)        : the minimum velocity of output
            vmaxs (list)        : the maximum velocity of output
            units (list)       : the unit of the model parameters
            dropout_prob (float): probability of dropout
            device (optional)   : cpu or cuda
        """
        super(CNNs,self).__init__()
        self.device = device
        self.vmins = vmins
        self.vmaxs = vmaxs
        self.units = units
        self.out_channels_number = out_channels_number
        
        # model setting
        self.layer_num = layer_num = len(in_channels)-1
        h_in        = math.ceil(model_shape[0]/(2**layer_num))
        w_in        = math.ceil(model_shape[1]/(2**layer_num))
        self.h_v0   = model_shape[0]
        self.w_v0   = model_shape[1]
        
        # neural network blocks
        self.in_features = random_state_num
        
        self.FNN_in = nn.Sequential(
            nn.Linear(in_features=self.in_features,out_features=h_in*w_in*in_channels[0],bias=False),
            nn.Unflatten(0,(-1,in_channels[0],h_in,w_in)),
            nn.LeakyReLU(0.1)
        )
        
        self.CNN_Blocks = nn.ModuleList()
        for i in range(layer_num):
            self.CNN_Blocks.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=(2,2)),
                    nn.Conv2d(in_channels = in_channels[i],out_channels=in_channels[i+1],kernel_size=4,stride=1,padding="same",bias=False),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(p=dropout_prob)  # add dropout layer
                )
            )
        # single layer for seperate output
        self.CNN_out = nn.Sequential(
            nn.Conv2d(in_channels = in_channels[-1],out_channels=self.out_channels_number,kernel_size=4,stride=1,padding="same",bias=False)
        )
        
        # latent variable
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(self.in_features).to(self.device)
    
    def forward(self):
        # neural network generation
        out = self.FNN_in(self.random_latent_vector)
        for i in range(self.layer_num):
            out = self.CNN_Blocks[i](out)
        out = self.CNN_out(out)
        
        # post process
        
        out = torch.squeeze(out,dim=0)
        
        out_res = []
        
        for i in range(self.out_channels_number):
            out_temp = out[i]
            vmin     = self.vmins[i]
            vmax     = self.vmaxs[i]
            unit     = self.units[i]
            if vmin != None and vmax != None:
                out_temp = ((vmax-vmin)*torch.tanh(out_temp) + (vmax+vmin))/2
            out_temp  = torch.squeeze(out_temp)*unit
            h_v,w_v = out_temp.shape
            h_v0,w_v0 = self.h_v0,self.w_v0
            out_temp = out_temp[(h_v-h_v0)//2:(h_v-h_v0)//2+h_v0,
                                (w_v-w_v0)//2:(w_v-w_v0)//2+w_v0]
            out_res.append(out_temp)
        
        if self.out_channels_number == 1:
            return out_res[0]
        else:
            return out_res