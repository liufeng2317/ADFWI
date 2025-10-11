'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2025-10-11 09:08:12
* LastEditors: LiuFeng
* LastEditTime: 2025-10-11 10:41:48
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLPs(torch.nn.Module):
    def __init__(self,model_shape,
                 random_state_num    = 100,
                 hidden_layer_number = [100,100],
                 out_channels_number = 1,
                 vmins               = [None], 
                 vmaxs               = [None],
                 units               = [1000],
                 device="cpu"):
        """
            model_shape (tuple) : the shape of velocity model
            hidden_layer_number (list) : the number of hidden layers
            out_channels_number (int) : the number of output channels
            vmin (float)        : the minimum velocity of output
            vmax (float)        : the maximum velocity of output
            units (float)       : the unit of the model parameters
            device (optional)   : cpu or cuda
        """
        super(MLPs,self).__init__()
        self.device = device
        self.vmins = vmins
        self.vmaxs = vmaxs
        self.units = units
        self.out_channels_number = out_channels_number
        self.model_shape = model_shape
        self.in_features = random_state_num
        
        self.MLP_in = nn.Sequential(
            nn.Linear(in_features=self.in_features,out_features=hidden_layer_number[0],bias=False),
            nn.LeakyReLU(0.1)
        )
        self.MLP_Blocks = nn.ModuleList()
        for i in range(len(hidden_layer_number)-1):
            self.MLP_Blocks.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layer_number[i],out_features=hidden_layer_number[i+1]),
                    nn.LeakyReLU(0.1)
                )
            )
            
        self.MLP_out = nn.Sequential(
            nn.Linear(in_features=hidden_layer_number[-1],out_features=2*model_shape[0]*model_shape[1])
        )
        
        # latent variable
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(self.in_features).to(self.device)
        
    def forward(self):
        # neural network generation
        out = self.MLP_in(self.random_latent_vector)
        for i in range(len(self.MLP_Blocks)):
            out = self.MLP_Blocks[i](out)
        out = self.MLP_out(out).view(2,self.model_shape[0],self.model_shape[1])
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
            out_res.append(out_temp)
            
        if self.out_channels_number == 1:
            return out_res[0]
        else:
            return out_res