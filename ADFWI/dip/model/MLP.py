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
import math

class MLP(torch.nn.Module):
    def __init__(self,model_shape,
                 random_state_num = 100,
                 hidden_layer_number = [100,100],
                 vmin=None,vmax=None,
                 unit = 1000,
                 device="cpu"
                 ):
        super(MLP,self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.device = device
        self.model_shape = model_shape
        self.in_features = random_state_num
        self.unit = unit
        
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
            nn.Linear(in_features=hidden_layer_number[-1],out_features=model_shape[0]*model_shape[1])
        )
        
        # latent variable
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(self.in_features).to(self.device)
        
    def forward(self):
        # neural network generation
        out = self.MLP_in(self.random_latent_vector)
        for i in range(len(self.MLP_Blocks)):
            out = self.MLP_Blocks[i](out)
        out = self.MLP_out(out).view(self.model_shape[0],self.model_shape[1])
        # post process
        out = torch.squeeze(out)
        if self.vmin != None and self.vmax != None:
            out = ((self.vmax-self.vmin)*torch.tanh(out) + (self.vmax+self.vmin))/2
        out = torch.squeeze(out)*self.unit
        return out