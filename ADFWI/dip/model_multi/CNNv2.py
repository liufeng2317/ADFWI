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
#    CNNs Encoder + multi-CNN Decoder for different parameters
############################################################################
# model
class CNNs(torch.nn.Module):
    def __init__(self,model_shape,
                 random_state_num    = 100,
                 backbone_channels   = [32,32],
                 branches_channels   = [1],
                 branches_number     = 1,
                 vmins               = [None], 
                 vmaxs               = [None],
                 units               = [1000],
                 dropout_prob        = 0,
                 device="cpu"):
        """
            model_shape (tuple) : the shape of velocity model
            backbone_channels (list)  : the channels of backbone
            branches_channels (list) : the channels of branches
            branches_number (int) : the number of branches
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
        self.branches_number = branches_number
        
        # model setting
        self.layer_num = layer_num = len(backbone_channels)-1
        h_in        = math.ceil(model_shape[0]/(2**layer_num))
        w_in        = math.ceil(model_shape[1]/(2**layer_num))
        self.h_v0   = model_shape[0]
        self.w_v0   = model_shape[1]
        
        # root part: feature extraction
        self.in_features = random_state_num
        
        self.FNN_in = nn.Sequential(
            nn.Linear(in_features=self.in_features,out_features=h_in*w_in*backbone_channels[0],bias=False),
            nn.Unflatten(0,(-1,backbone_channels[0],h_in,w_in)),
            nn.LeakyReLU(0.1)
        )
        
        self.CNN_Blocks = nn.ModuleList()
        for i in range(layer_num):
            self.CNN_Blocks.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=(2,2)),
                    nn.Conv2d(in_channels = backbone_channels[i],out_channels=backbone_channels[i+1],kernel_size=4,stride=1,padding="same",bias=False),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(p=dropout_prob)  # add dropout layer
                )
            )

        # Define separate branches for each output channel
        self.branches = nn.ModuleList()
        for _ in range(branches_number):
            layers = []
            in_ch = backbone_channels[-1]
            for idx, out_ch in enumerate(branches_channels):
                layers.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=1, padding="same", bias=False))
                if idx < len(branches_channels) - 1:
                    layers.append(nn.LeakyReLU(0.1))
                    layers.append(nn.Dropout(p=dropout_prob))
                in_ch = out_ch
            self.branches.append(nn.Sequential(*layers))

        # latent variable
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(self.in_features).to(self.device)
    
    def forward(self):
        # neural network generation
        out = self.FNN_in(self.random_latent_vector)
        for i in range(self.layer_num):
            out = self.CNN_Blocks[i](out)
        # out = self.CNN_out(out)
        
        # Apply separate branches
        out = [branch(out) for branch in self.branches]
        out = torch.cat(out, dim=0)  # Concatenate back to original shape
        
        # post process
        out = torch.squeeze(out,dim=0)
        
        out_res = []
        
        for i in range(self.branches_number):
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
        
        if self.branches_number == 1:
            return out_res[0]
        else:
            return out_res