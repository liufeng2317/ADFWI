'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-12-05 14:54:39
* LastEditors: LiuFeng
* LastEditTime: 2023-12-06 15:19:34
* FilePath: /Acoustic_AD/TorchInversion/NN/CNN.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

############################################################################
#                   CNN model implement by Weiqiang Zhu
############################################################################
# model
class CNN(torch.nn.Module):
    def __init__(self,init_v,vmin=None,vmax=None,device="cpu"):
        super(CNN,self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.device = device
        
        # model setting
        layer_num = 2
        h0 = math.ceil(init_v.shape[0]/(2**layer_num))
        w0 = math.ceil(init_v.shape[1]/(2**layer_num))
        
        # model
        self.in_features = 100
        self.FNN1 = nn.Sequential(
            nn.Linear(in_features=self.in_features,out_features=h0*w0*8,bias=False),
            nn.Unflatten(0,(-1,8,h0,w0)),
            nn.LeakyReLU(0.1)
            # nn.Tanh()
        )
        self.CNN_Block1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=(2,2)),
            nn.Conv2d(in_channels = 8,out_channels=32,kernel_size=4,stride=1,padding="same",bias=False),
            nn.LeakyReLU(0.1)
            # nn.Tanh()
        )
        self.CNN_Block2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=(2,2)),
            nn.Conv2d(in_channels = 32,out_channels=16,kernel_size=4,stride=1,padding="same",bias=False),
            nn.LeakyReLU(0.1)
            # nn.Tanh()
        )
        # self.CNN_Block3 = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor=(2,2)),
        #     nn.Conv2d(in_channels = 64,out_channels=32,kernel_size=(4,4),stride=(1,1),padding="same",bias=False),
        #     nn.LeakyReLU(0.1)
        #     # nn.Tanh()
        # )
        # self.CNN_Block4 = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor=(2,2)),
        #     nn.Conv2d(in_channels = 32,out_channels=16,kernel_size=(4,4),stride=(1,1),padding="same",bias=False),
        #     nn.LeakyReLU(0.1)
        #     # nn.Tanh()
        # )
        
        self.CNN1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels=1,kernel_size=4,stride=1,padding="same",bias=False)
        )
        # latent variable
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(self.in_features).to(self.device)
    
    def forward(self):
        # process
        out = self.FNN1(self.random_latent_vector)
        out = self.CNN_Block1(out)
        out = self.CNN_Block2(out)
        # dv = self.CNN_Block3(dv)
        # dv = self.CNN_Block4(dv)
        out = self.CNN1(out)
        out = torch.squeeze(out)
        if self.vmin != None and self.vmax != None:
            out = ((self.vmax-self.vmin)*torch.tanh(out) + (self.vmax+self.vmin))/2
        out = torch.squeeze(out)*1000
        return out
