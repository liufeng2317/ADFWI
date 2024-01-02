'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-12-05 14:54:39
* LastEditors: LiuFeng
* LastEditTime: 2023-12-20 15:02:53
* FilePath: /Acoustic_AD/TorchInversion/NN/RTM.py
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
        self.init_v = init_v
        
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
        self.CNN1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels=1,kernel_size=4,stride=1,padding="same",bias=False),
        )
        
        # latent variable
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(self.in_features).to(self.device)
        
        # CNN        
        self.CNN2 = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels=16,kernel_size=4,stride=1,padding="same",bias=False),
            nn.Tanh()
        )
        self.CNN3 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels=1,kernel_size=1,stride=1,padding="same",bias=False),
            nn.Tanh()
        )
        # self.CNN4 = nn.Sequential(
        #     nn.Conv2d(in_channels = 4,out_channels=1,kernel_size=4,stride=1,padding="same",bias=False),
        #     # nn.Tanh()
        # )
        # self.i = 0
        
    
    def padding_value(self,v,RTM_img):
        RTM_img = torch.nn.functional.pad(RTM_img,
                                          [(v.shape[3]-RTM_img.shape[2])//2,(v.shape[3]-RTM_img.shape[2])//2,
                                           (v.shape[2]-RTM_img.shape[1])//2,(v.shape[2]-RTM_img.shape[1])//2],
                                          'replicate')
        return RTM_img
        
    def forward(self,RTM_img=[],pretrain=True):
        # generate the velocity
        v = self.FNN1(self.random_latent_vector)
        v = self.CNN_Block1(v)
        v = self.CNN_Block2(v)
        v = self.CNN1(v)
        
        if pretrain:
            if self.vmin != None and self.vmax != None:
                v = ((self.vmax-self.vmin)*torch.tanh(v) + (self.vmax+self.vmin))/2
            v = torch.squeeze(v)*1000
            return v
        else:
            # cancate the velocity and RTM image
            RTM_img = torch.unsqueeze(RTM_img,dim=0)
            RTM_img = self.padding_value(v,RTM_img)
            RTM_img = self.CNN2(RTM_img)
            RTM_img = self.CNN3(RTM_img)
            # RTM_img = torch.squeeze(RTM_img)
            # v = torch.squeeze(v)
            # v = torch.stack((v,RTM_img))
            v = v + RTM_img
            # v = torch.unsqueeze(v,dim=0)
            # v = self.CNN2(v)
            # v = self.CNN3(v)
            # v = self.CNN4(v)
            if self.vmin != None and self.vmax != None:
                v = ((self.vmax-self.vmin)*torch.tanh(v) + (self.vmax+self.vmin))/2
            v = torch.squeeze(v)*1000
            return v
