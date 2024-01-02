'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-06-27 19:24:35
* LastEditors: LiuFeng
* LastEditTime: 2024-01-01 16:05:03
* FilePath: /ADFWI/TorchInversion/propagators.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import torch
import torch.nn as nn
from TorchInversion.utils import numpy2tensor
from TorchInversion.kernels import *

    
############################################################################
#                       Update using Torch optimizer
############################################################################
class Acoustic_Simulation(nn.Module):
    def __init__(self,param,model,src,rcv,v,rho,obs_data=[],device="cpu"):
        super(Acoustic_Simulation,self).__init__()
        # modeling parameter
        self.nx = param.nx;         self.ny = param.ny
        self.dx = param.dx;         self.dy = param.dy
        self.nt = param.nt;         self.dt = param.dt
        self.pml = param.pml;       self.fs = param.fs
        self.nx_pml = param.nx_pml; self.ny_pml = param.ny_pml
        self.vmin = param.vmin;     self.vmax = param.vmax
        # velocity model
        self.damp_global = model.damp_global
        
        # source 
        self.src_x = src.src_x
        self.src_y = src.src_y
        self.src_n = src.src_n
        self.stf_val = src.stf_val
        self.stf_t = src.stf_t
        
        # reveiver
        self.rcv_x = rcv.rcv_x
        self.rcv_y = rcv.rcv_y
        self.rcv_n = rcv.rcv_n

        # device 
        self.device = device
        
        # initial model
        self.rho = rho
        self.v = torch.nn.Parameter(v)
        
        self.obs_data = obs_data
        self.to_tensor()
    
    def to_tensor(self):
        # velocity model
        self.damp_global = numpy2tensor(self.damp_global).to(self.device)
        # reveiver
        self.rcv_x = numpy2tensor(self.rcv_x).to(self.device)
        self.rcv_y = numpy2tensor(self.rcv_y).to(self.device)
        self.stf_val = numpy2tensor(self.stf_val).to(self.device)
        self.src_x = numpy2tensor(self.src_x).to(self.device)
        self.src_y = numpy2tensor(self.src_y).to(self.device)
        # init v and rho
        self.v = numpy2tensor(self.v).to(self.device)
        self.rho = numpy2tensor(self.rho).to(self.device)
        # obs data
        self.obs_data = numpy2tensor(self.obs_data).to(self.device)
    
    def forward(self):
        if self.device == "cpu":
            self.rho = torch.pow(self.v.detach(),0.25)*310
        else:
            self.rho = torch.pow(self.v.cpu().detach(),0.25)*310
        self.rho = self.rho.to(self.device)
        # forward modeling
        csg,forw = acoustic_FM2_kernel(
            self.nx,self.ny,self.dx,self.dy,
            self.nt,self.dt,self.pml,self.fs,
            self.nx_pml,self.ny_pml,self.damp_global,
            self.src_x,self.src_y,self.src_n,self.stf_val,
            self.rcv_x,self.rcv_y,self.rcv_n,
            self.v,self.rho,device=self.device)
        return csg,forw
    