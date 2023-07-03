'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-06-27 19:24:35
* LastEditors: LiuFeng
* LastEditTime: 2023-07-03 11:22:47
* FilePath: /TorchInversion/TorchInversion/propagators.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import torch
import torch.nn as nn
from TorchInversion.utils import numpy2tensor
from TorchInversion.kernels import acoustic_FM2_kernel


class Model():
    def __init__(self,param,model,src,rcv,obs_data=[],device="cpu"):
        pass
    


class Acoustic_Simulation(nn.Module):
    def __init__(self,param,model,src,rcv,obs_data=[],device="cpu"):
        super(Acoustic_Simulation,self).__init__()
        # modeling parameter
        self.nx = param.nx;         self.ny = param.ny
        self.dx = param.dx;         self.dy = param.dy
        self.nt = param.nt;         self.dt = param.dt
        self.pml = param.pml;       self.fs = param.fs
        self.nx_pml = param.nx_pml; self.ny_pml = param.ny_pml
        
        # velocity model
        self.init_v = model.v
        self.init_rho = model.rho
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
        
        # inversion or forward
        self.obs_data = obs_data # 【shot,time,observedata】

        # device 
        self.device = device
        
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

    def inversion(self,v,rho):
        csg,forw = acoustic_FM2_kernel(
                self.nx,self.ny,self.dx,self.dy,
                self.nt,self.dt,self.pml,self.fs,
                self.nx_pml,self.ny_pml,self.damp_global,
                self.src_x,self.src_y,self.src_n,self.stf_val,
                self.rcv_x,self.rcv_y,self.rcv_n,
            v,rho,device=self.device)
        # normalize
        csg = csg/(torch.max(torch.abs(csg),axis=1,keepdim=True).values)
        # the observed data
        csg_obs = numpy2tensor(self.obs_data).to(self.device)
        # the L2 misfit
        loss = torch.sum(torch.sqrt(torch.sum((csg-csg_obs)*(csg-csg_obs)*self.dt,axis=1)))
        loss.backward()
        if self.device == 'cpu':
            forw = forw.detach()
            csg = csg.detach()
            grads_ = v.grad.detach()
            loss_ = loss.detach()
        else:
            forw = forw.cpu().detach()
            csg = csg.cpu().detach()
            grads_ = v.grad.cpu().detach()
            loss_ = loss.cpu().detach()
        return loss_,grads_,csg,forw
    
    def forward(self,v,rho):
        csg,_ = acoustic_FM2_kernel(
            self.nx,self.ny,self.dx,self.dy,
            self.nt,self.dt,self.pml,self.fs,
            self.nx_pml,self.ny_pml,self.damp_global,
            self.src_x,self.src_y,self.src_n,self.stf_val,
            self.rcv_x,self.rcv_y,self.rcv_n,
            v,rho,device=self.device)
        # normalize
        csg = csg/(torch.max(torch.abs(csg),axis=1,keepdim=True).values)
        # inversion return the loss and grads
        return csg