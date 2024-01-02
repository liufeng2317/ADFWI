'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-06-27 19:24:35
* LastEditors: LiuFeng
* LastEditTime: 2023-12-18 13:37:54
* FilePath: /Acoustic_AD/TorchInversion/propagators_NN.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import torch
import torch.nn as nn
import math
from TorchInversion.utils import numpy2tensor
from TorchInversion.kernels import *
from TorchInversion.gradient_precond import smooth2d
from TorchInversion.NN.CNN import CNN
from TorchInversion.NN.Unet import UNet
    
############################################################################
#                           Torch NN
############################################################################

class Acoustic_Simulation_NN(nn.Module):
    def __init__(self,param,vel_model,src,rcv,obs_data=[],device="cpu",NN_model="CNN",generator="dv"):
        super(Acoustic_Simulation_NN,self).__init__()
        # modeling parameter
        self.nx = param.nx;             self.ny = param.ny
        self.dx = param.dx;             self.dy = param.dy
        self.nt = param.nt;             self.dt = param.dt
        self.pml = param.pml;           self.fs = param.fs
        self.nx_pml = param.nx_pml;     self.ny_pml = param.ny_pml
        self.vmin = param.vmin;         self.vmax = param.vmax
        
        # velocity model
        self.init_v         =   vel_model.v
        self.init_rho       =   vel_model.rho
        self.damp_global    =   vel_model.damp_global
        
        # source 
        self.src_x      =   src.src_x
        self.src_y      =   src.src_y
        self.src_n      =   src.src_n
        self.stf_val    =   src.stf_val
        self.stf_t      =   src.stf_t
        
        # reveiver
        self.rcv_x = rcv.rcv_x
        self.rcv_y = rcv.rcv_y
        self.rcv_n = rcv.rcv_n

        # device 
        self.device = device
        
        # observed data
        self.obs_data = obs_data
        self.to_tensor()
        
        # Neural Network
        self.dvmin = -0.5
        self.dvmax = 0.5
        # define the network generate v or dv
        self.generator = generator
        self.NN_model = NN_model
        
        self.model = self.network_select()
        

    def network_select(self):
        if self.generator.lower() == "v":
            if self.NN_model == "CNN":
                self.model =  CNN(self.init_v,vmin=self.vmin/1e3,vmax=self.vmax/1e3,device=self.device)
            elif self.NN_model == "Unet":
                self.model = UNet(self.init_v,vmin=self.vmin/1e3,vmax=self.vmax/1e3,device=self.device)
        elif self.generator.lower() == "dv":
            if self.NN_model == "CNN":
                self.model = CNN(self.init_v,vmin=self.dvmin,vmax=self.dvmax,device=self.device)
            elif self.NN_model == "Unet":
                self.model = UNet(self.init_v,vmin=self.dvmin,vmax=self.dvmax,device=self.device)
        
        if self.device != "cpu":
            self.model = self.model.cuda()
        
        return self.model
    
    
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
        self.init_v = numpy2tensor(self.init_v).to(self.device)
        self.init_rho = numpy2tensor(self.init_rho).to(self.device)
        # obs data
        self.obs_data = numpy2tensor(self.obs_data).to(self.device)
    
    def get_v(self,v):
        h_v = v.shape[0]
        w_v = v.shape[1]
        h_v0 = self.init_v.shape[0]
        w_v0 = self.init_v.shape[1]
        v_new = v[(h_v-h_v0)//2:(h_v-h_v0)//2+h_v0,
                    (w_v-w_v0)//2:(w_v-w_v0)//2+w_v0]
        return v_new
    
        
    def preTrainModel(self,pretrain_param):
        lr = pretrain_param.lr
        iteration = pretrain_param.iteration
        step_size = pretrain_param.step_size
        gamma = pretrain_param.gamma
        optimizer = torch.optim.Adam(self.model.parameters(),lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
        
        # iterative inversion
        pbar = tqdm(range(iteration+1))
        for i in pbar:  
            v = self.model()
            v = self.get_v(v)
            loss = torch.sqrt(torch.sum((v - self.init_v)**2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
        
        
    def forward(self):
        # velocity generation 
        if self.generator == "dv":
            dv = self.model()
            dv = self.get_v(dv)
            # mask the top
            dv[:12] = 0
            v = self.init_v + dv
            v = torch.clip_(v,min=self.vmin,max=self.vmax)
        elif self.generator == 'v':
            v = self.model()
            v = self.get_v(v)
            v[:12] = self.init_v[:12]
            v = torch.clip_(v,min=self.vmin,max=self.vmax)
        
        # generate the density
        if self.device == "cpu":
            rho = torch.pow(v.detach(),0.25)*310
        else:
            rho = torch.pow(v.cpu().detach(),0.25)*310
        rho = rho.to(self.device)
        
        # forward modeling
        csg,forw = acoustic_FM2_kernel(
            self.nx,self.ny,self.dx,self.dy,
            self.nt,self.dt,self.pml,self.fs,
            self.nx_pml,self.ny_pml,self.damp_global,
            self.src_x,self.src_y,self.src_n,self.stf_val,
            self.rcv_x,self.rcv_y,self.rcv_n,
            v,rho,device=self.device)
        
        # normalize
        # csg = csg/(torch.max(torch.abs(csg),axis=1,keepdim=True).values)
        
        
        ##################################################################
        # preprocess the gradient
        # self.forw = forw.cpu().detach().numpy()
        # def grad_precond_inner(grad):
        #     #apply the inverse Hessian
        #     if min(self.nx, self.ny) > 40:      # set 40 grids in default
        #         span = 40
        #     else:                     # in case the grid number is less than 40
        #         span = int(min(self.nx, self.ny)/2)
        #     forw = self.forw
        #     forw = smooth2d(forw, span)
        #     epsilon = 0.0001
        #     forw = forw / np.max(forw)
        #     precond = forw
        #     precond = precond / np.max(precond)
        #     precond[precond < epsilon] = epsilon
        #     precond = numpy2tensor(precond).to(self.device)
        #     grad = grad / (precond*precond)
        #     grad = self.vmax * grad/torch.abs(grad).max()
        #     grad[:12] = 0
        #     return grad
        # v.register_hook(grad_precond_inner)
        ####################################################################
        
        # return the observed waveform and forward illumination
        return csg,forw