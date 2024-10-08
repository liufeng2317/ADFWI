'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-04-26 19:42:24
* LastEditors: LiuFeng
* LastEditTime: 2024-06-01 23:01:28
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
from typing import Optional,Union
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from ADFWI.model       import AbstractModel,IsotropicElasticModel,AnisotropicElasticModel
from ADFWI.propagator  import ElasticPropagator,GradProcessor
from ADFWI.survey      import SeismicData
from ADFWI.fwi.misfit  import Misfit
from ADFWI.fwi.regularization import Regularization
from ADFWI.utils       import numpy2tensor
from ADFWI.view        import plot_vp_vs_rho,plot_model,plot_eps_delta_gamma

class ElasticFWI(torch.nn.Module):
    """Acoustic Full waveform inversion class
    """
    def __init__(self,propagator:ElasticPropagator,model:AbstractModel,
                 optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler,
                 loss_fn:Union[Misfit,torch.autograd.Function],
                 obs_data:SeismicData,gradient_processor:GradProcessor,
                 regularization_fn:Optional[Regularization] = None,
                 waveform_normalize:Optional[bool]          = True,
                 cache_result:Optional[bool]                = True,
                 cache_gradient:Optional[bool]              = True,
                 save_fig_epoch:Optional[int]               = -1,
                 save_fig_path:Optional[str]                = "",
                 inversion_component:Optional[np.array]     = ["pressure"],
                ):
        """
        Parameters:
        --------------
            propagator (Elastic Propagator)                 : the propagator for the isotropic elastic wave
            model (Model)                                   : the velocity model class
            optimizer (torch.optim.Optimizer)               : the pytorch optimizer
            scheduler (torch.optim.scheduler)               : the pytorch learning rate decay scheduler
            loss_fn   (Misfit or torch.autograd.Function)   : the misfit function
            obs_data  (SeismicData)                         : the observed dataset
            gradient_processor (GradProcessor)              : the gradient processor
            waveform_normalize (bool)   : normalize the waveform or not, default True
            cache_result (bool)         : save the temp result of the inversion or not
        """
        super().__init__()
        self.propagator         = propagator
        self.model              = model
        self.optimizer          = optimizer
        self.scheduler          = scheduler
        self.loss_fn            = loss_fn
        self.regularization_fn  = regularization_fn
        self.obs_data           = obs_data
        self.gradient_processor = gradient_processor
        self.device             = self.propagator.device
        self.dtype              = self.propagator.dtype 
        
        # observed data
        self.waveform_normalize = waveform_normalize
        obs_p   = -(self.obs_data.data["txx"]+self.obs_data.data["tzz"])
        obs_p   = numpy2tensor(obs_p,self.dtype).to(self.device)
        obs_vx  = numpy2tensor(self.obs_data.data["vx"],self.dtype).to(self.device)
        obs_vz  = numpy2tensor(self.obs_data.data["vz"],self.dtype).to(self.device)
        if self.waveform_normalize:
            obs_p  =  obs_p/(torch.max(torch.abs(obs_p),axis=1,keepdim=True).values)
            obs_vx = obs_vx/(torch.max(torch.abs(obs_vx),axis=1,keepdim=True).values)
            obs_vz = obs_vz/(torch.max(torch.abs(obs_vz),axis=1,keepdim=True).values)
        self.obs_p = obs_p
        self.obs_vx = obs_vx
        self.obs_vz = obs_vz
        # save result
        self.cache_result   = cache_result
        self.cache_gradient = cache_gradient
        self.iter_vp,self.iter_vs,self.iter_rho = [],[],[]       
        self.iter_eps,self.iter_delta,self.iter_gamma = [],[],[]
        self.iter_vp_grad,self.iter_vs_grad,self.iter_rho_grad = [],[],[]
        self.iter_eps_grad,self.iter_delta_grad,self.iter_gamma_grad = [],[],[]
        self.iter_loss      = []
        
        # save figure
        self.save_fig_epoch = save_fig_epoch
        self.save_fig_path  = save_fig_path
        
        # inversion component
        self.inversion_component = inversion_component
    
    def save_vp_vs_rho_fig(self,i,vp,vs,rho):
        vp_bound    =  self.model.get_bound("vp")
        vs_bound    =  self.model.get_bound("vs")
        rho_bound   =  self.model.get_bound("rho")
        if vp_bound[0] is None and vp_bound[1] is None:
            self.vp_min = self.model.get_model("vp").min() - 500
            self.vp_max = self.model.get_model("vp").max() + 500
        else: 
            self.vp_min = vp_bound[0]
            self.vp_max = vp_bound[1]
        
        if vs_bound[0] is None and vs_bound[1] is None:
            self.vs_min = self.model.get_model("vs").min() - 500
            self.vs_max = self.model.get_model("vs").max() + 500
        else: 
            self.vs_min = vs_bound[0]
            self.vs_max = vs_bound[1]
        
        if rho_bound[0] is None and rho_bound[1] is None:
            self.rho_min = self.model.get_model("rho").min() - 200
            self.rho_max = self.model.get_model("rho").max() + 200
        else: 
            self.rho_min = rho_bound[0]
            self.rho_max = rho_bound[1]
        
        
        if self.save_fig_epoch == -1:
            pass
        elif i%self.save_fig_epoch == 0:
            if os.path.exists(self.save_fig_path):
                plot_vp_vs_rho(
                    vp=vp,vs=vs,rho=rho,
                    # title=f"Iteration {i}",
                    figsize=(12,5),wspace=0.2,cbar_pad_fraction=0.18,cbar_height=0.04,
                    dx=self.model.dx,dz=self.model.dz,
                    save_path=os.path.join(self.save_fig_path,f"model_{i}.png"),
                    show=False
                    )
        return
    
    def save_eps_delta_gamma_fig(self,i,eps,delta,gamma,model_type="eps"):
        eps_bound    =  self.model.get_bound("eps")
        delta_bound    =  self.model.get_bound("delta")
        gamma_bound   =  self.model.get_bound("gamma")
        if eps_bound[0] is None and eps_bound[1] is None:
            self.vp_min = self.model.get_model("eps").min() - 0.01
            self.vp_max = self.model.get_model("eps").max() + 0.01
        else: 
            self.vp_min = eps_bound[0]
            self.vp_max = eps_bound[1]
        
        if delta_bound[0] is None and delta_bound[1] is None:
            self.delta_min = self.model.get_model("delta").min() - 0.01
            self.delta_max = self.model.get_model("delta").max() + 0.01
        else: 
            self.vs_min = delta_bound[0]
            self.vs_max = delta_bound[1]
        
        if gamma_bound[0] is None and gamma_bound[1] is None:
            self.gamma_min = self.model.get_model("gamma").min() - 0.01
            self.gamma_max = self.model.get_model("gamma").max() + 0.01
        else: 
            self.rho_min = gamma_bound[0]
            self.rho_max = gamma_bound[1]
    
        if self.save_fig_epoch == -1:
            pass
        elif i%self.save_fig_epoch == 0:
            if os.path.exists(self.save_fig_path):
                plot_eps_delta_gamma(
                    eps=eps,delta=delta,gamma=gamma,
                    # title=f"Iteration {i}",
                    figsize=(12,5),wspace=0.2,cbar_pad_fraction=0.18,cbar_height=0.04,
                    dx=self.model.dx,dz=self.model.dz,
                    save_path=os.path.join(self.save_fig_path,f"anisotropic_model_{i}.png"),
                    show=False
                    )
        return
    
    def save_gradient_fig(self,i,data,model_type="vp"):
        if self.save_fig_epoch == -1:
            pass
        elif i%self.save_fig_epoch == 0:
            if os.path.exists(self.save_fig_path):
                plot_model(data,title=f"Iteration {i}",
                        dx=self.model.dx,dz=self.model.dz,
                        save_path=os.path.join(self.save_fig_path,f"{model_type}_{i}.png"),show=False)
        return

    def forward(self,
                iteration:int,
                fd_order:int                        = 4,
                batch_size:Optional[int]            = None,
                checkpoint_segments:Optional[int]   = 1 ,
                start_iter                          = 0,
                ):
        """
        Parameters:
        ------------
            iteration (int)             : the iteration number of inversion
            fd_order (int)              : the order of finite difference
            batch_size (int)            : the shots for each batch, default -1 means use all the shots
            checkpoint_segments (int)   : seperate all the time seris into N segments for saving memory, default 1
        """
        n_shots = self.propagator.src_n
        if batch_size is None or batch_size > n_shots:
            batch_size = n_shots
        
        # epoch
        pbar_epoch = tqdm(range(start_iter,start_iter+iteration),position=0,leave=False,colour='green',ncols=80)
        for i in pbar_epoch:
            # batch
            self.optimizer.zero_grad()
            loss_batch = 0
            pbar_batch = tqdm(range(math.ceil(n_shots/batch_size)),position=1,leave=False,colour='red',ncols=80)
            for batch in pbar_batch:
                # forward simulation
                begin_index = 0  if batch==0 else batch*batch_size
                end_index   = n_shots if batch==math.ceil(n_shots/batch_size)-1 else (batch+1)*batch_size
                shot_index  = np.arange(begin_index,end_index)
                record_waveform = self.propagator.forward(fd_order=fd_order,shot_index=shot_index,checkpoint_segments=checkpoint_segments)
                rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz = record_waveform["txx"],record_waveform["tzz"],record_waveform["txz"],record_waveform["vx"],record_waveform["vz"]
                forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz = record_waveform["forward_wavefield_txx"],record_waveform["forward_wavefield_tzz"],record_waveform["forward_wavefield_txz"],record_waveform["forward_wavefield_vx"],record_waveform["forward_wavefield_vz"]                
                
                # misfit calculation
                loss_pressure,loss_vx,loss_vz = 0,0,0
                if "pressure" in self.inversion_component:
                    # forward wavefiled
                    if batch == 0:
                        forw  = -(forward_wavefield_txx+forward_wavefield_tzz).cpu().detach().numpy()
                    else:
                        forw += -(forward_wavefield_txx+forward_wavefield_tzz).cpu().detach().numpy()
                    # synthetic waveform（Pressure = -(\tau_xx + \tau_zz)）
                    syn_p   = -(rcv_txx+rcv_tzz)
                    if self.waveform_normalize:
                        syn_p = syn_p/(torch.max(torch.abs(syn_p),axis=1,keepdim=True).values)
                    # misfit
                    if isinstance(self.loss_fn,Misfit):
                        loss_pressure = self.loss_fn.forward(syn_p,self.obs_p[shot_index])
                    else:
                        loss_pressure = self.loss_fn.apply(syn_p,self.obs_p[shot_index])
                if "vx" in self.inversion_component:
                    # misfit
                    forw = None
                    # forw = forward_wavefield_vx.cpu().detach().numpy()
                    syn_vx = rcv_vx
                    if self.waveform_normalize:
                        syn_vx = syn_vx/(torch.max(torch.abs(syn_vx),axis=1,keepdim=True).values)
                    if isinstance(self.loss_fn,Misfit):
                        loss_vx = self.loss_fn.forward(syn_vx,self.obs_vx[shot_index]) 
                    else:
                        loss_vx = self.loss_fn.apply(syn_vx,self.obs_vx[shot_index])
                if "vz" in self.inversion_component:
                    # misfit
                    forw = None
                    # forw = forward_wavefield_vz.cpu().detach().numpy()
                    syn_vz = rcv_vz
                    if self.waveform_normalize:
                        syn_vz = rcv_vz/(torch.max(torch.abs(rcv_vz),axis=1,keepdim=True).values)
                    if isinstance(self.loss_fn,Misfit):
                        loss_vz = self.loss_fn.forward(syn_vz,self.obs_vz[shot_index]) 
                    else:
                        loss_vz = self.loss_fn.apply(syn_vz,self.obs_vz[shot_index])
                data_loss = loss_pressure+loss_vx + loss_vz
                
                if self.regularization_fn is not None:
                    regularization_loss_vp,regularization_loss_vs,regularization_loss_rho = 0,0,0
                    if self.model.get_requires_grad("vp"):
                        regularization_loss_vp = self.regularization_fn.forward(self.model.vp)
                    if self.model.get_requires_grad("vs"):
                        temp_alphax = self.regularization_fn.alphax 
                        temp_alphaz = self.regularization_fn.alphaz
                        self.regularization_fn.alphax = temp_alphax * 5 
                        self.regularization_fn.alphaz = temp_alphaz * 5 
                        regularization_loss_vs = self.regularization_fn.forward(self.model.vs)
                        self.regularization_fn.alphax = temp_alphax 
                        self.regularization_fn.alphaz = temp_alphaz
                    if self.model.get_requires_grad("rho"):
                        regularization_loss_rho = self.regularization_fn.forward(self.model.rho)
                    regularization_loss = regularization_loss_vp+regularization_loss_vs+regularization_loss_rho
                    loss_batch = loss_batch + data_loss.item() + regularization_loss.item()
                    loss = data_loss + regularization_loss
                else:
                    loss_batch = loss_batch + data_loss.item()
                    loss = data_loss
                loss.backward()
                if math.ceil(n_shots/batch_size) == 1:
                    pbar_batch.set_description(f"Shot:{begin_index} to {end_index}")
            
            # gradient precondition : seperate writting for future processing
            if self.model.get_requires_grad("vp"):
                with torch.no_grad():
                    grads_vp   = self.model.vp.grad.cpu().detach().numpy()
                    vmax_vp    = np.max(self.model.vp.cpu().detach().numpy())
                    grads_vp   = self.gradient_processor.forward(nz=self.model.nz,nx=self.model.nx,vmax=vmax_vp,grad=grads_vp,forw=forw)
                    grads_vp   = numpy2tensor(grads_vp,dtype=self.propagator.dtype).to(self.propagator.device)
                    self.model.vp.grad = grads_vp
            
            if self.model.get_requires_grad("vs"):
                with torch.no_grad():
                    grads_vs   = self.model.vs.grad.cpu().detach().numpy()
                    vmax_vs    = np.max(self.model.vs.cpu().detach().numpy())
                    grads_vs   = self.gradient_processor.forward(nz=self.model.nz,nx=self.model.nx,vmax=vmax_vs,grad=grads_vs,forw=forw)
                    grads_vs   = numpy2tensor(grads_vs,dtype=self.propagator.dtype).to(self.propagator.device)
                    self.model.vs.grad = grads_vs
                        
            if self.model.get_requires_grad("rho"):
                with torch.no_grad():
                    grads_rho   = self.model.rho.grad.cpu().detach().numpy()
                    vmax_rho    = np.max(self.model.rho.cpu().detach().numpy())
                    grads_rho   = self.gradient_processor.forward(nz=self.model.nz,nx=self.model.nx,vmax=vmax_rho,grad=grads_rho,forw=forw)
                    grads_rho   = numpy2tensor(grads_rho,dtype=self.propagator.dtype).to(self.propagator.device)
                    self.model.rho.grad = grads_rho
            
            if isinstance(self.model,AnisotropicElasticModel):
                if self.model.get_requires_grad("eps"):
                    with torch.no_grad():
                        grads_eps   = self.model.eps.grad.cpu().detach().numpy()
                        vmax_eps    = np.max(self.model.eps.cpu().detach().numpy())
                        grads_eps   = self.gradient_processor.forward(nz=self.model.nz,nx=self.model.nx,vmax=vmax_eps,grad=grads_eps,forw=forw)
                        grads_eps   = numpy2tensor(grads_eps,dtype=self.propagator.dtype).to(self.propagator.device)
                        self.model.eps.grad = grads_eps    
                if self.model.get_requires_grad("delta"):
                    with torch.no_grad():
                        grads_delta    = self.model.delta.grad.cpu().detach().numpy()
                        vmax_delta    = np.max(self.model.delta.cpu().detach().numpy())
                        grads_delta   = self.gradient_processor.forward(nz=self.model.nz,nx=self.model.nx,vmax=vmax_delta,grad=grads_delta,forw=forw)
                        grads_delta   = numpy2tensor(grads_delta,dtype=self.propagator.dtype).to(self.propagator.device)
                        self.model.delta.grad = grads_delta
                      
            self.optimizer.step()
            self.scheduler.step()
            
            # save the temp result
            if self.cache_result:
                # save the model
                temp_vp      = self.propagator.model.vp.cpu().detach().numpy()
                self.iter_vp.append(temp_vp)
                temp_vs      = self.propagator.model.vs.cpu().detach().numpy()
                self.iter_vs.append(temp_vs)
                temp_rho     = self.propagator.model.rho.cpu().detach().numpy()
                self.iter_rho.append(temp_rho)
                if isinstance(self.model,AnisotropicElasticModel):
                    temp_eps     = self.propagator.model.eps.cpu().detach().numpy()
                    self.iter_eps.append(temp_eps)
                    temp_delta     = self.propagator.model.delta.cpu().detach().numpy()
                    self.iter_delta.append(temp_delta)
                    temp_gamma     = self.propagator.model.gamma.cpu().detach().numpy()
                    self.iter_gamma.append(temp_gamma)
                self.iter_loss.append(loss_batch)
                
                # save the gradient
                if self.model.get_requires_grad("vp"):
                    temp_grad_vp = grads_vp.cpu().detach().numpy()
                    self.iter_vp_grad.append(temp_grad_vp)
                    self.save_gradient_fig(i,temp_grad_vp,model_type="grad_vp")
                if self.model.get_requires_grad("vs"):
                    temp_grad_vs = grads_vs.cpu().detach().numpy()
                    self.iter_vs_grad.append(temp_grad_vs)
                    self.save_gradient_fig(i,temp_grad_vs,model_type="grad_vs")
                if self.model.get_requires_grad("rho"):
                    temp_grad_rho = grads_rho.cpu().detach().numpy()
                    self.iter_rho_grad.append(temp_grad_rho)
                    self.save_gradient_fig(i,temp_grad_rho,model_type="grad_rho")
                if isinstance(self.model,AnisotropicElasticModel):
                    if self.model.get_requires_grad("eps"):
                        temp_grad_eps = grads_eps.cpu().detach().numpy()
                        self.iter_eps_grad.append(temp_grad_eps)
                        self.save_gradient_fig(i,temp_grad_eps,model_type="grad_eps")
                    if self.model.get_requires_grad("delta"):
                        temp_grad_delta = grads_delta.cpu().detach().numpy()
                        self.iter_delta_grad.append(temp_grad_delta)
                        self.save_gradient_fig(i,temp_grad_delta,model_type="grad_delta")
                
                # save the figure
                self.save_vp_vs_rho_fig(i,temp_vp,temp_vs,temp_rho)
                if isinstance(self.model,AnisotropicElasticModel):
                    self.save_eps_delta_gamma_fig(i,temp_eps,temp_delta,temp_gamma,model_type="eps")
            pbar_epoch.set_description("Iter:{},Loss:{:.4}".format(i+1,loss_batch))