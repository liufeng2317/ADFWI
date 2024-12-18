'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-26 19:42:24
* LastEditors: LiuFeng
* LastEditTime: 2024-05-22 09:42:26
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''
from typing import Optional,Union,List
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from ADFWI.model       import AbstractModel
from ADFWI.propagator  import AcousticPropagator,GradProcessor
from ADFWI.survey      import SeismicData
from ADFWI.fwi.misfit  import Misfit,Misfit_NIM
from ADFWI.fwi.regularization import Regularization
from ADFWI.fwi.optimizer import NLCG
from ADFWI.utils       import numpy2tensor
from ADFWI.view        import plot_model

from ADFWI.fwi.multiScaleProcessing import lpass

class AcousticFWI(torch.nn.Module):
    """Acoustic Full waveform inversion class
    """
    def __init__(self,propagator:AcousticPropagator,model:AbstractModel,
                 optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler,
                 loss_fn:Union[Misfit,torch.autograd.Function],
                 obs_data:SeismicData,
                 gradient_processor: Union[GradProcessor,List[GradProcessor]] = None,
                 regularization_fn:Optional[Regularization]                   = None, 
                 regularization_weights_x:Optional[List[Union[float]]]        = [0,0], # vp/rho in x direction
                 regularization_weights_z:Optional[List[Union[float]]]        = [0,0], # vp/rho in z direction
                 waveform_normalize:Optional[bool]                            = True,
                 cache_result:Optional[bool]                                  = True,
                 save_fig_epoch:Optional[int]                                 = -1,
                 save_fig_path:Optional[str]                                  = "",
                ):
        """
        Parameters:
        --------------
            propagator (Acoustic Propagator)                : the propagator for the isotropic elastic wave
            model (Model)                                   : the velocity model class
            optimizer (torch.optim.Optimizer)               : the pytorch optimizer
            scheduler (torch.optim.scheduler)               : the pytorch learning rate decay scheduler
            loss_fn   (Misfit or torch.autograd.Function)   : the misfit function
            regularization_fn (Regularization)              : the regularization function
            obs_data  (SeismicData)                         : the observed dataset
            gradient_processor (GradProcessor)              : the gradient processor
            waveform_normalize (bool)   : normalize the waveform or not, default True
            cache_result (bool)         : save the temp result of the inversion or not
        """
        super().__init__()
        self.propagator                 = propagator
        self.model                      = model
        self.optimizer                  = optimizer
        self.scheduler                  = scheduler
        self.loss_fn                    = loss_fn
        self.regularization_fn          = regularization_fn
        self.regularization_weights_x   = regularization_weights_x
        self.regularization_weights_z   = regularization_weights_z
        self.obs_data                   = obs_data
        self.gradient_processor         = gradient_processor
        self.device                     = self.propagator.device
        self.dtype                      = self.propagator.dtype 
        
        # observed data
        self.waveform_normalize = waveform_normalize
        obs_p   = self.obs_data.data["p"]
        obs_p   = numpy2tensor(obs_p,self.dtype).to(self.device)
        if self.waveform_normalize:
            # obs_p = obs_p/(torch.max(torch.abs(obs_p),axis=1,keepdim=True).values)
            obs_p = self._normalize(obs_p)
        self.obs_p = obs_p
        
        # model boundary
        vp_bound =  self.model.get_bound("vp")
        if vp_bound[0] is None and vp_bound[1] is None:
            self.vp_min = self.model.get_model("vp").min() - 500
            self.vp_max = self.model.get_model("vp").max() + 500
        else: 
            self.vp_min = vp_bound[0]
            self.vp_max = vp_bound[1]
            if self.model.water_layer_mask is not None:
                self.vp_min = 1500

        rho_bound =  self.model.get_bound("rho")
        if rho_bound[0] is None and rho_bound[1] is None:
            self.rho_min = self.model.get_model("rho").min() - 500
            self.rho_max = self.model.get_model("rho").max() + 500
        else: 
            self.rho_min = rho_bound[0]
            self.rho_max = rho_bound[1]
            if self.model.water_layer_mask is not None:
                self.rho_min = 1000
        
        # save result
        self.cache_result   = cache_result
        self.iter_vp, self.iter_rho = [],[]
        self.iter_vp_grad, self.iter_rho_grad = [],[]
        self.iter_loss      = []
        
        # save figure
        self.save_fig_epoch = save_fig_epoch
        self.save_fig_path  = save_fig_path
    
    def _normalize(self,data):
        mask = torch.sum(torch.abs(data),axis=1,keepdim=True) == 0
        max_val = torch.max(torch.abs(data),axis=1,keepdim=True).values
        max_val = max_val.masked_fill(mask, 1)
        data = data/max_val
        return data
    
    # misfits calculation
    def calculate_loss(self, synthetic_waveform, observed_waveform, normalization, loss_fn, cutoff_freq=None, propagator_dt=None):
        """
        Generalized function to calculate misfit loss for a given component.
        """
        if normalization:
            # synthetic_waveform = synthetic_waveform / (torch.max(torch.abs(synthetic_waveform), axis=1, keepdim=True).values)
            synthetic_waveform = self._normalize(synthetic_waveform)
        # Apply low-pass filter if cutoff frequency is provided
        if cutoff_freq is not None:
            synthetic_waveform, observed_waveform = lpass(synthetic_waveform, observed_waveform, cutoff_freq, int(1 / propagator_dt))
        if isinstance(loss_fn, Misfit):
            return loss_fn.forward(synthetic_waveform, observed_waveform)
        elif isinstance(loss_fn,Misfit_NIM):
            return loss_fn.apply(synthetic_waveform,observed_waveform,loss_fn.p,loss_fn.trans_type,loss_fn.theta)
        else:
            return loss_fn.apply(synthetic_waveform, observed_waveform)
    
    # regularization calculation
    def calculate_regularization_loss(self, model_param, weight_x, weight_z, regularization_fn):
        """
        Generalized function to calculate regularization loss for a given parameter.
        """
        regularization_loss = torch.tensor(0.0, device=model_param.device)
        # Check if the parameter requires gradient
        if model_param.requires_grad:
            # Set the regularization weights for x and z directions
            regularization_fn.alphax = weight_x
            regularization_fn.alphaz = weight_z
            # Calculate regularization loss if any weight is greater than zero
            if regularization_fn.alphax > 0 or regularization_fn.alphaz > 0:
                regularization_loss = regularization_fn.forward(model_param)
        return regularization_loss

    # gradient precondition
    def process_gradient(self, parameter,forw,idx=None):
        with torch.no_grad():
            grads = parameter.grad.cpu().detach().numpy()
            vmax = np.max(parameter.cpu().detach().numpy())
            # Apply gradient processor
            if isinstance(self.gradient_processor, GradProcessor):
                grads = self.gradient_processor.forward(nz=self.model.nz, nx=self.model.nx, vmax=vmax, grad=grads, forw=forw)
            else:
                grads = self.gradient_processor[idx].forward(nz=self.model.nz, nx=self.model.nx, vmax=vmax, grad=grads, forw=forw)
            # Convert grads back to tensor and assign
            grads_tensor = numpy2tensor(grads, dtype=self.propagator.dtype).to(self.propagator.device)
            parameter.grad = grads_tensor

    def save_figure(self,i,data,model_type="vp"):
        if self.save_fig_epoch == -1:
            pass
        elif i%self.save_fig_epoch == 0:
            if os.path.exists(self.save_fig_path):
                if model_type == "vp":
                    plot_model(data,title=f"Iteration {i}",
                            dx=self.model.dx,dz=self.model.dz,
                            vmin=self.vp_min,vmax=self.vp_max,
                            save_path=os.path.join(self.save_fig_path,f"{model_type}_{i}.png"),show=False)
                elif model_type == "rho":
                    plot_model(data,title=f"Iteration {i}",
                            dx=self.model.dx,dz=self.model.dz,
                            vmin=self.rho_min,vmax=self.rho_max,
                            save_path=os.path.join(self.save_fig_path,f"{model_type}_{i}.png"),show=False)
                else:
                    plot_model(data,title=f"Iteration {i}",
                            dx=self.model.dx,dz=self.model.dz,
                            save_path=os.path.join(self.save_fig_path,f"{model_type}_{i}.png"),show=False,cmap='coolwarm')
        return
    
    def forward(self,
                iteration:int,
                batch_size:Optional[int]            = None,
                checkpoint_segments:Optional[int]   = 1 ,
                start_iter                          = 0,
                cutoff_freq                         = None,
                ):
        """
        Parameters:
        ------------
            iteration (int)             : the iteration number of inversion
            fd_order (int)              : the order of finite difference
            batch_size (int)            : the shots for each batch, default -1 means use all the shots
            checkpoint_segments (int)   : seperate all the time seris into N segments for saving memory, default 1
        """
        if isinstance(self.optimizer,torch.optim.LBFGS) or isinstance(self.optimizer,NLCG):
            return self.forward_closure(iteration=iteration,batch_size=batch_size,checkpoint_segments=checkpoint_segments,start_iter=start_iter,cutoff_freq=cutoff_freq)

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
                record_waveform = self.propagator.forward(shot_index=shot_index,checkpoint_segments=checkpoint_segments)
                rcv_p,rcv_u,rcv_w = record_waveform["p"],record_waveform["u"],record_waveform["w"]
                forward_wavefield_p,forward_wavefield_u,forward_wavefield_w = record_waveform["forward_wavefield_p"],record_waveform["forward_wavefield_u"],record_waveform["forward_wavefield_w"]
                
                # misfits
                if batch == 0:
                    forw  = forward_wavefield_p.cpu().detach().numpy()
                else:
                    forw += forward_wavefield_p.cpu().detach().numpy()
                syn_p   = rcv_p
                data_loss = self.calculate_loss(syn_p, self.obs_p[shot_index], self.waveform_normalize, self.loss_fn, cutoff_freq, self.propagator.dt)
                
                # regularization
                if self.regularization_fn is not None:
                    regularization_loss_vp  = self.calculate_regularization_loss(self.model.vp , self.regularization_weights_x[0], self.regularization_weights_z[0], self.regularization_fn)
                    regularization_loss_rho = self.calculate_regularization_loss(self.model.rho, self.regularization_weights_x[1], self.regularization_weights_z[1], self.regularization_fn)
                    regularization_loss = regularization_loss_vp+regularization_loss_rho
                    loss_batch = loss_batch + data_loss.item() + regularization_loss.item()
                    loss = data_loss + regularization_loss
                else:
                    loss_batch = loss_batch + data_loss.item()
                    loss = data_loss
                loss.backward()
                if math.ceil(n_shots/batch_size) == 1:
                    pbar_batch.set_description(f"Shot:{begin_index} to {end_index}")
            
            # gradient process
            if self.model.get_requires_grad("vp"):
                self.process_gradient(self.model.vp, forw=forw, idx=0)
            if self.model.get_requires_grad("rho"):
                self.process_gradient(self.model.rho, forw=forw, idx=1)
        
            self.optimizer.step()
            self.scheduler.step()
            
            # constrain the velocity model
            self.model.forward()
            
            if self.cache_result:
                # model
                temp_vp   = self.model.vp.cpu().detach().numpy()
                temp_rho  = self.model.rho.cpu().detach().numpy()
                self.iter_vp.append(temp_vp)
                self.iter_rho.append(temp_rho)
                self.iter_loss.append(loss_batch)
                self.save_figure(i,temp_vp     , model_type="vp")
                self.save_figure(i,temp_rho    , model_type="rho")
                # gradient
                if self.model.get_requires_grad("vp"):
                    grads_vp   = self.model.vp.grad.cpu().detach().numpy()
                    self.save_figure(i,grads_vp    , model_type="grad_vp")
                    self.iter_vp_grad.append(grads_vp)
                if self.model.get_requires_grad("rho"):
                    grads_rho  = self.model.rho.grad.cpu().detach().numpy()
                    self.save_figure(i,grads_rho   , model_type="grad_rho")
                    self.iter_rho_grad.append(grads_rho)

            self.true_epoch = 0
            pbar_epoch.set_description("Iter:{},Loss:{:.4}".format(i+1,loss_batch))
    
    def forward_closure(self,
                iteration:int,
                batch_size:Optional[int]            = None,
                checkpoint_segments:Optional[int]   = 1 ,
                start_iter                          = 0 ,
                cutoff_freq                         = None,
                ):
        """ inversion using closure version ==> LBFGS,NLCG
        """
        n_shots = self.propagator.src_n
        if batch_size is None or batch_size > n_shots:
            batch_size = n_shots
        
        # epoch
        pbar_epoch = tqdm(range(start_iter,start_iter+iteration),position=0,leave=False,colour='green',ncols=80)
        self.true_epoch = 0
        self.forw = None
        for i in pbar_epoch:
            def closure():
                # batch (for the clouser we hold 1 batch)
                self.optimizer.zero_grad()
                loss_batch = 0
                pbar_batch = tqdm(range(math.ceil(n_shots/batch_size)),position=1,leave=False,colour='red',ncols=80)
                for batch in pbar_batch:
                    # forward simulation
                    begin_index = 0  if batch==0 else batch*batch_size
                    end_index   = n_shots if batch==math.ceil(n_shots/batch_size)-1 else (batch+1)*batch_size
                    shot_index  = np.arange(begin_index,end_index)
                    record_waveform = self.propagator.forward(shot_index=shot_index,checkpoint_segments=checkpoint_segments)
                    rcv_p,rcv_u,rcv_w = record_waveform["p"],record_waveform["u"],record_waveform["w"]
                    forward_wavefield_p,forward_wavefield_u,forward_wavefield_w = record_waveform["forward_wavefield_p"],record_waveform["forward_wavefield_u"],record_waveform["forward_wavefield_w"]
                    
                    # forward wavefiled
                    if batch == 0:
                        self.forw  = forward_wavefield_p.cpu().detach().numpy()
                    else:
                        self.forw += forward_wavefield_p.cpu().detach().numpy()
                    
                    # misfits
                    syn_p   = rcv_p
                    data_loss = self.calculate_loss(syn_p, self.obs_p[shot_index], self.waveform_normalize, self.loss_fn, cutoff_freq, self.propagator.dt)
                    
                    # regularization
                    if self.regularization_fn is not None:
                        regularization_loss_vp  = self.calculate_regularization_loss(self.model.vp , self.regularization_weights_x[0], self.regularization_weights_z[0], self.regularization_fn)
                        regularization_loss_rho = self.calculate_regularization_loss(self.model.rho, self.regularization_weights_x[1], self.regularization_weights_z[1], self.regularization_fn)
                        regularization_loss = regularization_loss_vp+regularization_loss_rho
                        loss_batch = loss_batch + data_loss.item() + regularization_loss.item()
                        loss = data_loss + regularization_loss
                    else:
                        loss_batch = loss_batch + data_loss.item()
                        loss = data_loss
                    loss.backward()
                    if math.ceil(n_shots/batch_size) == 1:
                        pbar_batch.set_description(f"Shot:{begin_index} to {end_index}")
                self.true_epoch = self.true_epoch + 1
                # gradient process
                if self.model.get_requires_grad("vp"):
                    self.process_gradient(self.model.vp, forw=self.forw, idx=0)
                if self.model.get_requires_grad("rho"):
                    self.process_gradient(self.model.rho, forw=self.forw, idx=1)
                return loss_batch
            
            loss_batch = self.optimizer.step(closure=closure)
            self.scheduler.step()
            
            # constrain the velocity model
            self.model.forward()
            
            # save the result
            if self.cache_result:
                # save the inverted resutls
                temp_vp   = self.model.vp.cpu().detach().numpy()
                temp_rho  = self.model.rho.cpu().detach().numpy()
                self.iter_vp.append(temp_vp)
                self.iter_rho.append(temp_rho)
                self.iter_loss.append(loss_batch)
                
                self.save_figure(i,temp_vp     , model_type="vp")
                self.save_figure(i,temp_rho    , model_type="rho")
                
                # save the inverted gradient
                if self.model.get_requires_grad("vp"):
                    grads_vp   = self.model.vp.grad.cpu().detach().numpy()
                    self.save_figure(i,grads_vp    , model_type="grad_vp")
                    self.iter_vp_grad.append(grads_vp)
                if self.model.get_requires_grad("rho"):
                    grads_rho  = self.model.rho.grad.cpu().detach().numpy()
                    self.save_figure(i,grads_rho   , model_type="grad_rho")
                    self.iter_rho_grad.append(grads_rho)
            pbar_epoch.set_description("Iter:{},Loss:{:.4}".format(i+1,loss_batch))