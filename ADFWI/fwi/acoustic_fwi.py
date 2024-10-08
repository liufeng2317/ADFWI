'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-04-26 19:42:24
* LastEditors: LiuFeng
* LastEditTime: 2024-05-22 09:42:26
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
from typing import Optional,Union
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

class AcousticFWI(torch.nn.Module):
    """Acoustic Full waveform inversion class
    """
    def __init__(self,propagator:AcousticPropagator,model:AbstractModel,
                 optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler,
                 loss_fn:Union[Misfit,torch.autograd.Function],
                 obs_data:SeismicData,
                 gradient_processor:Optional[GradProcessor]     = None,
                 regularization_fn:Optional[Regularization]     = None, 
                 waveform_normalize:Optional[bool]              = True,
                 cache_result:Optional[bool]                    = True,
                 save_fig_epoch:Optional[int]                   = -1,
                 save_fig_path:Optional[str]                    = "",
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
        obs_p   = self.obs_data.data["p"]
        obs_p   = numpy2tensor(obs_p,self.dtype).to(self.device)
        if self.waveform_normalize:
            obs_p = obs_p/(torch.max(torch.abs(obs_p),axis=1,keepdim=True).values)
        self.obs_p = obs_p
        
        vp_bound =  self.model.get_bound("vp")
        if vp_bound[0] is None and vp_bound[1] is None:
            self.vp_min = self.model.get_model("vp").min() - 500
            self.vp_max = self.model.get_model("vp").max() + 500
        else: 
            self.vp_min = vp_bound[0]
            self.vp_max = vp_bound[1]
            
        # save result
        self.cache_result   = cache_result
        self.iter_vp        = []
        self.iter_vp_grad   = []
        self.iter_loss      = []
        
        # save figure
        self.save_fig_epoch = save_fig_epoch
        self.save_fig_path  = save_fig_path
    
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
                else:
                    plot_model(data,title=f"Iteration {i}",
                            dx=self.model.dx,dz=self.model.dz,
                            save_path=os.path.join(self.save_fig_path,f"{model_type}_{i}.png"),show=False)
        return

    def forward(self,
                iteration:int,
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
        if isinstance(self.optimizer,torch.optim.LBFGS) or isinstance(self.optimizer,NLCG):
            return self.forward_closure(iteration=iteration,batch_size=batch_size,checkpoint_segments=checkpoint_segments,start_iter=start_iter)

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
                # forward wavefiled
                if batch == 0:
                    forw  = forward_wavefield_p.cpu().detach().numpy()
                else:
                    forw += forward_wavefield_p.cpu().detach().numpy()
                # synthetic waveform
                syn_p   = rcv_p
                if self.waveform_normalize:
                    syn_p = syn_p/(torch.max(torch.abs(syn_p),axis=1,keepdim=True).values)
                # misfit
                if isinstance(self.loss_fn,Misfit):
                    data_loss = self.loss_fn.forward(syn_p,self.obs_p[shot_index])
                elif isinstance(self.loss_fn,Misfit_NIM):
                    data_loss = self.loss_fn.apply(syn_p,self.obs_p[shot_index],self.loss_fn.p,self.loss_fn.trans_type,self.loss_fn.theta)
                else:
                    data_loss = self.loss_fn.apply(syn_p,self.obs_p[shot_index])
                
                if self.regularization_fn is not None:
                    regularization_loss = self.regularization_fn.forward(self.model.vp)
                    loss_batch = loss_batch + data_loss.item() + regularization_loss.item()
                    loss = data_loss + regularization_loss
                else:
                    loss_batch = loss_batch + data_loss.item()
                    loss = data_loss
                loss.backward()
                if math.ceil(n_shots/batch_size) == 1:
                    pbar_batch.set_description(f"Shot:{begin_index} to {end_index}")
            
            # gradient precondition
            grads   = self.model.vp.grad.cpu().detach().numpy()
            if self.gradient_processor != None:
                with torch.no_grad():
                    vmax    = np.max(self.model.vp.cpu().detach().numpy())
                    grads   = self.gradient_processor.forward(nz=self.model.nz,nx=self.model.nx,vmax=vmax,grad=grads,forw=forw)
                    grads   = numpy2tensor(grads,dtype=self.propagator.dtype).to(self.propagator.device)
                    self.model.vp.grad = grads
            self.optimizer.step()
            self.scheduler.step()
            
            # save the temp result
            if self.cache_result:
                temp_vp   = self.propagator.model.vp.cpu().detach().numpy()
                temp_grad = grads.cpu().detach().numpy()
                self.iter_vp.append(temp_vp)
                self.iter_vp_grad.append(temp_grad)
                self.iter_loss.append(loss_batch)
            pbar_epoch.set_description("Iter:{},Loss:{:.4}".format(i+1,loss_batch))
            
            self.true_epoch = 0
            
            # save the result
            self.save_figure(i,temp_vp,model_type="vp")
            self.save_figure(i,temp_grad,model_type="grad_vp")
    
    def forward_closure(self,
                iteration:int,
                batch_size:Optional[int]            = None,
                checkpoint_segments:Optional[int]   = 1 ,
                start_iter                          = 0
                ):
        """ inversion using closure version ==> LBFGS,NLCG
        """
        n_shots = self.propagator.src_n
        if batch_size is None or batch_size > n_shots:
            batch_size = n_shots
        
        # epoch
        pbar_epoch = tqdm(range(start_iter,start_iter+iteration),position=0,leave=False,colour='green',ncols=80)
        self.true_epoch = 0
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
                        forw  = forward_wavefield_p.cpu().detach().numpy()
                    else:
                        forw += forward_wavefield_p.cpu().detach().numpy()
                    # synthetic waveform
                    syn_p   = rcv_p
                    if self.waveform_normalize:
                        syn_p = syn_p/(torch.max(torch.abs(syn_p),axis=1,keepdim=True).values)
                    # misfit
                    if isinstance(self.loss_fn,Misfit):
                        data_loss = self.loss_fn.forward(syn_p,self.obs_p[shot_index])
                    elif isinstance(self.loss_fn,Misfit_NIM):
                        data_loss = self.loss_fn.apply(syn_p,self.obs_p[shot_index],self.loss_fn.p,self.loss_fn.trans_type,self.loss_fn.theta)
                    else:
                        data_loss = self.loss_fn.apply(syn_p,self.obs_p[shot_index])
                    
                    if self.regularization_fn is not None:
                        regularization_loss = self.regularization_fn.forward(self.model.vp)
                        loss_batch = loss_batch + data_loss.item() + regularization_loss.item()
                        loss = data_loss + regularization_loss
                    else:
                        loss_batch = loss_batch + data_loss.item()
                        loss = data_loss
                    loss.backward()
                    if math.ceil(n_shots/batch_size) == 1:
                        pbar_batch.set_description(f"Shot:{begin_index} to {end_index}")
                        
                # gradient precondition (you'd better not do so much precondition for LBFGS and NLCG )
                if self.gradient_processor != None:
                    with torch.no_grad():
                        grads   = self.model.vp.grad.cpu().detach().numpy()
                        vmax    = np.max(self.model.vp.cpu().detach().numpy())
                        grads   = self.gradient_processor.forward(nz=self.model.nz,nx=self.model.nx,vmax=vmax,grad=grads,forw=forw)
                        grads   = numpy2tensor(grads,dtype=self.propagator.dtype).to(self.propagator.device)
                        self.model.vp.grad = grads
                self.true_epoch = self.true_epoch + 1
                return loss_batch
            
            loss_batch = self.optimizer.step(closure=closure)
            self.scheduler.step()
            
            # save the result
            if self.cache_result:
                temp_vp   = self.model.vp.cpu().detach().numpy()
                temp_grad = self.model.get_grad("vp")
                self.iter_vp.append(temp_vp)
                self.iter_vp_grad.append(temp_grad)
                self.iter_loss.append(loss_batch)
            self.save_figure(i,temp_vp     , model_type="vp")
            self.save_figure(i,temp_grad   , model_type="grad_vp")
            pbar_epoch.set_description("Iter:{},Loss:{:.4}".format(i+1,loss_batch))