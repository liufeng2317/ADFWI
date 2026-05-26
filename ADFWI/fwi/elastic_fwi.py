'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-26 19:42:24
* LastEditors: LiuFeng
* LastEditTime: 2024-06-01 23:01:28
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''
from typing import Optional,Union,List,Mapping
import os
import torch
import numpy as np
from tqdm import tqdm
from ADFWI.model       import AbstractModel,IsotropicElasticModel,AnisotropicElasticModel
from ADFWI.propagator  import ElasticPropagator,GradProcessor
from ADFWI.survey      import SeismicData
from ADFWI.fwi.misfit  import Misfit
from ADFWI.fwi.regularization import Regularization
from ADFWI.fwi.runtime import align_regularization_backend, validate_model_propagator_devices
from ADFWI.fwi.iteration import build_batch_loss, iter_batch_ranges, set_batch_description
from ADFWI.fwi.data import (
    ELASTIC_COMPONENTS,
    build_fwi_data_transform_pipeline,
    build_fwi_transform_context,
    elastic_component_loss_inputs,
    elastic_observed_components,
    elastic_synthetic_components,
    normalize_elastic_component_weights,
    evaluate_misfit_loss,
    normalize_waveform,
    prepare_fwi_loss_pair,
    prepare_loss_pair,
    sum_weighted_losses,
)
from ADFWI.fwi.transforms import DataTransformPipeline
from ADFWI.utils       import numpy2tensor
from ADFWI.view        import plot_vp_vs_rho,plot_model,plot_eps_delta_gamma


class ElasticFWI(torch.nn.Module):
    """Elastic Full waveform inversion class
    """
    def __init__(self,propagator:ElasticPropagator,model:AbstractModel,
                 loss_fn:Union[Misfit,torch.autograd.Function],
                 obs_data:SeismicData,
                 optimizer:Union[torch.optim.Optimizer,List[torch.optim.Optimizer]]      = None,
                 scheduler:torch.optim.lr_scheduler                                      = None,
                 gradient_processor: Union[GradProcessor,List[GradProcessor]]            = None,                # vp/vs/rho epsilon/delta/gamma
                 regularization_fn:Optional[Regularization]                              = None,
                 regularization_weights_x:Optional[List[Union[float]]]                   = None,              # vp/vs/rho epsilon/delta/gamma
                 regularization_weights_z:Optional[List[Union[float]]]                   = None,              # vp/vs/rho epsilon/delta/gamma
                 waveform_normalize:Optional[bool]                                       = True,
                 waveform_mute_late_window:Optional[float]                               = None,
                 waveform_mute_offset:Optional[float]                                    = None,
                 data_transform_pipeline:Optional[DataTransformPipeline]                  = None,
                 cache_result:Optional[bool]                                             = True,
                 cache_result_epoch:Optional[bool]                                       = 1,
                 cache_gradient:Optional[bool]                                           = False,
                 save_fig_epoch:Optional[int]                                            = -1,
                 save_fig_path:Optional[str]                                             = "",
                 inversion_component:Optional[np.array]                                  = None,
                 component_weights:Optional[Mapping[str, float]]                         = None,
                ):
        """
        Parameters:
        --------------
        propagator (ElasticPropagator)                                          : The propagator used for simulating elastic wave propagation.
        model (AbstractModel)                                                   : The model class representing the velocity structure.
        loss_fn (Union[Misfit, torch.autograd.Function])                        : The loss function used to compute the misfit between observed and predicted data.
        obs_data (SeismicData)                                                  : The observed seismic data.
        optimizer (Union[torch.optim.Optimizer, List[torch.optim.Optimizer]])   : The optimizer or list of optimizers for model parameters. Default is None.
        scheduler (Optional[torch.optim.lr_scheduler])                          : The learning rate scheduler for optimizing the model parameters. Default is None.
        gradient_processor (Union[GradProcessor, List[GradProcessor]])          : Processor(s) for handling gradients (e.g., vp/vs/rho, epsilon/delta/gamma). Default is None.
        regularization_fn (Optional[Regularization])                            : Regularization function(s) applied to parameters like vp/vs/rho/epsilon/delta/gamma. Default is None.
        regularization_weights_x (Optional[List[Union[float]]])                 : Regularization weights for the x-axis. Default is [0, 0, 0, 0, 0, 0].
        regularization_weights_z (Optional[List[Union[float]]])                 : Regularization weights for the z-axis. Default is [0, 0, 0, 0, 0, 0].
        waveform_normalize (Optional[bool])                                     : Whether to normalize waveforms. In bv1.2 the default path implements this through the internal transform pipeline. Set False when a custom pipeline already normalizes data.
        data_transform_pipeline (Optional[DataTransformPipeline])               : Optional extra synthetic/observed waveform transform pipeline. It is appended after legacy-compatible mute, low-pass, and data-mask transforms; transforms must operate on same-shape synthetic/observed tensors.
        cache_result (Optional[bool])                                           : Whether to save intermediate results during the inversion. Default is True.
        cache_gradient (Optional[bool])                                         : Whether to save model variations (not gradients) during inversion. Default is False.
        save_fig_epoch (Optional[int])                                          : The interval (in epochs) at which to save the inversion result figure. Default is -1 (no figure saved).
        save_fig_path (Optional[str])                                           : The path where to save the inversion result figure. Default is an empty string (no save path).
        inversion_component (Optional[np.array])                                : Elastic components used in the inversion. Supported names are "pressure", "vx", and "vz". Default is ["pressure"].
        component_weights (Optional[Mapping[str, float]])                       : Optional per-component loss weights for pressure/vx/vz. Missing active components default to 1.0; unknown names and negative weights raise ValueError.
        """
        super().__init__()
        self.propagator                 = propagator
        self.model                      = model
        self.optimizer                  = optimizer
        self.scheduler                  = scheduler
        self.loss_fn                    = loss_fn
        self.regularization_fn          = regularization_fn
        self.regularization_weights_x   = list(regularization_weights_x) if regularization_weights_x is not None else [0, 0, 0, 0, 0, 0]
        self.regularization_weights_z   = list(regularization_weights_z) if regularization_weights_z is not None else [0, 0, 0, 0, 0, 0]
        self.obs_data                   = obs_data
        self.gradient_processor         = gradient_processor
        self.device                     = self.propagator.device
        self.dtype                      = self.propagator.dtype 
        validate_model_propagator_devices(self.model, self.propagator)
        align_regularization_backend(self.regularization_fn, self.device, self.dtype)
        
        # Real-Case settings: for trace missing, partial data missing
        receiver_masks = self.propagator.receiver_masks
        if receiver_masks is None:
            receiver_masks  = np.ones((self.propagator.src_n,self.propagator.rcv_n))
        receiver_masks      = numpy2tensor(receiver_masks)
        self.receiver_masks_2D = receiver_masks # [shot, rcv]
        self.receiver_masks_3D = receiver_masks.unsqueeze(1).expand(-1, self.propagator.nt, -1).to(self.device)  # [shot, time, rcv]
        
        # Real-Case settings: mute late window (by first arrival picking) & mute offset
        self.data_transform_pipeline, self.waveform_normalize = self._configure_data_transform_pipeline(
            data_transform_pipeline, waveform_normalize
        )
        self.waveform_mute_late_window  = waveform_mute_late_window 
        self.waveform_mute_offset       = waveform_mute_offset
        
        # observed data
        observed_components = {
            name: numpy2tensor(component, self.dtype).to(self.device)
            for name, component in elastic_observed_components(self.obs_data.data).items()
        }
        if self.propagator.receiver_masks_obs: # mark the observed data need to be masked or not (trace)
            observed_components = {
                name: component * self.receiver_masks_3D
                for name, component in observed_components.items()
            }
        self.data_masks = numpy2tensor(self.obs_data.data_masks).to(self.device) if self.obs_data.data_masks is not None else None
        if self.data_masks is not None: # some of the data are unuseful (data)
            observed_components = {
                name: component * self.data_masks
                for name, component in observed_components.items()
            }
        self.obs_components = observed_components
        self.obs_p = observed_components["pressure"]
        self.obs_vx = observed_components["vx"]
        self.obs_vz = observed_components["vz"]
        
        # save result
        self.cache_result   = cache_result
        self.cache_result_epoch = cache_result_epoch
        self.cache_gradient = cache_gradient
        self.iter_vp,self.iter_vs,self.iter_rho = [],[],[]       
        self.iter_eps,self.iter_delta,self.iter_gamma = [],[],[]
        self.iter_vp_grad,self.iter_vs_grad,self.iter_rho_grad = [],[],[]
        self.iter_eps_grad,self.iter_delta_grad,self.iter_gamma_grad = [],[],[]
        self.cache_iter_index = []
        self.iter_loss      = []
        
        # save figure
        self.save_fig_epoch = save_fig_epoch
        self.save_fig_path  = save_fig_path
        
        # inversion component
        self.inversion_component = list(inversion_component) if inversion_component is not None else ["pressure"]
        self.component_weights = normalize_elastic_component_weights(self.inversion_component, component_weights)
    
    def _configure_data_transform_pipeline(self, data_transform_pipeline, waveform_normalize):
        return build_fwi_data_transform_pipeline(data_transform_pipeline, waveform_normalize)


    def _normalize(self, data):
        return normalize_waveform(data)
    
    def _build_transform_context(self, shot_index=None, cutoff_freq=None, propagator_dt=None):
        return build_fwi_transform_context(
            shot_index=shot_index,
            cutoff_freq=cutoff_freq,
            propagator_dt=propagator_dt,
            default_dt=self.propagator.dt,
            late_window=self.waveform_mute_late_window,
            offset_mute_threshold=self.waveform_mute_offset,
            dx=self.propagator.dx,
            receiver_masks_2d=self.receiver_masks_2D,
            src_x=self.propagator.src_x,
            rcv_x=self.propagator.rcv_x,
            data_masks=self.data_masks,
        )

    def _prepare_loss_pair(self, synthetic_waveform, observed_waveform, shot_index=None, cutoff_freq=None, propagator_dt=None):
        return prepare_fwi_loss_pair(
            synthetic_waveform,
            observed_waveform,
            shot_index=shot_index,
            cutoff_freq=cutoff_freq,
            propagator_dt=propagator_dt,
            default_dt=self.propagator.dt,
            late_window=self.waveform_mute_late_window,
            offset_mute_threshold=self.waveform_mute_offset,
            dx=self.propagator.dx,
            receiver_masks_2d=self.receiver_masks_2D,
            src_x=self.propagator.src_x,
            rcv_x=self.propagator.rcv_x,
            data_masks=self.data_masks,
            data_transform_pipeline=self.data_transform_pipeline,
        )

    # misfits calculation
    def calculate_loss(self, synthetic_waveform, observed_waveform, normalization, loss_fn, cutoff_freq=None, propagator_dt=None, shot_index=None, apply_transforms=True):
        """
        Generalized function to calculate misfit loss for a given component.
        """
        if apply_transforms:
            synthetic_waveform, observed_waveform = self._prepare_loss_pair(
                synthetic_waveform,
                observed_waveform,
                shot_index=shot_index,
                cutoff_freq=cutoff_freq,
                propagator_dt=propagator_dt,
            )

        if normalization:
            observed_waveform  = self._normalize(observed_waveform)
            synthetic_waveform = self._normalize(synthetic_waveform)
        
        return evaluate_misfit_loss(
            loss_fn,
            synthetic_waveform,
            observed_waveform,
            function_fallback="call",
        )
    
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
    
    def calculate_model_regularization_loss(self):
        parameter_names = ["vp", "vs", "rho"]
        if isinstance(self.model, AnisotropicElasticModel):
            parameter_names.extend(["eps", "delta", "gamma"])

        regularization_loss = None
        for idx, name in enumerate(parameter_names):
            parameter_loss = self.calculate_regularization_loss(
                getattr(self.model, name),
                self.regularization_weights_x[idx],
                self.regularization_weights_z[idx],
                self.regularization_fn,
            )
            regularization_loss = parameter_loss if regularization_loss is None else regularization_loss + parameter_loss
        return regularization_loss
    
    # gradient precondition
    def process_gradient(self, parameter, forw, idx=None):
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
        return

    def save_vp_vs_rho_fig(self,epoch_id,vp,vs,rho):
        vp_bound    =  self.model.get_bound("vp")
        vs_bound    =  self.model.get_bound("vs")
        rho_bound   =  self.model.get_bound("rho")
        if vp_bound[0] is None and vp_bound[1] is None:
            self.vp_min = self.model.get_model("vp").min() - 500
            self.vp_max = self.model.get_model("vp").max() + 500
        else: 
            self.vp_min = vp_bound[0]
            self.vp_max = vp_bound[1]
            if self.model.water_layer_mask is not None:
                self.vp_min = 1500
        if vs_bound[0] is None and vs_bound[1] is None:
            self.vs_min = self.model.get_model("vs").min() - 500
            self.vs_max = self.model.get_model("vs").max() + 500
        else: 
            self.vs_min = vs_bound[0]
            self.vs_max = vs_bound[1]
            if self.model.water_layer_mask is not None:
                self.vs_min = 0        
        if rho_bound[0] is None and rho_bound[1] is None:
            self.rho_min = self.model.get_model("rho").min() - 200
            self.rho_max = self.model.get_model("rho").max() + 200
        else: 
            self.rho_min = rho_bound[0]
            self.rho_max = rho_bound[1]
            if self.model.water_layer_mask is not None:
                self.rho_min = 1000
        
        if self.save_fig_epoch == -1:
            pass
        elif epoch_id%self.save_fig_epoch == 0:
            if os.path.exists(self.save_fig_path):
                plot_vp_vs_rho(
                    vp=vp,vs=vs,rho=rho,
                    # title=f"Iteration {i}",
                    figsize=(12,5),wspace=0.2,cbar_pad_fraction=0.18,cbar_height=0.04,
                    dx=self.model.dx,dz=self.model.dz,
                    vp_min=self.vp_min,vp_max=self.vp_max,
                    vs_min=self.vs_min,vs_max=self.vs_max,
                    rho_min=self.rho_min,rho_max=self.rho_max,
                    save_path=os.path.join(self.save_fig_path,f"model_{epoch_id}.png"),
                    show=False
                    )
        return
    
    def save_eps_delta_gamma_fig(self,epoch_id,eps,delta,gamma):
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
        elif epoch_id%self.save_fig_epoch == 0:
            if os.path.exists(self.save_fig_path):
                plot_eps_delta_gamma(
                    eps=eps,delta=delta,gamma=gamma,
                    # title=f"Iteration {i}",
                    figsize=(12,5),wspace=0.3,cbar_pad_fraction=0.01,cbar_height=0.04,
                    dx=self.model.dx,dz=self.model.dz,
                    save_path=os.path.join(self.save_fig_path,f"anisotropic_model_{epoch_id}.png"),
                    show=False
                    )
        return
    
    def save_gradient_fig(self,epoch_id,data,model_type="vp"):
        if self.save_fig_epoch == -1:
            pass
        elif epoch_id%self.save_fig_epoch == 0:
            if os.path.exists(self.save_fig_path):
                plot_model(data,title=f"Iteration {epoch_id}",
                        dx=self.model.dx,dz=self.model.dz,
                        save_path=os.path.join(self.save_fig_path,f"{model_type}_{epoch_id}.png"),
                        show=False,cmap='seismic')
        return
    
    def save_model_and_gradients(self,epoch_id,loss_epoch):
        """
            Save model parameters and gradients if caching is enabled.
        """
        # Save the loss
        self.iter_loss.append(loss_epoch)

        # Save the model parameters
        param_names = ["vp", "vs", "rho"]
        anisotropic_params = ["eps", "delta", "gamma"] if isinstance(self.model, AnisotropicElasticModel) else []
        if epoch_id % self.cache_result_epoch == 0:
            for name in param_names + anisotropic_params:
                param = getattr(self.model, name, None)
                if param is not None:
                    temp_param = param.cpu().detach().numpy()
                    getattr(self, f"iter_{name}").append(temp_param)
            self.cache_iter_index.append(epoch_id)
        
        # save the figure
        self.save_vp_vs_rho_fig(epoch_id,self.model.vp.cpu().detach().numpy(),
                                         self.model.vs.cpu().detach().numpy(),
                                         self.model.rho.cpu().detach().numpy())
        if isinstance(self.model,AnisotropicElasticModel):
            self.save_eps_delta_gamma_fig(epoch_id,
                                          self.model.eps.cpu().detach().numpy(),
                                          self.model.delta.cpu().detach().numpy(),
                                          self.model.gamma.cpu().detach().numpy())

        # Save gradients if required
        for name in param_names:
            if self.model.get_requires_grad(name):
                temp_grad = getattr(self.model, name).grad.cpu().detach().numpy()
                getattr(self, f"iter_{name}_grad").append(temp_grad)
                self.save_gradient_fig(epoch_id, temp_grad, model_type=f"grad_{name}")

        # For anisotropic model parameters, save gradients if required
        if isinstance(self.model, AnisotropicElasticModel):
            for name in anisotropic_params:
                if self.model.get_requires_grad(name):
                    temp_grad = getattr(self.model, name).grad.cpu().detach().numpy()
                    getattr(self, f"iter_{name}_grad").append(temp_grad)
                    self.save_gradient_fig(epoch_id, temp_grad, model_type=f"grad_{name}")
        return
    
    def real_case_data_selecting(self,rcv_p,shot_index):
        return prepare_loss_pair(
            rcv_p,
            self.obs_p[shot_index],
            receiver_mask=self.receiver_masks_2D[shot_index],
            data_transform_pipeline=None,
        )[0]
    
    def forward(self,
                iteration:int,
                fd_order:int                        = 4,
                batch_size:Optional[int]            = None,
                checkpoint_segments:Optional[int]   = 1 ,
                start_iter                          = 0,
                cutoff_freq                         = None,
                ):
        """
        Parameters:
        ------------
        iteration (int)                     : The maximum iteration number in the inversion process.
        fd_order (int)                      : The order of the finite difference scheme for wave propagation. Default is 4.
        batch_size (Optional[int])          : The number of shots (data samples) in each batch. Default is None, meaning use all available shots.
        checkpoint_segments (Optional[int]) : The number of segments into which the time series should be divided for memory efficiency. Default is 1, which means no segmentation.
        start_iter (int)                    : The starting iteration for the optimization process (e.g., for optimizers like Adam/AdamW, and learning rate schedulers like step_lr). Default is 0.
        cutoff_freq (Optional[float])       : The cutoff frequency for low-pass filtering, if specified. Default is None (no filtering applied).
        """
        n_shots = self.propagator.src_n
        batch_ranges = list(iter_batch_ranges(n_shots, batch_size))
        
        # epoch
        pbar_epoch = tqdm(range(start_iter,start_iter+iteration),position=0,leave=False,colour='green',ncols=80)
        for i in pbar_epoch:
            # batch
            self.optimizer.zero_grad()
            loss_epoch = 0
            pbar_batch = tqdm(batch_ranges,position=1,leave=False,colour='red',ncols=80)
            for batch_range in pbar_batch:
                # forward simulation
                begin_index     = batch_range.begin
                end_index       = batch_range.end
                shot_index      = batch_range.shot_index
                record_waveform = self.propagator.forward(fd_order=fd_order,shot_index=shot_index,checkpoint_segments=checkpoint_segments)
                rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz = record_waveform["txx"],record_waveform["tzz"],record_waveform["txz"],record_waveform["vx"],record_waveform["vz"]
                forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz = record_waveform["forward_wavefield_txx"],record_waveform["forward_wavefield_tzz"],record_waveform["forward_wavefield_txz"],record_waveform["forward_wavefield_vx"],record_waveform["forward_wavefield_vz"]
                if batch_range.batch == 0:
                    if "pressure" in self.inversion_component:
                        forw_p  = -(forward_wavefield_txx + forward_wavefield_tzz).cpu().detach().numpy()
                    if "vx" in self.inversion_component:
                        forw_vx = forward_wavefield_vx.cpu().detach().numpy()
                    if "vz" in self.inversion_component:
                        forw_vz = forward_wavefield_vz.cpu().detach().numpy()
                else:
                    if "pressure" in self.inversion_component:
                        forw_p  += -(forward_wavefield_txx + forward_wavefield_tzz).cpu().detach().numpy()
                    if "vx" in self.inversion_component:
                        forw_vx += forward_wavefield_vx.cpu().detach().numpy()
                    if "vz" in self.inversion_component:
                        forw_vz += forward_wavefield_vz.cpu().detach().numpy()

                # misfits
                synthetic_components = elastic_synthetic_components(record_waveform)
                component_losses = []
                for component, synthetic_component, observed_component, component_weight in elastic_component_loss_inputs(
                    synthetic_components,
                    self.obs_components,
                    self.inversion_component,
                    self.component_weights,
                ):
                    synthetic_waveform, observed_waveform = self._prepare_loss_pair(
                        synthetic_component,
                        observed_component[shot_index],
                        shot_index,
                        cutoff_freq,
                        self.propagator.dt,
                    )
                    component_loss = self.calculate_loss(
                        synthetic_waveform,
                        observed_waveform,
                        self.waveform_normalize,
                        self.loss_fn,
                        apply_transforms=False,
                    )
                    component_losses.append(component_loss * component_weight)
                data_loss = sum_weighted_losses(component_losses, device=self.device)
                
                # regularization
                regularization_loss = self.calculate_model_regularization_loss() if self.regularization_fn is not None else None
                batch_loss = build_batch_loss(data_loss, regularization_loss)
                loss_epoch += batch_loss.scalar
                batch_loss.tensor.backward()
                set_batch_description(pbar_batch, batch_range, len(batch_ranges))
            
            # gradient process
            if self.model.get_requires_grad("vp"):
                self.process_gradient(self.model.vp,  forw=forw_p if "pressure" in self.inversion_component else forw_vz, idx=0)
            if self.model.get_requires_grad("vs"):
                self.process_gradient(self.model.vs,  forw=forw_p if "pressure" in self.inversion_component else forw_vz, idx=1)
            if self.model.get_requires_grad("rho"):
                self.process_gradient(self.model.rho, forw=forw_p if "pressure" in self.inversion_component else forw_vz, idx=2)
            if isinstance(self.model, AnisotropicElasticModel):
                if self.model.get_requires_grad("eps"):
                    self.process_gradient(self.model.eps, forw=forw_p if "pressure" in self.inversion_component else forw_vz, idx=3)
                if self.model.get_requires_grad("delta"):
                    self.process_gradient(self.model.delta, forw=forw_p if "pressure" in self.inversion_component else forw_vz, idx=4)

            # update model parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # constrain the velocity model
            self.model.forward()

            # cache results
            if self.cache_result:
                self.save_model_and_gradients(epoch_id=i,loss_epoch=loss_epoch)   
                         
            pbar_epoch.set_description("Iter:{},Loss:{:.4}".format(i+1,loss_epoch))