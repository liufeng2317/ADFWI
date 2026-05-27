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
from ADFWI.fwi.runtime.backend import align_regularization_backend, validate_model_propagator_devices
from ADFWI.fwi.runtime.cache import (
    append_epoch_loss,
    append_model_snapshots,
    append_required_gradient_snapshots,
    should_cache_epoch,
    snapshot_model_parameters,
    tensor_to_numpy,
)
from ADFWI.fwi.runtime.gradient import (
    elastic_gradient_parameter_specs,
    elastic_parameter_names,
    process_named_parameter_gradients,
    process_parameter_gradient,
)
from ADFWI.fwi.runtime.regularization import calculate_model_regularization_loss, calculate_regularization_loss
from ADFWI.fwi.runtime.wavefield import select_elastic_gradient_wavefield
from ADFWI.fwi.iteration.components import elastic_observed_components, normalize_elastic_component_weights
from ADFWI.fwi.iteration.epoch import apply_epoch_update_step
from ADFWI.fwi.iteration.loss import apply_elastic_batch_loss_step
from ADFWI.fwi.iteration.misfit import evaluate_misfit_loss
from ADFWI.fwi.iteration.preparation import (
    build_fwi_data_transform_pipeline,
    build_fwi_transform_context,
    prepare_fwi_loss_pair,
    prepare_loss_pair,
)
from ADFWI.fwi.iteration.progress import finalize_epoch_progress
from ADFWI.fwi.iteration.range import iter_batch_ranges
from ADFWI.fwi.transforms import DataTransformPipeline
from ADFWI.fwi.transforms.amplitude import normalize_waveform
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
        return calculate_regularization_loss(model_param, weight_x, weight_z, regularization_fn)
    
    def calculate_model_regularization_loss(self):
        return calculate_model_regularization_loss(
            self.model,
            elastic_parameter_names(include_anisotropic=isinstance(self.model, AnisotropicElasticModel)),
            self.regularization_weights_x,
            self.regularization_weights_z,
            self.regularization_fn,
        )
    
    # gradient precondition
    def process_gradient(self, parameter, forw, idx=None):
        process_parameter_gradient(
            parameter,
            self.gradient_processor,
            model=self.model,
            propagator=self.propagator,
            forw=forw,
            idx=idx,
            processor_type=GradProcessor,
        )
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
        all_param_names = elastic_parameter_names(include_anisotropic=isinstance(self.model, AnisotropicElasticModel))

        append_epoch_loss(self, loss_epoch)
        if should_cache_epoch(epoch_id, self.cache_result_epoch):
            append_model_snapshots(
                self,
                snapshot_model_parameters(self.model, all_param_names),
                epoch_id,
            )
        
        # save the figure
        self.save_vp_vs_rho_fig(epoch_id,tensor_to_numpy(self.model.vp),
                                         tensor_to_numpy(self.model.vs),
                                         tensor_to_numpy(self.model.rho))
        if isinstance(self.model,AnisotropicElasticModel):
            self.save_eps_delta_gamma_fig(epoch_id,
                                          tensor_to_numpy(self.model.eps),
                                          tensor_to_numpy(self.model.delta),
                                          tensor_to_numpy(self.model.gamma))

        gradient_snapshots = append_required_gradient_snapshots(self, self.model, all_param_names)
        for name, gradient in gradient_snapshots.items():
            self.save_gradient_fig(epoch_id, gradient, model_type=f"grad_{name}")
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
            accumulated_wavefields = {}
            pbar_batch = tqdm(batch_ranges,position=1,leave=False,colour='red',ncols=80)
            for batch_range in pbar_batch:
                # forward simulation and batch loss
                batch_result = apply_elastic_batch_loss_step(
                    epoch_loss_scalar=loss_epoch,
                    accumulated_wavefields=accumulated_wavefields,
                    propagator=self.propagator,
                    batch_range=batch_range,
                    fd_order=fd_order,
                    checkpoint_segments=checkpoint_segments,
                    observed_components=self.obs_components,
                    inversion_components=self.inversion_component,
                    component_weights=self.component_weights,
                    prepare_loss_pair=self._prepare_loss_pair,
                    loss_fn=self.loss_fn,
                    normalization=self.waveform_normalize,
                    cutoff_freq=cutoff_freq,
                    regularization_loss_fn=self.calculate_model_regularization_loss if self.regularization_fn is not None else None,
                    progress_bar=pbar_batch,
                    batch_count=len(batch_ranges),
                    device=self.device,
                )
                loss_epoch = batch_result.epoch_loss_scalar
                accumulated_wavefields = batch_result.accumulated_wavefields
            
            # gradient process
            gradient_wavefield = select_elastic_gradient_wavefield(accumulated_wavefields)
            process_named_parameter_gradients(
                self.model,
                elastic_gradient_parameter_specs(include_anisotropic=isinstance(self.model, AnisotropicElasticModel)),
                self.process_gradient,
                forw=gradient_wavefield,
            )

            # update model parameters
            apply_epoch_update_step(self.optimizer, self.scheduler, self.model)

            # cache results
            finalize_epoch_progress(
                pbar_epoch,
                epoch_id=i,
                loss_epoch=loss_epoch,
                cache_result=self.cache_result,
                cache_callback=self.save_model_and_gradients,
            )
