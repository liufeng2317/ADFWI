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
import torch
import numpy as np
from tqdm import tqdm
from ADFWI.model       import AbstractModel
from ADFWI.propagator  import AcousticPropagator,GradProcessor
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
)
from ADFWI.fwi.runtime.gradient import (
    acoustic_gradient_parameter_specs,
    acoustic_parameter_names,
    process_named_parameter_gradients,
    process_parameter_gradient,
)
from ADFWI.fwi.runtime.regularization import calculate_model_regularization_loss, calculate_regularization_loss
from ADFWI.fwi.iteration.epoch import apply_epoch_update_step
from ADFWI.fwi.iteration.loss import apply_acoustic_batch_loss_step
from ADFWI.fwi.iteration.misfit import evaluate_misfit_loss
from ADFWI.fwi.iteration.preparation import (
    build_fwi_data_transform_pipeline,
    build_fwi_transform_context,
    prepare_fwi_loss_pair,
)
from ADFWI.fwi.iteration.progress import finalize_epoch_progress
from ADFWI.fwi.iteration.range import iter_batch_ranges
from ADFWI.fwi.transforms import DataTransformPipeline
from ADFWI.fwi.transforms.amplitude import normalize_waveform
from ADFWI.fwi.optimizer import NLCG
from ADFWI.utils       import numpy2tensor
from ADFWI.view        import plot_model

    

class AcousticFWI(torch.nn.Module):
    """Acoustic Full waveform inversion class
    """
    def __init__(self,
                 propagator:AcousticPropagator,model:AbstractModel,
                 optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler,
                 loss_fn:Union[Misfit,torch.autograd.Function],
                 obs_data:SeismicData,
                 gradient_processor: Union[GradProcessor,List[GradProcessor]] = None,
                 regularization_fn:Optional[Regularization]                   = None, 
                 regularization_weights_x:Optional[List[Union[float]]]        = None, # vp/rho in x direction
                 regularization_weights_z:Optional[List[Union[float]]]        = None, # vp/rho in z direction
                 waveform_normalize:Optional[bool]                            = True,
                 waveform_mute_late_window:Optional[float]                    = None,
                 waveform_mute_offset:Optional[float]                         = None,
                 data_transform_pipeline:Optional[DataTransformPipeline]       = None,
                 cache_result:Optional[bool]                                  = True,
                 cache_result_epoch:Optional[bool]                            = 1,
                 save_fig_epoch:Optional[int]                                 = -1,
                 save_fig_path:Optional[str]                                  = "",
                ):
        """
        Description:
        --------------
        Acoustic Full Waveform Inversion Class
        
        Parameters:
        --------------
        propagator (AcousticPropagator)                                : The propagator used for simulating acoustic wave propagation.
        model (AbstractModel)                                          : The model class representing the velocity or acoustic property structure.
        optimizer (torch.optim.Optimizer)                              : The optimizer used for parameter optimization (e.g., SGD, Adam).
        scheduler (torch.optim.lr_scheduler)                           : The learning rate scheduler for adjusting the learning rate during training.
        loss_fn (Union[Misfit, torch.autograd.Function])               : The loss function or misfit function used to compute the difference between predicted and observed data.
        obs_data (SeismicData)                                         : The observed seismic data for comparison against the model predictions.
        gradient_processor (Union[GradProcessor, List[GradProcessor]]) : The gradient processor or list of processors for handling gradients, applied to different parameters if specified.
        regularization_fn (Optional[Regularization])                   : The regularization function for model parameters (e.g., for smoothing or penalty terms). Default is None.
        regularization_weights_x (Optional[List[Union[float]]])        : Regularization weights for the x direction (e.g., vp/rho regularization). Default is [0, 0].
        regularization_weights_z (Optional[List[Union[float]]])        : Regularization weights for the z direction (e.g., vp/rho regularization). Default is [0, 0].
        waveform_normalize (Optional[bool])                            : Whether to normalize waveforms. In bv1.2 the default path implements this through the internal transform pipeline. Set False when a custom pipeline already normalizes data.
        waveform_mute_late_window:Optional[float]                      : Clipping data after picking the first arrival with the given window size.
        waveform_mute_offset:Optional[float]                           : Clipping data larger than the given offset threshold.
        data_transform_pipeline (Optional[DataTransformPipeline])       : Optional extra synthetic/observed waveform transform pipeline. It is appended after legacy-compatible mute, low-pass, and data-mask transforms; transforms must operate on same-shape synthetic/observed tensors.
        cache_result (Optional[bool])                                  : Whether to cache intermediate inversion results for later use. Default is True.
        save_fig_epoch (Optional[int])                                 : The interval (in epochs) at which to save the inversion result as a figure. Default is -1 (no figure saved).
        save_fig_path (Optional[str])                                  : The path where to save the inversion result figure. Default is an empty string (no path specified).
        """
        super().__init__()
        self.propagator                 = propagator
        self.model                      = model
        self.optimizer                  = optimizer
        self.scheduler                  = scheduler
        self.loss_fn                    = loss_fn
        self.regularization_fn          = regularization_fn
        self.regularization_weights_x   = list(regularization_weights_x) if regularization_weights_x is not None else [0, 0]
        self.regularization_weights_z   = list(regularization_weights_z) if regularization_weights_z is not None else [0, 0]
        self.obs_data                   = obs_data
        self.gradient_processor         = gradient_processor
        self.data_transform_pipeline, self.waveform_normalize = self._configure_data_transform_pipeline(
            data_transform_pipeline, waveform_normalize
        )
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
        self.waveform_mute_late_window  = waveform_mute_late_window 
        self.waveform_mute_offset       = waveform_mute_offset
        
        # observed data
        obs_p   = self.obs_data.data["p"]
        obs_p   = numpy2tensor(obs_p,self.dtype).to(self.device)
        if self.propagator.receiver_masks_obs: # mark the observed data need to be masked or not
            obs_p   = obs_p*self.receiver_masks_3D
        self.data_masks = numpy2tensor(self.obs_data.data_masks).to(self.device) if self.obs_data.data_masks is not None else None
        if self.data_masks is not None: # some of the data are unuseful
            obs_p = obs_p*self.data_masks
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
        
        # result saving
        self.cache_result   = cache_result
        self.cache_result_epoch = cache_result_epoch
        self.iter_vp, self.iter_rho = [],[]
        self.iter_vp_grad, self.iter_rho_grad = [],[]
        self.cache_iter_index = []
        self.iter_loss      = []
        
        # figure saving
        self.save_fig_epoch = save_fig_epoch
        self.save_fig_path  = save_fig_path
    
    def _configure_data_transform_pipeline(self, data_transform_pipeline, waveform_normalize):
        return build_fwi_data_transform_pipeline(data_transform_pipeline, waveform_normalize)


    def _normalize(self, data):
        return normalize_waveform(data)
    
    # misfits calculation
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

    def calculate_loss(self, synthetic_waveform, observed_waveform, normalization, loss_fn, cutoff_freq=None, propagator_dt=None,shot_index=None, apply_transforms=True):
        """
        Generalized function to calculate misfit loss for a given component.
        Real-Data Processing
            (1) first arrival picking
            (2) mute data by first arrival & giving window
            (3) mute data by offset
            (4) low-pass filter
            (5) data normalize
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
            synthetic_waveform = self._normalize(synthetic_waveform)
            observed_waveform  = self._normalize(observed_waveform)
        
        return evaluate_misfit_loss(
            loss_fn,
            synthetic_waveform,
            observed_waveform,
            function_fallback="apply",
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
            acoustic_parameter_names(),
            self.regularization_weights_x,
            self.regularization_weights_z,
            self.regularization_fn,
        )

    # gradient precondition
    def process_gradient(self, parameter,forw,idx=None):
        process_parameter_gradient(
            parameter,
            self.gradient_processor,
            model=self.model,
            propagator=self.propagator,
            forw=forw,
            idx=idx,
            processor_type=GradProcessor,
        )

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
    
    def save_model_and_gradients(self, epoch_id, loss_epoch):
        model_snapshots = snapshot_model_parameters(self.model, ["vp", "rho"])
        if should_cache_epoch(epoch_id, self.cache_result_epoch):
            append_model_snapshots(self, model_snapshots, epoch_id)
        append_epoch_loss(self, loss_epoch)
        temp_vp = model_snapshots["vp"]
        temp_rho = model_snapshots["rho"]
        self.save_figure(epoch_id, temp_vp, model_type="vp")
        self.save_figure(epoch_id, temp_rho, model_type="rho")

        gradient_snapshots = append_required_gradient_snapshots(self, self.model, ["vp", "rho"])
        for name, gradient in gradient_snapshots.items():
            self.save_figure(epoch_id, gradient, model_type=f"grad_{name}")
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
        iteration (int)                     : The maximum iteration number in the inversion process.
        batch_size (Optional[int])          : The number of shots (data samples) in each batch. Default is None, meaning use all available shots.
        checkpoint_segments (Optional[int]) : The number of segments into which the time series should be divided for memory efficiency. Default is 1, which means no segmentation.
        start_iter (int)                    : The starting iteration for the optimization process (e.g., for optimizers like Adam/AdamW, and learning rate schedulers like step_lr). Default is 0.
        cutoff_freq (Optional[float])       : The cutoff frequency for low-pass filtering, if specified. Default is None (no filtering applied).
        """
        if isinstance(self.optimizer,torch.optim.LBFGS) or isinstance(self.optimizer,NLCG):
            return self.forward_closure(iteration=iteration,batch_size=batch_size,checkpoint_segments=checkpoint_segments,start_iter=start_iter,cutoff_freq=cutoff_freq)

        n_shots = self.propagator.src_n
        batch_ranges = list(iter_batch_ranges(n_shots, batch_size))

        # epoch
        pbar_epoch = tqdm(range(start_iter,start_iter+iteration),position=0,leave=False,colour='green',ncols=80)
        for i in pbar_epoch:
            # batch
            self.optimizer.zero_grad()
            loss_batch = 0
            forw = None
            pbar_batch = tqdm(batch_ranges,position=1,leave=False,colour='red',ncols=80)
            for batch_range in pbar_batch:
                # forward simulation and batch loss
                batch_result = apply_acoustic_batch_loss_step(
                    epoch_loss_scalar=loss_batch,
                    accumulated_wavefield=forw,
                    propagator=self.propagator,
                    batch_range=batch_range,
                    checkpoint_segments=checkpoint_segments,
                    observed_pressure=self.obs_p,
                    prepare_loss_pair=self._prepare_loss_pair,
                    loss_fn=self.loss_fn,
                    normalization=self.waveform_normalize,
                    cutoff_freq=cutoff_freq,
                    regularization_loss_fn=self.calculate_model_regularization_loss if self.regularization_fn is not None else None,
                    progress_bar=pbar_batch,
                    batch_count=len(batch_ranges),
                    device=self.device,
                )
                loss_batch = batch_result.epoch_loss_scalar
                forw = batch_result.accumulated_wavefield
            
            # gradient process
            process_named_parameter_gradients(
                self.model,
                acoustic_gradient_parameter_specs(),
                self.process_gradient,
                forw=forw,
            )
        
            apply_epoch_update_step(self.optimizer, self.scheduler, self.model)
            
            finalize_epoch_progress(
                pbar_epoch,
                epoch_id=i,
                loss_epoch=loss_batch,
                cache_result=self.cache_result,
                cache_callback=self.save_model_and_gradients,
            )
            self.true_epoch = 0
    
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
        batch_ranges = list(iter_batch_ranges(n_shots, batch_size))
                
        # epoch
        pbar_epoch = tqdm(range(start_iter,start_iter+iteration),position=0,leave=False,colour='green',ncols=80)
        self.true_epoch = 0
        self.forw = None
        for i in pbar_epoch:
            def closure():
                # batch (for the clouser we hold 1 batch)
                self.optimizer.zero_grad()
                loss_batch = 0
                self.forw = None
                pbar_batch = tqdm(batch_ranges,position=1,leave=False,colour='red',ncols=80)
                for batch_range in pbar_batch:
                    # forward simulation and batch loss
                    batch_result = apply_acoustic_batch_loss_step(
                        epoch_loss_scalar=loss_batch,
                        accumulated_wavefield=self.forw,
                        propagator=self.propagator,
                        batch_range=batch_range,
                        checkpoint_segments=checkpoint_segments,
                        observed_pressure=self.obs_p,
                        prepare_loss_pair=self._prepare_loss_pair,
                        loss_fn=self.loss_fn,
                        normalization=self.waveform_normalize,
                        cutoff_freq=cutoff_freq,
                        regularization_loss_fn=self.calculate_model_regularization_loss if self.regularization_fn is not None else None,
                        progress_bar=pbar_batch,
                        batch_count=len(batch_ranges),
                        device=self.device,
                    )
                    loss_batch = batch_result.epoch_loss_scalar
                    self.forw = batch_result.accumulated_wavefield
                self.true_epoch = self.true_epoch + 1
                # gradient process
                process_named_parameter_gradients(
                    self.model,
                    acoustic_gradient_parameter_specs(),
                    self.process_gradient,
                    forw=self.forw,
                )
                return loss_batch
            
            loss_batch = apply_epoch_update_step(
                self.optimizer,
                self.scheduler,
                self.model,
                closure=closure,
            )
            
            # save the result
            finalize_epoch_progress(
                pbar_epoch,
                epoch_id=i,
                loss_epoch=loss_batch,
                cache_result=self.cache_result,
                cache_callback=self.save_model_and_gradients,
            )
