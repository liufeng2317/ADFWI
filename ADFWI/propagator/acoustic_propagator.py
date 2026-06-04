"""Acoustic propagator wrapper.

`AcousticPropagator` adapts an acoustic model and survey into backend tensors,
builds boundary damping tensors, and dispatches to the acoustic finite-
difference kernel. Numerical wavefield updates live in `acoustic_kernels.py`.
"""

from typing import Dict, Optional

import torch
from torch import Tensor

from ADFWI.backends import get_backend
from ADFWI.model import AbstractModel
from ADFWI.survey import Survey
from ADFWI.utils import numpy2tensor

from .acoustic_custom_kernels import rematerialized_pressure_custom_chunk_forward_kernel
from .acoustic_kernels import forward_kernel
from .boundary_condition import bc_gerjan, bc_pml, bc_sincos


CUSTOM_STRATEGY_REMAT_PRESSURE_STRIDE2 = "remat_pressure_stride2"
SUPPORTED_CUSTOM_CHUNK_STRATEGIES = {
    None,
    CUSTOM_STRATEGY_REMAT_PRESSURE_STRIDE2,
}


def _resolve_custom_chunk_strategy(custom_chunk_strategy):
    if custom_chunk_strategy not in SUPPORTED_CUSTOM_CHUNK_STRATEGIES:
        raise ValueError(
            "custom_chunk_strategy must be None, "
            f"or '{CUSTOM_STRATEGY_REMAT_PRESSURE_STRIDE2}'"
        )
    return custom_chunk_strategy

class AcousticPropagator(torch.nn.Module):
    """Isotropic acoustic finite-difference propagator interface.

    This wrapper owns model/survey/backend adaptation and kernel dispatch. It
    does not own FWI loss construction, waveform transforms, or finite-
    difference formulas.

    Parameters:
    -----------
    model (AbstractModel)   : The model object
    survey (Survey)         : The survey object
    device (Optional[str])  : Device type, default is 'cpu'
    cpu_num (Optional[int]) : Number of CPU threads, default is 1
    gpu_num (Optional[int]) : Number of GPU devices, default is 1
    dtype (torch.dtype)     : Data type for tensors, default is torch.float32
    """
    def __init__(self,
                 model  : AbstractModel,
                 survey : Survey,
                 device : Optional[str] = None,
                 cpu_num: Optional[int] = 1,
                 gpu_num: Optional[int] = 1,
                 dtype  : Optional[torch.dtype] = None
                 ):
        super().__init__()
        
        # Validate model and survey types
        if not isinstance(model, AbstractModel):
            raise ValueError("model is not an instance of AbstractModel")

        if not isinstance(survey, Survey):
            raise ValueError("survey is not an instance of Survey")
        
        backend = get_backend(device=model.device if device is None else device, dtype=model.dtype if dtype is None else dtype)

        # ---------------------------------------------------------------
        # set the model and survey
        # ---------------------------------------------------------------
        self.model          = model
        self.survey         = survey
        self.device         = backend.device
        self.dtype          = backend.dtype
        self.cpu_num        = cpu_num
        self.gpu_num        = gpu_num
        
        # ---------------------------------------------------------------
        # parse parameters for model
        # ---------------------------------------------------------------
        self.ox, self.oz    = model.ox,model.oz
        self.dx, self.dz    = model.dx,model.dz
        self.nx, self.nz    = model.nx,model.nz
        self.nt             = survey.source.nt
        self.dt             = survey.source.dt
        self.f0             = survey.source.f0
        
        # ---------------------------------------------------------------
        # set the boundary: [top, bottom, left, right]
        # ---------------------------------------------------------------
        self.abc_type       = model.abc_type
        self.nabc           = model.nabc
        self.free_surface   = model.free_surface
        self.bcx,self.bcz,self.damp   = None,None,None
        self.boundary_condition()
        
        # ---------------------------------------------------------------
        # parameters for source
        # ---------------------------------------------------------------
        self.source         = self.survey.source
        self.src_loc        = self.source.get_loc()
        self.src_x          = numpy2tensor(self.src_loc[...,0],torch.long).to(self.device)
        self.src_z          = numpy2tensor(self.src_loc[...,1],torch.long).to(self.device)
        self.src_n          = self.source.num
        self.wavelet        = numpy2tensor(self.source.get_wavelet(),self.dtype).to(self.device)
        self.moment_tensor  = numpy2tensor(self.source.get_moment_tensor(),self.dtype).to(self.device)
        
        # ---------------------------------------------------------------
        # parameters for receiver
        # ---------------------------------------------------------------
        self.receiver       = self.survey.receiver
        self.rcv_loc        = self.receiver.get_loc()
        self.rcv_x          = numpy2tensor(self.rcv_loc[:,0],torch.long).to(self.device)
        self.rcv_z          = numpy2tensor(self.rcv_loc[:,1],torch.long).to(self.device)
        self.rcv_n          = self.receiver.num
        
        self.receiver_masks     = self.survey.receiver_masks
        self.receiver_masks_obs = self.survey.receiver_masks_obs
        
    def boundary_condition(self, vmax=None):
        """Set boundary conditions based on the specified ABC type."""
        if self.abc_type.lower() == "pml":
            if vmax is not None:
                damp = bc_pml(self.nx, self.nz, self.dx, self.dz, pml=self.nabc, vmax=vmax, free_surface=False)
            else:
                damp = bc_pml(self.nx, self.nz, self.dx, self.dz, pml=self.nabc,
                               vmax=self.model.vp.cpu().detach().numpy().max(),
                               free_surface=False)
        elif self.abc_type.lower() == 'gerjan':
            damp = bc_gerjan(self.nx, self.nz, self.dx, self.dz, pml=self.nabc, alpha=self.model.abc_jerjan_alpha,
                             free_surface=False)
        else:
            damp = bc_sincos(self.nx, self.nz, self.dx, self.dz, pml=self.nabc,
                             free_surface=False)

        self.damp = numpy2tensor(damp, self.dtype).to(self.device) 
    
    def forward(self,
                model: Optional[AbstractModel] = None,
                shot_index: Optional[int] = None,
                checkpoint_segments: int = 1,
                save_forward_wavefield: bool = True,
                custom_chunk_strategy: Optional[str] = None,
                pressure_only: bool = False,
                ) -> Dict[str, Tensor]:
        """Forward simulation for selected shots.

        Parameters:
        -----------
        model (Optional[AbstractModel]) : Model to use for simulation, defaults to the instance's model
        shot_index (Optional[int])       : Index of the shot to simulate
        checkpoint_segments (int)        : Number of segments for checkpointing to save memory in the default path
        save_forward_wavefield (bool)    : Whether to accumulate detached forward wavefield summaries
        custom_chunk_strategy (Optional[str]): Expert opt-in custom backward strategy. Supported values are None and "remat_pressure_stride2".
        pressure_only (bool)             : Opt-in acoustic FWI path that records only pressure outputs. Default keeps full p/u/w outputs.

        Returns:
        --------
        record_waveform (dict) : Dictionary containing recorded waveforms
        """
        # calculate the thomson/lame and elastic moduli parameters
        model = self.model if model is None else model
        model.forward()
        
        # foward simulation for select shots
        src_x = self.src_x[shot_index] if shot_index is not None else self.src_x
        src_z = self.src_z[shot_index] if shot_index is not None else self.src_z
        src_n = len(src_x)
        wavelet = self.wavelet[shot_index] if shot_index is not None else self.wavelet

        custom_chunk_strategy = _resolve_custom_chunk_strategy(custom_chunk_strategy)
        if custom_chunk_strategy is not None and save_forward_wavefield:
            raise ValueError(
                "custom_chunk_strategy requires save_forward_wavefield=False, got save_forward_wavefield=True"
            )
        if custom_chunk_strategy == CUSTOM_STRATEGY_REMAT_PRESSURE_STRIDE2 and not pressure_only:
            raise ValueError(
                f"custom_chunk_strategy='{CUSTOM_STRATEGY_REMAT_PRESSURE_STRIDE2}' requires pressure_only=True"
            )

        if custom_chunk_strategy == CUSTOM_STRATEGY_REMAT_PRESSURE_STRIDE2:
            kernel = rematerialized_pressure_custom_chunk_forward_kernel
        else:
            kernel = forward_kernel
        
        kernel_kwargs = {
            "checkpoint_segments": checkpoint_segments,
            "save_forward_wavefield": save_forward_wavefield,
            "device": self.device,
            "dtype": self.dtype,
        }
        if custom_chunk_strategy is None:
            kernel_kwargs["pressure_only"] = pressure_only
        elif custom_chunk_strategy == CUSTOM_STRATEGY_REMAT_PRESSURE_STRIDE2:
            kernel_kwargs["divergence_cache_stride"] = 2
            kernel_kwargs["divergence_cache_components"] = "p,u,w"

        record_waveform = kernel(
            self.nx,self.nz,self.dx,self.dz,self.nt,self.dt,
            self.nabc,self.free_surface,
            src_x,src_z,src_n,wavelet,
            self.rcv_x,self.rcv_z,self.rcv_n,
            self.damp,
            model.vp,model.rho,
            **kernel_kwargs,
        )
        return record_waveform
