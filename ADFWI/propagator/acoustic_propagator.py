"""Acoustic propagator wrapper.

`AcousticPropagator` adapts an acoustic model and survey into backend tensors,
builds boundary damping tensors, and dispatches to the acoustic finite-
difference kernel. Numerical wavefield updates live in `acoustic_kernels.py`.
"""

from typing import Optional,Dict
import torch
from torch import Tensor
from ADFWI.model import AbstractModel
from ADFWI.survey import Survey
from ADFWI.utils import numpy2tensor
from ADFWI.backends import get_backend
from .boundary_condition import bc_pml,bc_gerjan,bc_sincos
from .acoustic_custom_kernels import custom_chunk_forward_kernel
from .acoustic_kernels import forward_kernel

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
                use_custom_chunk_backward: bool = False,
                ) -> Dict[str, Tensor]:
        """Forward simulation for selected shots.

        Parameters:
        -----------
        model (Optional[AbstractModel]) : Model to use for simulation, defaults to the instance's model
        shot_index (Optional[int])       : Index of the shot to simulate
        checkpoint_segments (int)        : Number of segments for checkpointing to save memory in the default path
        save_forward_wavefield (bool)    : Whether to accumulate detached forward wavefield summaries
        use_custom_chunk_backward (bool) : Expert opt-in high-memory custom-chunk backward path. This improves backward speed on measured acoustic FWI cases, but it is not PyTorch checkpoint rematerialization and does not preserve checkpoint memory savings.

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

        if use_custom_chunk_backward and save_forward_wavefield:
            raise ValueError("use_custom_chunk_backward=True requires save_forward_wavefield=False")

        kernel = custom_chunk_forward_kernel if use_custom_chunk_backward else forward_kernel
        
        record_waveform = kernel(
            self.nx,self.nz,self.dx,self.dz,self.nt,self.dt,
            self.nabc,self.free_surface,
            src_x,src_z,src_n,wavelet,
            self.rcv_x,self.rcv_z,self.rcv_n,
            self.damp,
            model.vp,model.rho,
            checkpoint_segments=checkpoint_segments,
            save_forward_wavefield=save_forward_wavefield,
            device=self.device,dtype=self.dtype
        )
        return record_waveform
