"""Elastic propagator wrapper.

`ElasticPropagator` adapts an elastic model and survey into backend tensors,
builds boundary tensors, and dispatches to the elastic finite-difference
kernel. Numerical wavefield updates live in `elastic_kernels.py`.
"""

from typing import Optional
import torch
from ADFWI.model import AbstractModel
from ADFWI.survey import Survey
from .boundary_condition import bc_pml_xz,bc_gerjan,bc_sincos
from .elastic_kernels import forward_kernel
from ADFWI.utils import numpy2tensor
from ADFWI.backends import get_backend

class ElasticPropagator(torch.nn.Module):
    """Elastic finite-difference propagator interface.

    This wrapper owns model/survey/backend adaptation and kernel dispatch. It
    does not own FWI loss construction, waveform transforms, or finite-
    difference formulas.

    Parameters:
    -----------
    model: IsotropicElasticModel
        The model object
    survey: Survey
        The survey object
    """
    def __init__(self,
                model:AbstractModel,
                survey:Survey,
                device:Optional[str]    = None,
                cpu_num:Optional[int]   = 1,
                gpu_num:Optional[int]   = 1,
                dtype                   = None
                ):
        super().__init__()
        
        if not isinstance(model, AbstractModel):
            raise ValueError("model is not AbstractModel")

        if not isinstance(survey, Survey):
            raise ValueError("survey is not Survey")
        
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
        self.src_x          = numpy2tensor(self.src_loc[:,0],torch.long).to(self.device)
        self.src_z          = numpy2tensor(self.src_loc[:,1],torch.long).to(self.device)
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
        
    def boundary_condition(self):
        if self.abc_type == "PML":
            bcx,bcz = bc_pml_xz(self.nx,self.nz,self.dx,self.dz,pml=self.nabc,
                                vmax=self.model.vp.cpu().detach().numpy().max(),
                                free_surface=self.free_surface)
            self.bcx = numpy2tensor(bcx,self.dtype).to(self.device)
            self.bcz = numpy2tensor(bcz,self.dtype).to(self.device)
        elif self.abc_type=='gerjan':
            damp = bc_gerjan(self.nx,self.nz,self.dx,self.dz,pml=self.nabc,alpha=self.model.abc_jerjan_alpha,
                            free_surface=self.free_surface)
            self.damp = numpy2tensor(damp,self.dtype).to(self.device)
        else:
            damp = bc_sincos(self.nx,self.nz,self.dx,self.dz,pml=self.nabc,
                             free_surface=self.free_surface)
            self.damp = numpy2tensor(damp,self.dtype).to(self.device)
        return 
    
    def forward(self,
                model:Optional[AbstractModel] = None,
                shot_index=None,
                fd_order = 4,
                checkpoint_segments=1):
        # calculate the thomson/lame and elastic moduli parameters
        model = self.model if model is None else model
        model.forward()
        
        # foward simulation for select shots
        src_x = self.src_x[shot_index] if shot_index is not None else self.src_x
        src_z = self.src_z[shot_index] if shot_index is not None else self.src_z
        src_n = len(src_x)
        wavelet = self.wavelet[shot_index] if shot_index is not None else self.wavelet
        moment_tensor = self.moment_tensor[shot_index] if shot_index is not None else self.moment_tensor
        
        record_waveform = forward_kernel(
            self.nx,self.nz,self.dx,self.dz,self.nt,self.dt,
            self.nabc,self.free_surface,
            src_x,src_z,src_n,wavelet,moment_tensor,
            self.rcv_x,self.rcv_z,self.rcv_n,
            self.abc_type,self.bcx,self.bcz,self.damp,
            model.lamu,model.lam,model.bx,model.bz,model.CC,
            fd_order=fd_order,n_segments=checkpoint_segments,
            device=self.device,dtype=self.dtype
        )
        return record_waveform
