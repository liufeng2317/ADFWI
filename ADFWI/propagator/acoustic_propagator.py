from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from ADFWI.model import AbstractModel
from ADFWI.survey import Survey
from ADFWI.utils import numpy2tensor
from .boundary_condition import bc_pml,bc_gerjan,bc_sincos
from .acoustic_kernels import forward_kernel

class AcousticPropagator(torch.nn.Module):
    """The class of defining the propagator for the isotropic acoustic wave
    equation (stress-velocity form), which is solved by the finite
    difference method.

    Parameters:
    -----------
    model (AbstractModel)   : The model object
    survey (Survey)         :The survey object
    """
    def __init__(self,
                model:AbstractModel,survey:Survey,
                device:Optional[str]    = 'cpu',
                cpu_num:Optional[int]   = 1,
                gpu_num:Optional[int]   = 1,
                dtype                   = torch.float32
                ):
        super().__init__()
        
        if not isinstance(model, AbstractModel):
            raise ValueError("model is not AbstractModel")

        if not isinstance(survey, Survey):
            raise ValueError("survey is not Survey")
        
        # ---------------------------------------------------------------
        # set the model and survey
        # ---------------------------------------------------------------
        self.model          = model
        self.survey         = survey
        self.device         = device
        self.dtype          = dtype
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
        self.src_x          = self.src_loc[:,0]
        self.src_z          = self.src_loc[:,1]
        self.src_n          = self.source.num
        self.wavelet        = numpy2tensor(self.source.get_wavelet(),self.dtype).to(self.device)
        self.moment_tensor  = numpy2tensor(self.source.get_moment_tensor(),self.dtype).to(self.device)
        
        # ---------------------------------------------------------------
        # parameters for receiver
        # ---------------------------------------------------------------
        self.receiver       = self.survey.receiver
        self.rcv_loc        = self.receiver.get_loc()
        self.rcv_x          = self.rcv_loc[:,0]
        self.rcv_z          = self.rcv_loc[:,1]
        self.rcv_n          = self.receiver.num
        
        
    def boundary_condition(self,vmax=None):
        # for the acoustic propagator, we set free_surface = False alwalys
        if self.abc_type.lower() in ["pml"]:
            if vmax is not None:
                damp = bc_pml(self.nx,self.nz,self.dx,self.dz,pml=self.nabc,
                                vmax=vmax,
                                free_surface=False)
            else:
                damp = bc_pml(self.nx,self.nz,self.dx,self.dz,pml=self.nabc,
                                vmax=self.model.vp.cpu().detach().numpy().max(),
                                free_surface=False)
        elif self.abc_type.lower() in ['gerjan']:
            damp = bc_gerjan(self.nx,self.nz,self.dx,self.dz,pml=self.nabc,alpha=self.model.abc_jerjan_alpha,
                            free_surface=False)
        else:
            damp = bc_sincos(self.nx,self.nz,self.dx,self.dz,pml=self.nabc,
                             free_surface=False)
        self.damp = numpy2tensor(damp,self.dtype).to(self.device)
        return 
    
    def forward(self,
                model:Optional[AbstractModel]   = None,
                shot_index                      = None,
                checkpoint_segments             = 1,
                ):
        # calculate the thomson/lame and elastic moduli parameters
        model = self.model if model is None else model
        model.forward()
        
        # foward simulation for select shots
        src_x = self.src_x[shot_index] if shot_index is not None else self.src_x
        src_z = self.src_z[shot_index] if shot_index is not None else self.src_z
        src_n = len(src_x)
        wavelet = self.wavelet[shot_index] if shot_index is not None else self.wavelet
        
        record_waveform = forward_kernel(
            self.nx,self.nz,self.dx,self.dz,self.nt,self.dt,
            self.nabc,self.free_surface,
            src_x,src_z,src_n,wavelet,
            self.rcv_x,self.rcv_z,self.rcv_n,
            self.damp,
            model.vp,model.rho,
            checkpoint_segments=checkpoint_segments,
            device=self.device,dtype=self.dtype
        )
        return record_waveform