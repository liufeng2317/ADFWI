from typing import Optional,Dict
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from ADFWI.model import AbstractModel
from ADFWI.survey import Survey
from ADFWI.utils import numpy2tensor
from .boundary_condition import bc_pml,bc_gerjan,bc_sincos
from .acoustic_kernels import forward_kernel

class AcousticPropagator(torch.nn.Module):
    """Defines the propagator for the isotropic acoustic wave
    equation (stress-velocity form), solved by the finite
    difference method.

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
                 device : Optional[str] = 'cpu',
                 cpu_num: Optional[int] = 1,
                 gpu_num: Optional[int] = 1,
                 dtype  : torch.dtype = torch.float32
                 ):
        super().__init__()
        
        # Validate model and survey types
        if not isinstance(model, AbstractModel):
            raise ValueError("model is not an instance of AbstractModel")

        if not isinstance(survey, Survey):
            raise ValueError("survey is not an instance of Survey")
        
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
                ) -> Dict[str, Tensor]:
        """Forward simulation for selected shots.

        Parameters:
        -----------
        model (Optional[AbstractModel]) : Model to use for simulation, defaults to the instance's model
        shot_index (Optional[int])       : Index of the shot to simulate
        checkpoint_segments (int)        : Number of segments for checkpointing to save memory

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