from ADFWI.model import AbstractModel
from ADFWI.utils import numpy2tensor
from ADFWI.view import plot_vp_rho,plot_model
from typing import Optional,Tuple,Union
import torch
from torch import Tensor
import numpy as np
from torchinfo import summary

class DIP_AcousticModel(AbstractModel):
    """
    Acoustic Velocity model with deep parameterization of vp or rho.

    Parameters
    ----------
    ox : float
        Not used. The origin coordinates of the model in the x-direction (meters).
    oz : float
        Not used. The origin coordinates of the model in the z-direction (meters).
    nx : int
        The number of grid points in the x-direction.
    nz : int
        The number of grid points in the z-direction.
    dx : float
        The grid spacing in the x-direction (meters).
    dz : float
        The grid spacing in the z-direction (meters).
    DIP_model_vp : Optional[torch.nn.Module]
        Reparameterized vp using a deep neural network, by default None.
    DIP_model_rho : Optional[torch.nn.Module]
        Reparameterized rho using a deep neural network, by default None.
    reparameterization_strategy : str
        Reparameterization strategy ('vel' for generating velocity model, 'vel_diff' for generating velocity variation).
    vp_init : Optional[torch.Tensor]
        The initial vp model (must be provided when using the 'vel_diff' strategy).
    rho_init : Optional[torch.Tensor]
        The initial rho model (must be provided when using the 'vel_diff' strategy).
    vp_bound : Optional[Tuple[float, float]], default=None
        The lower and upper bounds for the P-wave velocity model.
    rho_bound : Optional[Tuple[float, float]], default=None
        The lower and upper bounds for the density model.
    auto_update_rho : Optional[bool], default=True
        Whether to automatically update the density model during inversion.
    auto_update_vp : Optional[bool], default=False
        Whether to automatically update the P-wave velocity model during inversion.
    water_layer_mask : Optional[Union[np.array, Tensor]], default=None
        A mask for the water layer (not updated), if applicable.
    free_surface : Optional[bool], default=False
        Indicates whether a free surface is present in the model.
    abc_type : Optional[str], default='PML'
        The type of absorbing boundary condition used in the model. Options: 'PML' or 'Jerjan'.
    abc_jerjan_alpha : Optional[float], default=0.0053
        The attenuation factor for the Jerjan boundary condition.
    nabc : Optional[int], default=20
        The number of grid cells dedicated to the absorbing boundary.
    device : str, default='cpu'
        The device on which to run the model ('cpu' or 'cuda').
    dtype : torch.dtype, default=torch.float32
        The data type for PyTorch tensors.
    """

    def __init__(self,
                ox:float,oz:float,
                nx:int  ,nz:int,
                dx:float,dz:float,
                DIP_model_vp                                     = None,     # deep image prior models
                DIP_model_rho                                    = None,
                reparameterization_strategy                      = "vel",       # vel/vel_diff
                vp_init:Optional[Union[np.array,Tensor]]         = None,     # initial model parameter
                rho_init:Optional[Union[np.array,Tensor]]        = None,
                vp_bound    : Optional[Tuple[float, float]]      = None,     # model parameter's boundary
                rho_bound   : Optional[Tuple[float, float]]      = None,
                water_layer_mask:Optional[Union[np.array,Tensor]]= None,
                auto_update_rho:Optional[bool]                   = True,
                auto_update_vp :Optional[bool]                   = False,
                free_surface: Optional[bool]                     = False,
                abc_type    : Optional[str]                      = 'PML',
                abc_jerjan_alpha:Optional[float]                 = 0.0053,
                nabc:Optional[int]                               = 20,
                device                                           = 'cpu',
                dtype                                            = torch.float32
                ):
        # initialize the common model parameters
        super().__init__(ox,oz,nx,nz,dx,dz,free_surface,abc_type,abc_jerjan_alpha,nabc,device,dtype)
        self.reparameterization_strategy = reparameterization_strategy
        
        # update rho/vp using the empirical function
        self.auto_update_rho = auto_update_rho
        self.auto_update_vp  = auto_update_vp
        
        # gradient mask
        if water_layer_mask is not None:
            self.water_layer_mask = numpy2tensor(water_layer_mask,dtype=torch.bool).to(device)
        else:
            self.water_layer_mask = None
        
        # Neural networks
        self.DIP_model_vp   = DIP_model_vp
        self.DIP_model_rho  = DIP_model_rho
        
        # initialize the model parameters
        self.pars       = ["vp","rho"]
        self.vp_init    = torch.zeros((nz,nx),dtype=dtype).to(device) if  vp_init is None else numpy2tensor(vp_init,dtype=dtype).to(device)
        self.rho_init   = torch.zeros((nz,nx),dtype=dtype).to(device) if rho_init is None else numpy2tensor(rho_init,dtype=dtype).to(device)
        self.vp         = self.vp_init.clone()
        self.rho        = self.rho_init.clone()
        self._parameterization()
        

        # set model bounds
        self.lower_bound["vp"]  =  vp_bound[0]  if vp_bound  is not None else None
        self.lower_bound["rho"] = rho_bound[0]  if rho_bound is not None else None
        self.upper_bound["vp"]  =  vp_bound[1]  if vp_bound  is not None else None
        self.upper_bound["rho"] = rho_bound[1]  if rho_bound is not None else None
        
        # check the input model
        self._check_bounds()
        self.check_dims()
        
    def get_requires_grad(self, par: str):
        if par not in self.pars:
            raise ValueError("Parameter {} not in model".format(par))
        if par == "vp":
            return self.DIP_model_vp is not None
        if par == "rho":
            return self.DIP_model_rho is not None

    def get_model(self, par: str):
        if par not in ["vp","rho"]:
            raise "Error input parametrs"
        elif par == "vp":
            vp  = self.vp.cpu().detach().numpy()
            return vp
        elif par == "rho":
            rho = self.rho.cpu().detach().numpy()
            return rho

    def get_bound(self, par: str):
        if par not in ["vp","rho"]:
            raise "Error input parameters"
        else:
            m_min = self.lower_bound[par]
            m_max = self.upper_bound[par]        
        return [m_min,m_max]
    
    def __repr__(self):
        info = f"   Model with parameters {self.pars}:\n"
        info += f"  Model orig: ox = {self.ox:6.2f}, oz = {self.oz:6.2f} m\n"
        info += f"  Model grid: dx = {self.dx:6.2f}, dz = {self.dz:6.2f} m\n"
        info += f"  Model dims: nx = {self.nx:6d}, nz = {self.nz:6d}\n"
        info += f"  Model size: {self.nx * self.nz * len(self.pars)}\n"
        info += f"  Free surface: {self.free_surface}\n"
        info += f"  Absorbing layers: {self.nabc}\n"
        info += f"  NN structure\n"
        if self.DIP_model_vp is not None:
            info += str(summary(self.DIP_model_vp,device=self.device))
        if self.DIP_model_rho is not None:
            info += str(summary(self.DIP_model_rho,device=self.device))
        return info
    
    def set_rho_using_empirical_function(self):
        """approximate rho via empirical relations with vp
        """
        vp          = self.vp.cpu().detach().numpy()
        rho         = self.rho.cpu().detach().numpy()
        rho_emprical= np.power(vp, 0.25) * 310
        if self.water_layer_mask is not None:
            grad_mask = self.water_layer_mask.cpu().detach().numpy()
            rho_emprical[grad_mask] = rho[grad_mask]
        self.rho    = numpy2tensor(rho_emprical,self.dtype).to(self.device)
        return
    
    def set_vp_using_empirical_function(self):
        """approximate vp via empirical relations with rho
        """
        rho         = self.rho.cpu().detach().numpy()
        vp          = self.vp.cpu().detach().numpy()
        vp_empirical= np.power(rho / 310, 4)
        if self.water_layer_mask is not None:
            grad_mask = self.water_layer_mask.cpu().detach().numpy()
            vp_empirical[grad_mask] = vp[grad_mask]
        self.vp     = numpy2tensor(vp_empirical,self.dtype).to(self.device)
        return   
    
    def _parameterization(self,*args,**kw_args):
        """setting variable and gradients
        """
        if self.DIP_model_vp is not None:
            if self.reparameterization_strategy == "vel":
                self.vp     = self.DIP_model_vp(*args,**kw_args)
            elif self.reparameterization_strategy == "vel_diff":
                self.vp     = self.vp_init + self.DIP_model_vp(*args,**kw_args)
        elif self.auto_update_vp:
            self.set_vp_using_empirical_function()
            
        if self.DIP_model_rho is not None:
            if self.reparameterization_strategy == "vel":
                self.rho    = self.DIP_model_rho(*args,**kw_args)
            elif self.reparameterization_strategy == "vel_diff":
                self.rho    = self.rho_init + self.DIP_model_rho(*args,**kw_args)
        elif self.auto_update_rho:
            self.set_rho_using_empirical_function()
        return
    
    def _plot_vp_rho(self,**kwargs):
        """plot velocity model
        """
        plot_vp_rho(self.vp,self.rho, dx=self.dx,dz=self.dz,**kwargs)
        return
    
    def _plot(self,var,**kwargs):
        """plot single velocity model
        """
        model_data = self.get_model(var)
        plot_model(model_data,title=var,**kwargs)
        return
    
    def clip_params(self,par):
        """Clip the model parameters to the given bounds
        """
        if self.get_requires_grad(par):
            if self.lower_bound[par] is not None and self.upper_bound[par] is not None:
                # Retrieve the model parameter
                m = getattr(self, par)
                min_value = self.lower_bound[par]
                max_value = self.upper_bound[par]
                # Create a temporary copy for masking purposes
                m_temp = m.clone()  # Use .clone() instead of .copy() to avoid issues with gradients

                # Clip the values of the parameter using in-place modification with .data
                m.data.clamp_(min_value, max_value)
                
                # Apply the water layer mask if it is not None, using in-place modification
                if self.water_layer_mask is not None:
                    m.data = torch.where(self.water_layer_mask.contiguous(), m_temp.data, m.data)
        return

    def forward(self,*args,**kwargs):
        """Forward method of the elastic model class
        """
        self._parameterization()
        
        self.clip_params("vp")
        self.clip_params("rho")
        return