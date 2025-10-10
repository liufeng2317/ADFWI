'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-26 19:42:24
* LastEditors: LiuFeng
* LastEditTime: 2024-05-13 23:16:49
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

from ADFWI.model import AbstractModel
from ADFWI.utils import numpy2tensor
from ADFWI.model.parameters import (thomsen_init,elastic_moduli_init,
                         vs_vp_to_Lame,thomsen_to_elastic_moduli,
                         elastic_moduli_for_isotropic,elastic_moduli_for_TI,
                         parameter_staggered_grid)
from ADFWI.view import (plot_vp_vs_rho,plot_eps_delta_gamma,plot_lam_mu,plot_model)
from typing import Optional,Tuple,Union
import torch
from torch import Tensor
import numpy as np
from torchinfo import summary

class DIP_ElasticModel_vp_vs(AbstractModel):
    """Elastic Velocity model with parameterization of vp and rho
    Parameters:
    --------------
        ox (float),oz(float)        : Origin of the model in x- and z- direction (m)
        nx (int),nz(int)            : NUmber of grid points in x- and z- direction (m)
        dx (float),dz(float)        : Grid size in x- and z- direction (m)
        vp_bound (tuple,Optional)   : Bounds for the P-wave velocity model, default None
        rho_bound (tuple,Optional)  : Bounds for the density model, default None
        vp_grad (bool,Optional)     : Flag for gradient of P-wave velocity model, default is False
        rho_grad (bool,Optional)    : Flag for gradient of density, default is False
        free_surface (bool,Optional): Flag for free surface, default is False
        abc_type (str)              : the type of absorbing boundary conditoin : PML,jerjan and other
        abc_jerjan_alpha (float)    : the attenuation factor for jerjan boundary condition
        nabc (int)                  : Number of absorbing boundary cells, default is 20
        device (str,Optional)       : The runing device
        dtype (dtypes,Optional)     : The dtypes for pytorch variable, default is torch.float32
    """
    def __init__(self,
                ox:float,oz:float,
                nx:int  ,nz:int,
                dx:float,dz:float,
                DIP_model                                        = None,     # deep image prior models
                reparameterization_strategy                      = "vel",       # vel/vel_diff
                vp_init:Optional[Union[np.array,Tensor]]         = None,     # initial model parameter
                vs_init:Optional[Union[np.array,Tensor]]         = None,     # initial model parameter
                rho_init:Optional[Union[np.array,Tensor]]        = None,
                vp_bound:Optional[Tuple[float,float]]            = None,     # model parameter's boundary
                vs_bound:Optional[Tuple[float,float]]            = None,     # model parameter's boundary
                rho_bound:Optional[Tuple[float,float]]           = None,
                water_layer_mask:Optional[Union[np.array,Tensor]]= None,
                auto_update_rho:Optional[bool]                   = True,
                auto_update_vp :Optional[bool]                   = False,
                free_surface: Optional[bool]                     = False,
                abc_type    : Optional[str]                      = 'PML',
                abc_jerjan_alpha:Optional[float]                 = 0.0053,
                nabc:Optional[int]                               = 20,
                device                                           = 'cpu',
                dtype                                            = torch.float32
                )->None:
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
        self.DIP_model = DIP_model
        
        # initialize the model parameters
        self.pars       = ["vp","vs","rho"]
            
        self.vp_init    = torch.zeros((nz,nx),dtype=dtype,device=device) if  vp_init is None else numpy2tensor(vp_init,dtype=dtype).to(device)
        self.vs_init    = torch.zeros((nz,nx),dtype=dtype,device=device) if  vs_init is None else numpy2tensor(vs_init,dtype=dtype).to(device)
        self.rho_init   = torch.zeros((nz,nx),dtype=dtype,device=device) if rho_init is None else numpy2tensor(rho_init,dtype=dtype).to(device)
        self.vp         = self.vp_init.clone()
        self.vs         = self.vs_init.clone()
        self.rho        = self.rho_init.clone()
        self.eps        = np.zeros((nz,nx))
        self.gamma      = np.zeros((nz,nx))
        self.delta      = np.zeros((nz,nx))
        self._parameterization_thomson()
        
        # initialize the lame constant
        self.mu         = None
        self.lamu       = None
        self.lam        = None
        self.muxz       = None
        self.b          = None
        self.bx         = None
        self.bz         = None
        self._parameterization_Lame()
        
        # initialize the elastic moduli
        self.CC         = []
        self._parameterization_elastic_moduli()

        # set model bounds
        self.lower_bound["vp"]  = vp_bound[0]  if vp_bound  is not None else None
        self.lower_bound["vs"]  = vs_bound[0]  if vs_bound  is not None else None
        self.lower_bound["rho"] = rho_bound[0] if rho_bound is not None else None
        self.upper_bound["vp"]  = vp_bound[1]  if vp_bound  is not None else None
        self.upper_bound["vs"]  = vs_bound[1]  if vs_bound  is not None else None
        self.upper_bound["rho"] = rho_bound[1] if rho_bound is not None else None
        
        # check the input model
        self._check_bounds()
        self.check_dims()
        
    def get_requires_grad(self, par: str) -> bool:
        """Return the gradient of the model
        """
        if par not in self.pars:
            raise ValueError("Parameter {} not in model".format(par))
        if par == "vp":
            return self.DIP_model is not None
        if par == "vs":
            return self.DIP_model is not None
        if par == "rho":
            return False

    def get_model(self, par: str):
        if par not in ["vp","vs","rho"]:
            raise "Error input parametrs"
        elif par == "vp":
            vp  = self.vp.cpu().detach().numpy()
            return vp
        elif par == "vs":
            vs  = self.vs.cpu().detach().numpy()
            return vs
        elif par == "rho":
            rho = self.rho.cpu().detach().numpy()
            return rho

    def get_bound(self, par: str) -> Tuple[float, float]:
        if par not in ["vp","vs","rho"]:
            raise "Error input parameters"
        else:
            m_min = self.lower_bound[par]
            m_max = self.upper_bound[par]        
        return [m_min,m_max]
    
    def __repr__(self) -> str:
        """Representation of the model object
        """
        info = f"   Model with parameters {self.pars}:\n"
        info += f"  Model orig: ox = {self.ox:6.2f}, oz = {self.oz:6.2f} m\n"
        info += f"  Model grid: dx = {self.dx:6.2f}, dz = {self.dz:6.2f} m\n"
        info += f"  Model dims: nx = {self.nx:6d}, nz = {self.nz:6d}\n"
        info += f"  Model size: {self.nx * self.nz * len(self.pars)}\n"
        info += f"  Free surface: {self.free_surface}\n"
        info += f"  Absorbing layers: {self.nabc}\n"
        info += f"  NN structure\n"
        if self.DIP_model is not None:
            info += str(summary(self.DIP_model,device=self.device))
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
    
    def _parameterization_thomson(self,*args,**kw_args):
        """setting variable and gradients
        """
        # get the model
        if self.reparameterization_strategy == "vel":
            self.vp = self.DIP_model(*args,**kw_args)[0]
            self.vs = self.DIP_model(*args,**kw_args)[1]
        elif self.reparameterization_strategy == "vel_diff":
            self.vp = self.vp_init+self.DIP_model(*args,**kw_args)[0]
            self.vs = self.vs_init+self.DIP_model(*args,**kw_args)[1]
                    
        if self.auto_update_rho:
            self.set_rho_using_empirical_function()
            
        self.eps    = numpy2tensor(self.eps  ,self.dtype).to(self.device)
        self.gamma  = numpy2tensor(self.gamma,self.dtype).to(self.device)
        self.delta  = numpy2tensor(self.delta,self.dtype).to(self.device)
        
        # set model parameters
        self.eps    = torch.nn.Parameter(self.eps   ,requires_grad=False)
        self.gamma  = torch.nn.Parameter(self.gamma ,requires_grad=False)
        self.delta  = torch.nn.Parameter(self.delta ,requires_grad=False)
        return
    
    def _parameterization_Lame(self):
        """Calculate the lame parameters
        """
        mu,lamu,lam,b   = vs_vp_to_Lame(self.vp,self.vs,self.rho)
        self.mu         = mu
        self.lamu       = lamu
        self.lam        = lam
        self.b          = b
        return
    
    def _parameterization_elastic_moduli(self):
        """calculate the 21 dependent elastic moduli
        """
        # initialize elastic moduli
        CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66] = elastic_moduli_init(self.nz,self.nx,self.device,self.dtype)
        # transform thomsen parameter to elastic moduli 
        C11,C13,C33,C44,C66 = thomsen_to_elastic_moduli(self.vp,self.vs,self.rho,self.eps,self.delta,self.gamma)
        CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66]
        # define elastic moduli for isotropic model
        CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66] = elastic_moduli_for_isotropic(CC)
        # prepare staggered grid settings
        bx,bz,muxz,C44,C55,C66 = parameter_staggered_grid(self.mu,self.b,C44,C55,C66,self.nx,self.nz)
        self.bx = bx
        self.bz = bz
        self.muxz = muxz
        CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66]
        self.CC = CC
        return 
    
    def _plot_vp_vs_rho(self,**kwargs):
        """plot velocity model
        """
        plot_vp_vs_rho(self.vp,self.vs,self.rho,
                            dx=self.dx,dz=self.dz,**kwargs)
        return
    
    def _plot_eps_delta_gamma(self,**kwargs):
        """plot anisotropic parameters
        """
        plot_eps_delta_gamma(self.eps,self.delta,self.gamma,
                            dx=self.dx,dz=self.dz,**kwargs)
        return
    
    def _plot_lam_mu(self,**kwargs):
        """plot lame parameters
        """
        plot_lam_mu(self.lam,self.mu,
                            dx=self.dx,dz=self.dz,**kwargs)
        return
    
    def _plot(self,var,**kwargs):
        """plot single velocity model
        """
        model_data = self.get_model(var)
        plot_model(model_data,title=var,**kwargs)
        return
    
    def clip_params(self,par)->None:
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

    def forward(self,*args,**kwargs) -> Tuple:
        """Forward method of the elastic model class
        """
        self._parameterization_thomson()
            
        # Clip the model parameters
        self.clip_params("vp")
        self.clip_params("vs")
        self.clip_params("rho")
        
        # calculate the thomson/lame and elastic moduli parameters
        self._parameterization_Lame()
        self._parameterization_elastic_moduli()
        return 