'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: LiuFeng
* LastEditTime: 2024-05-15 19:30:38
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''
import numpy as np
import torch
from torch import Tensor
from typing import Optional,Tuple,Union,List
from ADFWI.utils import gpu2cpu,numpy2tensor,tensor2numpy
from ADFWI.model.base import AbstractModel
from ADFWI.model.parameters import (thomsen_init,elastic_moduli_init,
                         vs_vp_to_Lame,thomsen_to_elastic_moduli,
                         elastic_moduli_for_isotropic,elastic_moduli_for_TI,
                         parameter_staggered_grid)
from ADFWI.view import (plot_vp_rho,plot_vp_vs_rho,plot_eps_delta_gamma,plot_lam_mu,plot_model)
from ADFWI.survey import Survey

class IsotropicElasticModel(AbstractModel):
    """Isotropic Elastic Velocity model with parameterization of vp vs and rho
    """
    def __init__(self,
                ox:float,oz:float,
                nx:int,nz:int,
                dx:float,dz:float,
                vp:Optional[Union[np.array,Tensor]]              = None,     # model parameter
                vs:Optional[Union[np.array,Tensor]]              = None,
                rho:Optional[Union[np.array,Tensor]]             = None,
                vp_bound: Optional[Tuple[float, float]]          = None,     # model parameter's boundary
                vs_bound: Optional[Tuple[float, float]]          = None,
                rho_bound: Optional[Tuple[float, float]]         = None,
                vp_grad:Optional[bool]                           = False,    # requires gradient or not
                vs_grad:Optional[bool]                           = False,
                rho_grad:Optional[bool]                          = False,
                free_surface:Optional[bool]                      = False,
                abc_type:Optional[str]                           = 'PML',
                abc_jerjan_alpha:Optional[float]                 = 0.0053,
                nabc:Optional[int]                               = 20,
                auto_update_rho:Optional[bool]                   = True,
                auto_update_vp:Optional[bool]                    = False,
                water_layer_mask:Optional[Union[np.array,Tensor]]= None,
                device                                           = 'cpu',
                dtype                                            = torch.float32
                )->None:
        """
        Parameters:
        --------------
        ox (float), oz (float)                   : Non use, the origin coordinates of the model in the x- and z- directions (in meters).
        nx (int), nz (int)                       : The number of grid points in the x- and z- directions.
        dx (float), dz (float)                   : The grid spacing in the x- and z- directions (in meters).
        vp (Optional[Union[np.array, Tensor]])   : P-wave velocity model with shape (nz, nx). Default is None.
        vs (Optional[Union[np.array, Tensor]])   : S-wave velocity model with shape (nz, nx). Default is None.
        rho (Optional[Union[np.array, Tensor]])  : Density model with shape (nz, nx). Default is None.
        vp_bound (Optional[Tuple[float, float]]) : The lower and upper bounds for the P-wave velocity model. Default is None.
        vs_bound (Optional[Tuple[float, float]]) : The lower and upper bounds for the S-wave velocity model. Default is None.
        rho_bound (Optional[Tuple[float, float]]): The lower and upper bounds for the density model. Default is None.
        vp_grad (Optional[bool])                 : A flag to indicate if the gradient of the P-wave velocity model is needed. Default is False.
        vs_grad (Optional[bool])                 : A flag to indicate if the gradient of the S-wave velocity model is needed. Default is False.
        rho_grad (Optional[bool])                : A flag to indicate if the gradient of the density model is needed. Default is False.
        free_surface (Optional[bool])            : A flag to indicate the presence of a free surface in the model. Default is False.
        abc_type (Optional[str])                 : The type of absorbing boundary condition used in the model. Options include 'PML', 'Jerjan', etc. Default is 'PML'.
        abc_jerjan_alpha (Optional[float])       : The attenuation factor for the Jerjan boundary condition. Default is 0.0053.
        nabc (Optional[int])                     : The number of grid cells dedicated to the absorbing boundary. Default is 20.
        auto_update_rho (Optional[bool])         : Whether to automatically update the density model during inversion. Default is True.
        auto_update_vp (Optional[bool])          : Whether to automatically update the P-wave velocity model during inversion. Default is False.
        water_layer_mask (Optional[Union[np.array, Tensor]]) : A mask for the water layer (not update), if applicable. Default is None.
        device (str)                             : The device on which to run the model. Options are 'cpu' or 'cuda'. Default is 'cpu'.
        dtype (torch.dtype)                      : The data type for PyTorch tensors. Default is torch.float32.
        """
        # initialize the common model parameters
        super().__init__(ox,oz,nx,nz,dx,dz,free_surface,abc_type,abc_jerjan_alpha,nabc,device,dtype)

        # initialize the thomson model model parameters
        self.pars       = ["vp","vs","rho"]
        self.vp         = vp.copy()
        self.vs         = vs.copy()
        self.rho        = rho.copy()
        self.vp_grad    = vp_grad
        self.vs_grad    = vs_grad
        self.rho_grad   = rho_grad
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
        
        # set model gradients
        self.requires_grad["vp"]    = self.vp_grad
        self.requires_grad["vs"]    = self.vs_grad
        self.requires_grad["rho"]   = self.rho_grad
        
        # check the input model
        self._check_bounds()
        self.check_dims()
        
        # update rho using the empirical function
        self.auto_update_rho = auto_update_rho
        self.auto_update_vp  = auto_update_vp
    
        if water_layer_mask is not None:
            self.water_layer_mask = numpy2tensor(water_layer_mask,dtype=torch.bool).to(device)
        else:
            self.water_layer_mask = None
            
    def _parameterization_thomson(self):
        """setting variable and gradients
        """
        # numpy2tensor
        self.vp     = numpy2tensor(self.vp   ,self.dtype).to(self.device)
        self.vs     = numpy2tensor(self.vs   ,self.dtype).to(self.device)
        self.rho    = numpy2tensor(self.rho  ,self.dtype).to(self.device)
        self.eps    = numpy2tensor(self.eps  ,self.dtype).to(self.device)
        self.gamma  = numpy2tensor(self.gamma,self.dtype).to(self.device)
        self.delta  = numpy2tensor(self.delta,self.dtype).to(self.device)
        # set model parameters
        self.vp     = torch.nn.Parameter(self.vp    ,requires_grad=self.vp_grad)
        self.vs     = torch.nn.Parameter(self.vs    ,requires_grad=self.vs_grad)
        self.rho    = torch.nn.Parameter(self.rho   ,requires_grad=self.rho_grad)
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
    
    def get_clone_data(self) -> Tuple:
        kwargs = super().get_clone_data()
        return kwargs
    
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
    
    def set_rho_using_empirical_function(self):
        """approximate rho via empirical relations with vp
        """
        rho         = self.rho.cpu().detach().numpy()
        vp          = self.vp.cpu().detach().numpy()
        rho_empirical  = np.power(vp, 0.25) * 310
        if self.water_layer_mask is not None:
            mask = self.water_layer_mask.cpu().detach().numpy()
            rho_empirical[mask] = rho[mask]
        rho         = numpy2tensor(rho_empirical,self.dtype).to(self.device)
        self.rho    = torch.nn.Parameter(rho   ,requires_grad=self.rho_grad)
        return
    
    def set_vp_using_empirical_function(self):
        """approximate vp via empirical relations with vs
        """
        vp = self.vp.cpu().detach().numpy()
        vs = self.vs.cpu().detach().numpy()
        vp_empirical = vs*np.sqrt(3)
        if self.water_layer_mask is not None:
            mask = self.water_layer_mask.cpu().detach().numpy()
            vp_empirical[mask] = vp[mask]
        vp = numpy2tensor(vp_empirical,self.dtype).to(self.device)
        self.vp = torch.nn.Parameter(vp,requires_grad=self.vp_grad)
        return
    
    def clip_params(self)->None:
        """Clip the model parameters to the given bounds
        """
        for par in self.pars:
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
                    m.data = torch.where(self.water_layer_mask, m_temp.data, m.data)
        return
    

    def forward(self) -> None:
        """Forward method of the elastic model class
        
        """
        # set the constraints on the parameters if necessary
        if self.auto_update_rho:
            self.set_rho_using_empirical_function()
            
        if self.auto_update_vp:
            self.set_vp_using_empirical_function()
        
        # Clip the model parameters
        self.clip_params()
        
        # calculate the thomson/lame and elastic moduli parameters
        self._parameterization_Lame()
        self._parameterization_elastic_moduli()
        return 
    
    
class AnisotropicElasticModel(AbstractModel):
    """AnIsotropic Elastic Velocity model with parameterization of vp vs rho eps and delta. (VTI/HTI)
    """
    def __init__(self,
                ox:float,oz:float,
                nx:int,nz:int,
                dx:float,dz:float,
                vp:Optional[Union[np.array,Tensor]]         = None,     # model parameter
                vs:Optional[Union[np.array,Tensor]]         = None,
                rho:Optional[Union[np.array,Tensor]]        = None,
                eps:Optional[Union[np.array,Tensor]]        = None,
                gamma:Optional[Union[np.array,Tensor]]      = None,
                delta:Optional[Union[np.array,Tensor]]      = None,
                vp_bound: Optional[Tuple[float, float]]     = None,     # model parameter's boundary
                vs_bound: Optional[Tuple[float, float]]     = None,
                rho_bound: Optional[Tuple[float, float]]    = None,
                eps_bound: Optional[Tuple[float, float]]    = None,
                gamma_bound: Optional[Tuple[float, float]]  = None,
                delta_bound: Optional[Tuple[float, float]]  = None,
                vp_grad:Optional[bool]                      = False,    # requires gradient or not
                vs_grad:Optional[bool]                      = False,
                rho_grad:Optional[bool]                     = False,
                eps_grad:Optional[bool]                     = False,
                gamma_grad:Optional[bool]                   = False,
                delta_grad:Optional[bool]                   = False,
                free_surface:Optional[bool]                 = False,
                anisotropic_type:Optional[str]              = "vti",
                abc_type:Optional[str]                      = 'PML',
                abc_jerjan_alpha:Optional[float]            = 0.0053,
                nabc:Optional[int]                          = 20,
                auto_update_rho:Optional[bool]              = False,    # auto update parameters
                auto_update_vp:Optional[bool]               = False,
                water_layer_mask:Optional[Union[np.array,Tensor]]= None,
                device                                      = 'cpu',
                dtype                                       = torch.float32
                )->None:
        """
        Parameters:
        --------------
        ox (float), oz (float)                      : Non use, the origin coordinates of the model in the x- and z- directions (meters).
        nx (int), nz (int)                          : The number of grid points in the x- and z- directions.
        dx (float), dz (float)                      : The grid spacing in the x- and z- directions (meters).
        vp (Optional[Union[np.array, Tensor]])      : P-wave velocity model with shape (nz, nx). Default is None.
        vs (Optional[Union[np.array, Tensor]])      : S-wave velocity model with shape (nz, nx). Default is None.
        rho (Optional[Union[np.array, Tensor]])     : Density model with shape (nz, nx). Default is None.
        eps (Optional[Union[np.array, Tensor]])     : Anisotropic parameter epsilon (vti/hti model), shape (nz, nx). Default is None.
        gamma (Optional[Union[np.array, Tensor]])   : Anisotropic parameter gamma (vti/hti model), shape (nz, nx). Default is None.
        delta (Optional[Union[np.array, Tensor]])   : Anisotropic parameter delta (vti/hti model), shape (nz, nx). Default is None.
        vp_bound (Optional[Tuple[float, float]])    : The lower and upper bounds for the P-wave velocity model. Default is None.
        vs_bound (Optional[Tuple[float, float]])    : The lower and upper bounds for the S-wave velocity model. Default is None.
        rho_bound (Optional[Tuple[float, float]])   : The lower and upper bounds for the density model. Default is None.
        eps_bound (Optional[Tuple[float, float]])   : The lower and upper bounds for epsilon. Default is None.
        gamma_bound (Optional[Tuple[float, float]]) : The lower and upper bounds for gamma. Default is None.
        delta_bound (Optional[Tuple[float, float]]) : The lower and upper bounds for delta. Default is None.
        vp_grad (Optional[bool])                    : Whether to compute the gradient of P-wave velocity. Default is False.
        vs_grad (Optional[bool])                    : Whether to compute the gradient of S-wave velocity. Default is False.
        rho_grad (Optional[bool])                   : Whether to compute the gradient of the density model. Default is False.
        eps_grad (Optional[bool])                   : Whether to compute the gradient of epsilon. Default is False.
        gamma_grad (Optional[bool])                 : Whether to compute the gradient of gamma. Default is False.
        delta_grad (Optional[bool])                 : Whether to compute the gradient of delta. Default is False.
        free_surface (Optional[bool])               : Whether to include a free surface in the model. Default is False.
        anisotropic_type (Optional[str])            : Type of anisotropic model ('vti', 'hti', etc.). Default is 'vti'.
        abc_type (Optional[str])                    : Type of absorbing boundary condition ('PML', 'Jerjan', etc.). Default is 'PML'.
        abc_jerjan_alpha (Optional[float])          : Attenuation factor for Jerjan boundary condition. Default is 0.0053.
        nabc (Optional[int])                        : Number of absorbing boundary cells. Default is 20.
        auto_update_rho (Optional[bool])            : Whether to auto-update the density model during inversion. Default is False.
        auto_update_vp (Optional[bool])             : Whether to auto-update the P-wave velocity model during inversion. Default is False.
        water_layer_mask (Optional[Union[np.array, Tensor]]) : Mask for the water layer (not update), if applicable. Default is None.
        device (str)                                : Device for running the model ('cpu' or 'cuda'). Default is 'cpu'.
        dtype (torch.dtype)                         : Data type for PyTorch tensors. Default is torch.float32.
        """
        # initialize the common model parameters
        super().__init__(ox,oz,nx,nz,dx,dz,free_surface,abc_type,abc_jerjan_alpha,nabc,device,dtype)

        # initialize the thomson model model parameters
        self.pars       = ["vp","vs","rho","eps","gamma","delta"]
        self.vp         = vp.copy()
        self.vs         = vs.copy()
        self.rho        = rho.copy()
        self.eps        = eps.copy()
        self.gamma      = gamma.copy()
        self.delta      = delta.copy()
        self.vp_grad    = vp_grad
        self.vs_grad    = vs_grad
        self.rho_grad   = rho_grad
        self.eps_grad   = eps_grad
        self.gamma_grad = gamma_grad
        self.delta_grad = delta_grad
        
        self.anisotropic_type = anisotropic_type
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
        self.lower_bound["vp"]      = vp_bound[0] if vp_bound  is not None else None
        self.lower_bound["vs"]      = vs_bound[0] if vs_bound  is not None else None
        self.lower_bound["rho"]     = rho_bound[0] if rho_bound is not None else None
        self.lower_bound['eps']     = eps_bound[0] if eps_bound is not None else None
        self.lower_bound['gamma']   = gamma_bound[0] if gamma_bound is not None else None
        self.lower_bound['delta']   = delta_bound[0] if delta_bound is not None else None
        self.upper_bound["vp"]      = vp_bound[1] if vp_bound  is not None else None
        self.upper_bound["vs"]      = vs_bound[1] if vs_bound  is not None else None
        self.upper_bound["rho"]     = rho_bound[1] if rho_bound is not None else None
        self.upper_bound["eps"]     = eps_bound[1] if eps_bound  is not None else None
        self.upper_bound["gamma"]   = gamma_bound[1] if gamma_bound  is not None else None
        self.upper_bound["delta"]   = delta_bound[1] if delta_bound is not None else None
        
        # set model gradients
        self.requires_grad["vp"]    = self.vp_grad
        self.requires_grad["vs"]    = self.vs_grad
        self.requires_grad["rho"]   = self.rho_grad
        self.requires_grad["eps"]   = self.eps_grad
        self.requires_grad["gamma"] = self.gamma_grad
        self.requires_grad["delta"] = self.delta_grad
        
        # check the input model
        self._check_bounds()
        self.check_dims()
        
        # update rho using the empirical function
        self.auto_update_rho = auto_update_rho
        self.auto_update_vp  = auto_update_vp
        
        if water_layer_mask is not None:
            self.water_layer_mask = numpy2tensor(water_layer_mask,dtype=torch.bool).to(device)
        else:
            self.water_layer_mask = None
                
    def _parameterization_thomson(self):
        # numpy2tensor
        self.vp     = numpy2tensor(self.vp   ,self.dtype).to(self.device)
        self.vs     = numpy2tensor(self.vs   ,self.dtype).to(self.device)
        self.rho    = numpy2tensor(self.rho  ,self.dtype).to(self.device)
        self.eps    = numpy2tensor(self.eps  ,self.dtype).to(self.device)
        self.gamma  = numpy2tensor(self.gamma,self.dtype).to(self.device)
        self.delta  = numpy2tensor(self.delta,self.dtype).to(self.device)
        # set model parameters
        self.vp     = torch.nn.Parameter(self.vp    ,requires_grad=self.vp_grad)
        self.vs     = torch.nn.Parameter(self.vs    ,requires_grad=self.vs_grad)
        self.rho    = torch.nn.Parameter(self.rho   ,requires_grad=self.rho_grad)
        self.eps    = torch.nn.Parameter(self.eps   ,requires_grad=self.eps_grad)
        self.gamma  = torch.nn.Parameter(self.gamma ,requires_grad=self.gamma_grad)
        self.delta  = torch.nn.Parameter(self.delta ,requires_grad=self.delta_grad)
        return
    
    def _parameterization_Lame(self):
        mu,lamu,lam,b   = vs_vp_to_Lame(self.vp,self.vs,self.rho)
        self.mu         = mu
        self.lamu       = lamu
        self.lam        = lam
        self.b          = b
        return
    
    def _parameterization_elastic_moduli(self):
        # initialize elastic moduli
        CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66] = elastic_moduli_init(self.nz,self.nx,self.device,self.dtype)
        # transform thomsen parameter to elastic moduli 
        C11,C13,C33,C44,C66 = thomsen_to_elastic_moduli(self.vp,self.vs,self.rho,self.eps,self.delta,self.gamma)
        CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66]
        # define elastic moduli for anisotropic model
        CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66] = elastic_moduli_for_TI(CC,anisotropic_type=self.anisotropic_type)
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
    
    def set_rho_using_empirical_function(self):
        """approximate rho via empirical relations with vp
        """
        vp          = self.vp.cpu().detach().numpy()
        rho         = np.power(vp, 0.25) * 310
        rho         = numpy2tensor(rho,self.dtype).to(self.device)
        self.rho    = torch.nn.Parameter(rho   ,requires_grad=self.rho_grad)
        return
    
    def set_vp_using_empirical_function(self):
        """approximate vp via empirical relations with vs
        """
        vs = self.vs.cpu().detach().numpy()
        vp = vs*np.sqrt(3)
        vp = numpy2tensor(vp,self.dtype).to(self.device)
        self.vp = torch.nn.Parameter(vp,requires_grad=self.vp_grad)
        return
    
    def clip_params(self)->None:
        """Clip the model parameters to the given bounds
        """
        for par in self.pars:
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
                    m.data = torch.where(self.water_layer_mask, m_temp.data, m.data)
        return
    
        
    def forward(self) -> None:
        """Forward method of the elastic model class
        """
        # set the constraints on the parameters if necessary
        if self.auto_update_rho:
            self.set_rho_using_empirical_function()
            
        if self.auto_update_vp:
            self.set_vp_using_empirical_function()
            
        # Clip the model parameters
        self.clip_params()
        
        # calculate the thomson/lame and elastic moduli parameters
        self._parameterization_Lame()
        self._parameterization_elastic_moduli()
        return 