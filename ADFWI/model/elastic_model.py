"""Elastic model containers and derived elastic parameter refresh logic."""

import numpy as np
from torch import Tensor
from typing import Optional,Tuple,Union
from ADFWI.model.base import AbstractModel
from ADFWI.model.parameters import (elastic_moduli_init, vs_vp_to_Lame, thomsen_to_elastic_moduli,
                         elastic_moduli_for_isotropic,elastic_moduli_for_TI,
                         parameter_staggered_grid)
from ADFWI.view import (plot_vp_vs_rho,plot_eps_delta_gamma,plot_lam_mu,plot_model)

class IsotropicElasticModel(AbstractModel):
    """Isotropic elastic model with persistent ``vp``, ``vs``, and ``rho``."""
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
                device                                           = None,
                dtype                                            = None
                )->None:
        """
        Parameters:
        --------------
        ox (float), oz (float)                   : Origin coordinates of the model in the x- and z-directions (meters).
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
        device (str)                             : Device on which to place the model. Uses the active ADFWI backend when omitted.
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
        self._set_parameter_bounds({
            "vp": vp_bound,
            "vs": vs_bound,
            "rho": rho_bound,
        })
        
        # set model gradients
        self._set_requires_grad_flags({
            "vp": self.vp_grad,
            "vs": self.vs_grad,
            "rho": self.rho_grad,
        })
        
        # check the input model
        self._check_bounds()
        self.check_dims()
        
        # update rho using the empirical function
        self.auto_update_rho = auto_update_rho
        self.auto_update_vp  = auto_update_vp
    
        self.water_layer_mask = self._prepare_water_layer_mask(water_layer_mask)
            
    def _parameterization_thomson(self):
        """setting variable and gradients
        """
        self._register_model_parameter("vp", self.vp, self.vp_grad)
        self._register_model_parameter("vs", self.vs, self.vs_grad)
        self._register_model_parameter("rho", self.rho, self.rho_grad)
        self._register_model_parameter("eps", self.eps, False)
        self._register_model_parameter("gamma", self.gamma, False)
        self._register_model_parameter("delta", self.delta, False)
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
        self._register_model_parameter("rho", rho_empirical, self.rho_grad)
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
        self._register_model_parameter("vp", vp_empirical, self.vp_grad)
        return
    

    def forward(self) -> None:
        """Refresh constraints and derived elastic quantities for propagation."""
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
    """Anisotropic elastic model with velocity, density, and Thomsen parameters."""
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
                device                                      = None,
                dtype                                       = None
                )->None:
        """
        Parameters:
        --------------
        ox (float), oz (float)                      : Origin coordinates of the model in the x- and z-directions (meters).
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
        device (str)                                : Device on which to place the model. Uses the active ADFWI backend when omitted.
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
        self._set_parameter_bounds({
            "vp": vp_bound,
            "vs": vs_bound,
            "rho": rho_bound,
            "eps": eps_bound,
            "gamma": gamma_bound,
            "delta": delta_bound,
        })
        
        # set model gradients
        self._set_requires_grad_flags({
            "vp": self.vp_grad,
            "vs": self.vs_grad,
            "rho": self.rho_grad,
            "eps": self.eps_grad,
            "gamma": self.gamma_grad,
            "delta": self.delta_grad,
        })
        
        # check the input model
        self._check_bounds()
        self.check_dims()
        
        # update rho using the empirical function
        self.auto_update_rho = auto_update_rho
        self.auto_update_vp  = auto_update_vp
        
        self.water_layer_mask = self._prepare_water_layer_mask(water_layer_mask)
                
    def _parameterization_thomson(self):
        self._register_model_parameter("vp", self.vp, self.vp_grad)
        self._register_model_parameter("vs", self.vs, self.vs_grad)
        self._register_model_parameter("rho", self.rho, self.rho_grad)
        self._register_model_parameter("eps", self.eps, self.eps_grad)
        self._register_model_parameter("gamma", self.gamma, self.gamma_grad)
        self._register_model_parameter("delta", self.delta, self.delta_grad)
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
        self._register_model_parameter("rho", rho, self.rho_grad)
        return
    
    def set_vp_using_empirical_function(self):
        """approximate vp via empirical relations with vs
        """
        vs = self.vs.cpu().detach().numpy()
        vp = vs*np.sqrt(3)
        self._register_model_parameter("vp", vp, self.vp_grad)
        return
    
        
    def forward(self) -> None:
        """Refresh constraints and derived elastic quantities for propagation."""
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
