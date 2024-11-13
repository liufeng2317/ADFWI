'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: LiuFeng
* LastEditTime: 2024-06-01 14:47:30
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''
import numpy as np
import torch
from torch import Tensor
from typing import Optional,Tuple,Union
from ADFWI.utils       import gpu2cpu,numpy2tensor
from ADFWI.model.base  import AbstractModel
from ADFWI.view        import (plot_vp_rho,plot_model)
from ADFWI.survey      import Survey

class AcousticModel(AbstractModel):
    """Acoustic Velocity model with parameterization of vp and rho
    Parameters:
    --------------
        ox (float),oz(float)        : Origin of the model in x- and z- direction (m)
        nx (int),nz(int)            : NUmber of grid points in x- and z- direction (m)
        dx (float),dz(float)        : Grid size in x- and z- direction (m)
        vp (array)                  : P-wave velocity model with shape (nz,nx)
        rho (array)                 : density model with shape (nz,nx)
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
                nx:int,nz:int,
                dx:float,dz:float,
                vp:Optional[Union[np.array,Tensor]]              = None,     # model parameter
                rho:Optional[Union[np.array,Tensor]]             = None,
                vp_bound: Optional[Tuple[float, float]]          = None,     # model parameter's boundary
                rho_bound: Optional[Tuple[float, float]]         = None,
                vp_grad:Optional[bool]                           = False,    # requires gradient or not
                rho_grad:Optional[bool]                          = False,
                free_surface:Optional[bool]                      = False,
                abc_type:Optional[str]                           = 'PML',
                abc_jerjan_alpha:Optional[float]                 = 0.0053,
                nabc:Optional[int]                               = 20,
                auto_update_rho:Optional[bool]                   = True,
                water_layer_mask:Optional[Union[np.array,Tensor]]= None,
                device                                           = 'cpu',
                dtype                                            = torch.float32
                )->None:
        # initialize the common model parameters
        super().__init__(ox,oz,nx,nz,dx,dz,free_surface,abc_type,abc_jerjan_alpha,nabc,device,dtype)

        # initialize the model parameters
        self.pars       = ["vp","rho"]
        self.vp         = vp.copy()
        self.rho        = rho.copy()
        self.vp_grad    = vp_grad
        self.rho_grad   = rho_grad
        self._parameterization()
        
        # set model bounds
        self.lower_bound["vp"]  =  vp_bound[0]  if vp_bound  is not None else None
        self.lower_bound["rho"] = rho_bound[0]  if rho_bound is not None else None
        self.upper_bound["vp"]  =  vp_bound[1]  if vp_bound  is not None else None
        self.upper_bound["rho"] = rho_bound[1]  if rho_bound is not None else None
        
        # set model gradients
        self.requires_grad["vp"]    = self.vp_grad
        self.requires_grad["rho"]   = self.rho_grad
        
        # check the input model
        self._check_bounds()
        self.check_dims()
        
        # update rho using the empirical function
        self.auto_update_rho = auto_update_rho
        
        if water_layer_mask is not None:
            self.water_layer_mask = numpy2tensor(water_layer_mask,dtype=torch.bool).to(device)
        else:
            self.water_layer_mask = None
        
    def _parameterization(self):
        """setting variable and gradients
        """
        # numpy2tensor
        self.vp     = numpy2tensor(self.vp   ,self.dtype).to(self.device)
        self.rho    = numpy2tensor(self.rho  ,self.dtype).to(self.device)
        # set model parameters
        self.vp     = torch.nn.Parameter(self.vp    ,requires_grad=self.vp_grad)
        self.rho    = torch.nn.Parameter(self.rho   ,requires_grad=self.rho_grad)
        return
    
    def get_clone_data(self) -> Tuple:
        """clone the class
        """
        kwargs = super().get_clone_data()
        return kwargs
    
    def _plot_vp_rho(self,**kwargs):
        """plot velocity model
        """
        plot_vp_rho(self.vp,self.rho,
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
                    m.data = torch.where(self.water_layer_mask.contiguous(), m_temp.data, m.data)
        return
        
    def forward(self) -> Tuple:
        """Forward method of the elastic model class
        """
        # using the empirical function to setting rho
        if self.auto_update_rho and not self.rho_grad:
            self.set_rho_using_empirical_function()
            
        # Clip the model parameters
        self.clip_params()
        return 