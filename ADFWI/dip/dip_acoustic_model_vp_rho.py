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
from ADFWI.view import plot_vp_rho,plot_model
from typing import Optional,Tuple,Union
import torch
from torch import Tensor
import numpy as np
from torchinfo import summary

class DIP_AcousticModel2(AbstractModel):
    """Acoustic Velocity model with re-parameterize vp and rho using one model
    """
    def __init__(self,
                ox:float,oz:float,
                nx:int  ,nz:int,
                dx:float,dz:float,
                DIP_model                                        = None,     # deep image prior models
                reparameterization_strategy                      = "vel",    # vel/vel_diff
                vp_init:Optional[Union[np.array,Tensor]]         = None,     # initial model parameter
                rho_init:Optional[Union[np.array,Tensor]]        = None,
                vp_bound    : Optional[Tuple[float, float]]      = None,     # model parameter's boundary
                rho_bound   : Optional[Tuple[float, float]]      = None,
                water_layer_mask:Optional[Union[np.array,Tensor]]= None,
                free_surface: Optional[bool]                     = False,
                abc_type    : Optional[str]                      = 'PML',
                abc_jerjan_alpha:Optional[float]                 = 0.0053,
                nabc:Optional[int]                               = 20,
                device                                           = 'cpu',
                dtype                                            = torch.float32
                )->None:
        """
        Parameters:
        --------------
        ox (float), oz (float)                               : Non use, the origin coordinates of the model in the x- and z- directions (in meters).
        nx (int), nz (int)                                   : The number of grid points in the x- and z- directions.
        dx (float), dz (float)                               : The grid spacing in the x- and z- directions (in meters).
        DIP_model                                            : reparameterized vp and rho using a deep neural network, by default None
        vp_init                                              : the initial vp model
        rho_init                                             : the initial rho model
        vp_bound (Optional[Tuple[float, float]])             : The lower and upper bounds for the P-wave velocity model. Default is None.
        rho_bound (Optional[Tuple[float, float]])            : The lower and upper bounds for the density model. Default is None.
        water_layer_mask (Optional[Union[np.array, Tensor]]) : A mask for the water layer (not update), if applicable. Default is None.
        free_surface (Optional[bool])                        : A flag to indicate the presence of a free surface in the model. Default is False.
        abc_type (Optional[str])                             : The type of absorbing boundary condition used in the model. Options include 'PML' and 'Jerjan'. Default is 'PML'.
        abc_jerjan_alpha (Optional[float])                   : The attenuation factor for the Jerjan boundary condition. Default is 0.0053.
        nabc (Optional[int])                                 : The number of grid cells dedicated to the absorbing boundary, default is 20.
        device (str)                                         : The device on which to run the model. Options are 'cpu' or 'cuda'. Default is 'cpu'.
        dtype (torch.dtype)                                  : The data type for PyTorch tensors. Default is torch.float32.
        """
        # initialize the common model parameters
        super().__init__(ox,oz,nx,nz,dx,dz,free_surface,abc_type,abc_jerjan_alpha,nabc,device,dtype)
        
        self.reparameterization_strategy = reparameterization_strategy
        
        # gradient mask
        if water_layer_mask is not None:
            self.water_layer_mask = numpy2tensor(water_layer_mask,dtype=torch.bool).to(device)
        else:
            self.water_layer_mask = None
        
        # Neural networks
        self.DIP_model      = DIP_model
        
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
        
        
    def get_requires_grad(self, par: str) -> bool:
        """Return the gradient of the model

        Parameters
        ----------
        par (str) : Model parameter name

        Returns
        -------
        grad (bool) : Flag for gradient of the model
        """

        if par not in self.pars:
            raise ValueError("Parameter {} not in model".format(par))
        if par == "vp":
            return self.DIP_model is not None
        if par == "rho":
            return self.DIP_model is not None

    def get_model(self, par: str):
        if par not in ["vp","rho"]:
            raise "Error input parametrs"
        elif par == "vp":
            vp  = self.vp.cpu().detach().numpy()
            return vp
        elif par == "rho":
            rho = self.rho.cpu().detach().numpy()
            return rho

    def get_bound(self, par: str) -> Tuple[float, float]:
        if par not in ["vp","rho"]:
            raise "Error input parameters"
        else:
            m_min = self.lower_bound[par]
            m_max = self.upper_bound[par]        
        return [m_min,m_max]
    
    def __repr__(self) -> str:
        """Representation of the model object

        Returns
        -------
        repr (str) : Representation of the model object
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
     
    
    def _parameterization(self,*args,**kw_args):
        """setting variable and gradients
        """
        if self.reparameterization_strategy == "vel":
            self.vp     = self.DIP_model(*args,**kw_args)[0]
            self.rho    = self.DIP_model(*args,**kw_args)[1]
        elif self.reparameterization_strategy == "vel_diff":
            self.vp     = self.vp_init  + self.DIP_model(*args,**kw_args)[0]
            self.rho    = self.rho_init + self.DIP_model(*args,**kw_args)[1]
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
        self._parameterization()
        
        self.clip_params("vp")
        self.clip_params("rho")
        return