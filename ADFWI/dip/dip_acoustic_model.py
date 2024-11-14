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

class DIP_AcousticModel(AbstractModel):
    """Acoustic Velocity model with parameterization of vp and rho
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
                DIP_model                                       = None,
                DIP_model_rho                                   = None,
                vp_init:Optional[Union[np.array,Tensor]]        = None,     # model parameter
                rho_init:Optional[Union[np.array,Tensor]]       = None,
                vp_bound    : Optional[Tuple[float, float]]     = None,     # model parameter's boundary
                rho_bound   : Optional[Tuple[float, float]]     = None,
                gradient_mask:Optional[Union[np.array,Tensor]]  = None,
                gradient_mute:Optional[int]                     = None,
                free_surface: Optional[bool]                    = False,
                abc_type    : Optional[str]                     = 'PML',
                abc_jerjan_alpha:Optional[float]                = 0.0053,
                nabc:Optional[int]                              = 20,
                device                                          = 'cpu',
                dtype                                           = torch.float32
                )->None:
        # initialize the common model parameters
        super().__init__(ox,oz,nx,nz,dx,dz,free_surface,abc_type,abc_jerjan_alpha,nabc,device,dtype)
        
        self.DIP_model      = DIP_model
        self.DIP_model_rho  = DIP_model_rho
        
        # initialize the model parameters
        self.pars       = ["vp","rho"]
            
        if vp_init is not None:
            self.vp_init    = numpy2tensor(vp_init,dtype=dtype).to(device)
        if rho_init is not None:
            self.rho_init   = numpy2tensor(rho_init,dtype=dtype).to(device)
        self.vp         = torch.zeros((nz,nx),dtype=dtype).to(device)
        self.rho        = torch.zeros((nz,nx),dtype=dtype).to(device)
        self._parameterization()
        
        # gradient mask
        if gradient_mask is not None:
            self.gradient_mask = numpy2tensor(gradient_mask).to(device)
        else:
            self.gradient_mask = gradient_mask
        
        if gradient_mute is not None:
            self.gradient_mask = torch.ones_like(self.vp,dtype=dtype).to(device)
            self.gradient_mask[:gradient_mute,:] = 0

        # set model bounds
        self.lower_bound["vp"]  =  vp_bound[0]  if vp_bound  is not None else None
        self.lower_bound["rho"] = rho_bound[0]  if rho_bound is not None else None
        self.upper_bound["vp"]  =  vp_bound[1]  if vp_bound  is not None else None
        self.upper_bound["rho"] = rho_bound[1]  if rho_bound is not None else None
        
        # check the input model
        self._check_bounds()
        self.check_dims()
    
    def get_clone_data(self) -> Tuple:
        """Return the data required for cloning the model

        Returns
        -------
        args (Tuple)    : Arguments of the model
        kwargs (Dict)   : Keyword arguments of the model
        """
        kwargs = {}
        for par in self.pars:
            kwargs[par] = self.get_model(par)
            kwargs[par + "_bound"] = self.get_bound(par)
            kwargs[par + "_grad"]  = self.get_requires_grad(par)

        kwargs['ox']           = self.ox 
        kwargs['oz']           = self.oz 
        kwargs['dx']           = self.dx 
        kwargs['dz']           = self.dz 
        kwargs['nx']           = self.nx 
        kwargs['nz']           = self.nz 
        kwargs["free_surface"] = self.free_surface
        kwargs["nabc"]         = self.nabc

        return kwargs

    def save(self, filename: str) -> None:
        """Save the model object to a file

        Parameters
        ----------
        filename (str) : File name of the model object to be saved
        """
        kwargs = self.get_clone_data()

        # save the model to npz file
        np.savez(filename, **kwargs)
        return
        
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
        if self.DIP_model_rho is not None:
            info += str(summary(self.DIP_model_rho,device=self.device))
        return info
    
    def set_rho_using_empirical_function(self):
        """approximate rho via empirical relations with vp
        """
        vp          = self.vp.cpu().detach().numpy()
        rho         = np.power(vp, 0.25) * 310
        rho         = numpy2tensor(rho,self.dtype).to(self.device)
        self.rho    = rho
        return
    
    def _parameterization(self,*args,**kw_args):
        """setting variable and gradients
        """
        if self.DIP_model is not None:
            self.vp     = self.DIP_model(*args,**kw_args)
        if self.DIP_model_rho is not None:
            self.rho    = self.DIP_model_rho(*args,**kw_args)
        else:
            self.set_rho_using_empirical_function()
        return
    
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

    def forward(self,*args,**kwargs) -> Tuple:
        """Forward method of the elastic model class
        """
        vp_last  = self.vp.detach().clone()
        rho_last = self.rho.detach().clone()
        self._parameterization()
        
        if self.gradient_mask is not None:
            mask = self.gradient_mask == 0
            self.vp[mask] = vp_last[mask]
            self.rho[mask] = rho_last[mask]
        self.constrain_range(self.vp,self.lower_bound["vp"],self.upper_bound["vp"])
        self.constrain_range(self.rho, self.lower_bound["rho"], self.upper_bound["rho"])
        return