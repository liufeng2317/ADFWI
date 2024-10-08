import numpy as np
import torch
from torch import Tensor
from abc import abstractmethod
from typing import Optional,Tuple,Union
from ADFWI.utils import gpu2cpu,numpy2tensor


units = {
    "vp"    : "m/s",
    "vs"    : "m/s",
    "rho"   : "kg/m^3",
    "lam"   : "Pa",
    "mu"    : "Pa",
    "eps"   : "none",
    "gamma" : "none",
    "delta" : "none",
    "vs_vp" : "none",
}
eps = 1e-7

class AbstractModel(torch.nn.Module):
    """ Abstract model class for FWI models

    Parameters
    ----------
        ox (float)                  : Origin of the model in x-direction (m)
        oz (float)                  : Origin of the model in z-direction (m)
        dx (float)                  : Grid size in x-direction (m)
        dz (float)                  : Grid size in z-direction (m)
        nx (int)                    : Number of grid points in x-direction
        nz (int)                    : Number of grid points in z-direction
        free_surface (bool)         : Flag for free surface, default is False
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
                free_surface:Optional[bool]     = False,
                abc_type:Optional[str]          = 'PML',
                abc_jerjan_alpha:Optional[float]= 0.0053,
                nabc:Optional[int]              = 20,
                device                          = 'cpu',
                dtype                           = torch.float32
                )->None:
        super().__init__()
        # initialize the common model parameters
        self.ox             = ox
        self.oz             = oz
        self.dx             = dx
        self.dz             = dz 
        self.nx             = nx 
        self.nz             = nz
        self.free_surface   = free_surface
        self.abc_type       = abc_type
        self.abc_jerjan_alpha = abc_jerjan_alpha
        self.nabc           = nabc
        self.device         = device
        self.dtype          = dtype
        
        assert self.dx == self.dz, "Model grid size dx and dz must be the same"
        
        # set derived model parameters 
        self.x = np.arange(nx)*self.dx + self.ox
        self.z = np.arange(nz)*self.dz + self.oz
        
        # 
        
        # initialize some empty model and associated parameters
        self.pars           = []
        self.requires_grad  = {}
        self.lower_bound    = {}
        self.upper_bound    = {}
    
    def __repr__(self) -> str:
        """Representation of the model object

        Returns
        -------
        repr (str) : Representation of the model object
        """

        info = f"model with parameters {self.pars}:\n"
        for par in self.pars:
            par_min = self.get_model(par).min()
            par_max = self.get_model(par).max()
            requires_grad = self.requires_grad[par]
            lower_bound = self.lower_bound[par]
            upper_bound = self.upper_bound[par]
            info += (
                f"  Model {par:4s}: {par_min:8.2f} - {par_max:8.2f} {units[par]:6s}, "
                f"requires_grad = {requires_grad}, "
                f"constrain bound: {lower_bound} - {upper_bound}\n"
            )

        info += f"  Model orig: ox = {self.ox:6.2f}, oz = {self.oz:6.2f} m\n"
        info += f"  Model grid: dx = {self.dx:6.2f}, dz = {self.dz:6.2f} m\n"
        info += f"  Model dims: nx = {self.nx:6d}, nz = {self.nz:6d}\n"
        info += f"  Model size: {self.nx * self.nz * len(self.pars)}\n"
        info += f"  Free surface: {self.free_surface}\n"
        info += f"  Absorbing layers: {self.nabc}\n"

        return info
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """ Forward method of the model class that outputs the elastic 
        parameters of lambda, mu, and buoyancy required for the wave equation
        propogator.
        """
        raise NotImplementedError("Forward method must be implemented by the subclass")

    def _check_bounds(self):
        """Check the provided model bounds are legal
        """
        for par in self.pars:
            if self.lower_bound[par] is not None and self.upper_bound[par] is not None:
                assert (
                    self.lower_bound[par] < self.upper_bound[par]
                ), "Lower bound must be smaller than upper bound"

            if self.lower_bound[par] is not None:
                if self.lower_bound[par] + eps > self.get_model(par).min():
                    Warning(f"Lower bound must be larger than minimum value, set to {self.get_model(par).min()}")
                    self.lower_bound[par] = self.get_model(par).min() - eps

            if self.upper_bound[par] is not None:
                if self.upper_bound[par] - eps < self.get_model(par).max():
                    Warning(f"Upper bound must be smaller than maximum value, set to {self.get_model(par).max()}")
                    self.upper_bound[par] = self.get_model(par).max() + eps
    
    def check_dims(self) -> None:
        """Check the provided model dimensions are legal
        """
        for par in self.pars:
            assert (
                self.get_model(par).shape == (self.nz, self.nx)
            ), "Model dimensions must be (nz, nx)"
        return

    def get_model(self, par: str) -> np.ndarray:
        """Return the model as numpy array

        Parameters
        ----------
        par (str) : Model parameter name

        Returns
        -------
        data (np.ndarray) : Model array with shape (nz, nx)
        """
        if par not in self.pars:
            raise ValueError(f"Parameter {par} not in model")
        
        data = getattr(self, par)
        
        data = gpu2cpu(data.clone())

        return data
    
    def set_model(self, par: str, model:Optional[Union[np.array,Tensor]]) -> None:
        """Set the model as nn.Parameter

        Parameters
        ----------
        par (str) : Model parameter name
        model (np.ndarray) : Model array with shape (nz, nx)
        """

        if par not in self.pars:
            raise ValueError(f"Parameter {par} not in model")

        if model.shape != (self.nz, self.nx):
            raise ValueError("Model dimensions must be (nz, nx)")
        model = gpu2cpu(model)
        model = numpy2tensor(model)
        setattr(self, par, 
                torch.nn.Parameter(model.to(torch.float32), 
                requires_grad=self.requires_grad[par]))
    
    def get_bound(self, par: str) -> Tuple[float, float]:
        """Return the bound of the model

        Parameters
        ----------
        par (str) : Model parameter name

        Returns
        -------
        bound (Tuple) : Bound of the model, i.e., (min_value, max_value)
        """

        if par not in self.pars:
            raise ValueError("Parameter {} not in model".format(par))

        m_min = self.lower_bound[par]
        m_max = self.upper_bound[par]

        if m_min is None or m_max is None:
            return [None, None]

        return [m_min, m_max]

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
        
        return self.requires_grad[par]

    def get_grad(self, par: str) -> np.ndarray:
        """Return the gradient of the model as numpy array

        Parameters
        ----------
        par (str) : Model parameter name

        Returns
        -------
        grad (np.ndarray) : Gradient of the model with shape (nz, nx)
        """
        if par not in self.pars:
            raise ValueError("Parameter {} not in model".format(par))

        # access the model parameter
        m = getattr(self, par)

        if m.grad is None:
            return np.zeros(m.shape)
        else:
            return m.grad.cpu().detach().numpy()

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
    
    def clip_params(self)->None:
        """Clip the model parameters to the given bounds
        """
        for par in self.pars:
            if self.lower_bound[par] is not None and self.upper_bound[par] is not None:
                m = getattr(self,par)
                min_value = self.lower_bound[par]
                max_value = self.upper_bound[par]
                # clip the model parametrs
                m.data.clamp_(min_value,max_value)
        return
    
    def constrain_range(self, value, min_value, max_value) -> Tensor:
        """Constrain the value to the range [min_value, max_value] by using
        the torch.clamp function

        Parameters
        ----------
        value (Tensor)      : Value to be constrained
        min_value (float)   : Minimum value
        max_value (float)   : Maximum value

        Returns
        -------
        value_const (Tensor) : Constrained value
        """

        if min_value is not None and max_value is not None:
            # return torch.clamp(value, min_value, max_value)

            if torch.isnan(value).any():
                # replace nan with the mean value
                value[torch.isnan(value)] = max_value # value_const[~torch.isnan(value_const)].mean()
            
            value = torch.clamp(value, min_value, max_value)
            # print(value.min(), value.max())           
            # Using the following scheme occurs NaN sometimes
            # value = torch.logit((value - min_value) / (max_value - min_value))
            # value = (torch.sigmoid(value) * (max_value - min_value) + min_value)

            if torch.isinf(value).any():
                raise ValueError("Value is inf")
            
            if (value < 0.0).any():
                raise ValueError("Value is negative")

            if torch.isnan(value).any():
                raise ValueError("Value is nan")

            return value

        else:
            return value