"""
* Modified from: https://github.com/seisfwi/SWIT
* Original Author: Haipeng Li
* Original Author Email: haipeng@stanford.edu
=========================================================
* Author: Liu Feng (SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: Liu Feng
* LastEditTime: 2024-05-15 19:30:38
* Description: 
* Copyright (c) 2024 by Liu Feng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
"""

import numpy as np
from typing import List, Optional
import numpy as np
from ADFWI.utils import list2numpy,numpy2list
from ADFWI.view import plot_wavelet

class Source(object):
    """Seismic Source class
    """
    def __init__(self,nt:int,dt:float,f0:float)->None:
        self.nt             = nt
        self.dt             = dt
        self.f0             = f0
        self.t              = np.arange(nt)*dt
        self.loc_x          = []
        self.loc_z          = []
        self.loc            = []
        self.type           = []
        self.wavelet        = []
        self.moment_tensor  = []
        self.num            = 0
    

    def __repr__(self):
        """Reimplement the repr function for printing the source information"""
        try:
            src_x = list2numpy(self.loc_x)
            src_z = list2numpy(self.loc_z)
            xmin = src_x.min()
            xmax = src_x.max()
            zmin = src_z.min()
            zmax = src_z.max()

            info = f"Seismic Source:\n"
            info += f"  Source wavelet: {self.nt} samples at {self.dt * 1000:.2f} ms\n"
            info += f"  Source number : {self.num}\n"
            info += f"  Source types  : {self.get_type(unique = True)}\n"
            info += f"  Source x range: {xmin} - {xmax} (grids)\n"
            info += f"  Source z range: {zmin} - {zmax} (grids)\n"
        except:
            info = f"Seismic Source:\n"
            info += f"  empty\n"
        return info
    
    def add_encoded_sources(self,
            src_x       : np.array,
            src_z       : np.array,
            src_wavelet : np.ndarray, #3D: [encode_n,[src_n,t]]
            src_type    : Optional[str]='mt',
            src_mt      : Optional[np.ndarray] = np.array([[1,0,0],[0,1,0],[0,0,1]]), 
        ) -> None:
        """source encoded
        """
        if src_x.shape != src_z.shape:
            raise ValueError(
                "Source location along x and z direction must have the same shape"
            )
        if src_type.lower() not in ["mt"]:
            raise ValueError(
                "Source type must be either mt"
            )
        if src_wavelet.shape[-1] != self.nt:
            raise ValueError(
                "Source wavelet must have the same length as the number of time samples"
            )
        if src_mt.shape != (3, 3):
            raise ValueError("Moment tensor must be a 3x3 matrix")

        if src_type.lower() == "mt" and src_mt is None:
            raise ValueError("Moment tensor must be provided for mt source")
        src_n = len(src_x)
        # add source
        self.loc_x.extend(numpy2list(src_x))
        self.loc_z.extend(numpy2list(src_z))
        self.type.extend([src_type]*src_n)
        self.wavelet.extend(src_wavelet)
        self.moment_tensor.extend(np.ones((src_n,3,3))*src_mt)
        self.num += src_n
        return
    
    def add_sources(self,
            src_x       : np.array,
            src_z       : np.array,
            src_wavelet : np.ndarray,
            src_type    : Optional[str]='mt',
            src_mt      : Optional[np.ndarray] = np.array([[1,0,0],[0,1,0],[0,0,1]]), 
        ) -> None:
        """add multiple sources with same wavelet
        """
        if src_x.shape != src_z.shape:
            raise ValueError(
                "Source location along x and z direction must have the same shape"
            )
        if src_type.lower() not in ["mt"]:
            raise ValueError(
                "Source type must be either mt"
            )
        if src_wavelet.shape[0] != self.nt:
            raise ValueError(
                "Source wavelet must have the same length as the number of time samples"
            )
        if src_mt.shape != (3, 3):
            raise ValueError("Moment tensor must be a 3x3 matrix")

        if src_type.lower() == "mt" and src_mt is None:
            raise ValueError("Moment tensor must be provided for mt source")
        src_n = len(src_x)
        # add source
        self.loc_x.extend(numpy2list(src_x.reshape(-1)))
        self.loc_z.extend(numpy2list(src_z.reshape(-1)))
        self.type.extend([src_type]*src_n)
        self.wavelet.extend(np.ones((src_n,self.nt))*src_wavelet)
        self.moment_tensor.extend(np.ones((src_n,3,3))*src_mt)
        self.num += src_n
        return
        
    def add_source(self,
            src_x       : int,
            src_z       : int,
            src_wavelet : np.ndarray,
            src_type    : Optional[str]='mt',
            src_mt      : Optional[np.ndarray] = np.array([[1,0,0],[0,1,0],[0,0,1]]), 
        ) -> None:
        """Append single source
        """
        if src_type.lower() not in ["mt"]:
            raise ValueError(
                "Source type must be either mt"
            )
        if src_wavelet.shape[0] != self.nt:
            raise ValueError(
                "Source wavelet must have the same length as the number of time samples"
            )
        if src_mt.shape != (3, 3):
            raise ValueError("Moment tensor must be a 3x3 matrix")

        if src_type.lower() == "mt" and src_mt is None:
            raise ValueError("Moment tensor must be provided for mt source")

        # add source
        self.loc_x.append(src_x)
        self.loc_z.append(src_z)
        self.type.append(src_type)
        self.wavelet.append(src_wavelet)
        self.moment_tensor.append(src_mt)
        self.num += 1
    
    def get_loc(self):
        """Return the source location
        """
        src_x = list2numpy(self.loc_x)
        src_z = list2numpy(self.loc_z)
        if len(list2numpy(self.loc_x).shape) == 1:
            src_x   = src_x.reshape(-1,1)
            src_z   = src_z.reshape(-1, 1)
            src_loc = np.hstack((src_x,src_z))
        else:
            
            src_loc = np.concatenate((src_x[..., np.newaxis], 
                                      src_z[..., np.newaxis]), 
                                      axis=-1)  # Add new axis and concatenate
        self.loc = src_loc.copy()
        return src_loc 
    
    def get_wavelet(self):
        """Return the source wavelets
        """
        wavelet = list2numpy(self.wavelet)
        return wavelet
    
    def get_moment_tensor(self):
        """Return the source wavelets
        """
        mt = list2numpy(self.moment_tensor)
        return mt
    
    def get_type(self, unique=False) -> List[str]:
        """Return the source type
        """
        type = list2numpy(self.type)
        
        if unique:
            type = list2numpy(list(set(self.type)))
        return type
    
    def plot_wavelet(self,index=0,src_idx=None,**kwargs):
        tlist = self.t
        wavelet = self.get_wavelet()
        # for encoded source
        if len(wavelet.shape) == 3:
            wavelet = wavelet[index][src_idx]
        else:
            wavelet = wavelet[index]
        plot_wavelet(tlist,wavelet,**kwargs)