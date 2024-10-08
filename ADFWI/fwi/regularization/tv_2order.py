'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-05-08 16:07:17
* LastEditors: LiuFeng
* LastEditTime: 2024-05-08 16:07:25
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
from .base import Regularization,regular_StepLR
import torch
import numpy as np
from ADFWI.utils import numpy2tensor

class TV_2order(Regularization):
    """2nd-Order Tikhonov Regularization
        math: 
            ||alphax*L1 m_x + alphaz*L1 m_z||_2 
        Du, Z., et al., 2021. A high-order total-variation regularisation method for full-waveform inversion. Journal of Geophysics and Engineering, 18, 241–252. doi:10.1093/jge/gxab010
    """
    def __init__(self,nx,nz,dx,dz,alphax,alphaz,step_size=1000,gamma=1) -> None:
        super().__init__(nx,nz,dx,dz,alphax,alphaz,step_size,gamma)
        
    def forward(self,m):
        nz,nx  = self.nz,self.nx
        dz,dx  = self.dz/1000,self.dx/1000 # unit: km
        device = m.device
        
        # vertical constraint
        L0 = np.diag(1*np.ones(nz-1),1) + np.diag(-2*np.ones(nz)) + np.diag(1*np.ones(nz-1),-1)
        L0[0,:]  = 0
        L0[-1,:] = 0
        L0 = numpy2tensor(L0,dtype=torch.float32).to(device)
        m_norm_z = torch.zeros((nz,nx))
        for i in range(nx):
            m_norm_z[:,i] = torch.matmul(L0,m[:,i])/dz

        # horizontal constraint
        L1 = np.diag(1*np.ones(nx-1),1) + np.diag(1*np.ones(nx-1),-1) + np.diag(-2*np.ones(nx))
        L1[0,:]  = 0
        L1[-1,:] = 0
        L1 = numpy2tensor(L1,dtype=torch.float32).to(device)
        m_norm_x = torch.zeros((nz,nx))
        for i in range(nz):
            m_norm_x[i,:] = torch.matmul(L1,m[i,:])/dx
        
        # update the alpha
        alphax = regular_StepLR(self.iter,self.step_size,self.alphax,self.gamma)
        alphaz = regular_StepLR(self.iter,self.step_size,self.alphaz,self.gamma)
        
        # misfit
        # misfit_norm = _l1_norm(alphax*m_norm_x) + _l1_norm(alphaz*m_norm_z)
        misfit_norm = torch.sum(alphax*torch.abs(m_norm_x) + alphaz*torch.abs(m_norm_z))
        return misfit_norm