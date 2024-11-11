'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-05-05 19:51:52
* LastEditors: LiuFeng
* LastEditTime: 2024-05-05 20:29:15
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

from .base import Regularization,regular_StepLR
import torch
import numpy as np
from ADFWI.utils import numpy2tensor

class Tikhonov_1order(Regularization):
    """1-Order Tikhonov Regularization
        math:
            alphax*||L_1 m_x||_p + alphaz*||L_1 m_z||_p 
        Du, Z., et al., 2021. A high-order total-variation regularisation method for full-waveform inversion. Journal of Geophysics and Engineering, 18, 241–252. doi:10.1093/jge/gxab010
    """
    def __init__(self,nx,nz,dx,dz,alphax=0,alphaz=0,step_size=1000,gamma=1,device='cpu',dtype=torch.float32) -> None:
        super().__init__(nx,nz,dx,dz,alphax,alphaz,step_size,gamma,device,dtype)
        
        # vertical constraint
        L0 = np.diag(1*np.ones(nz),0) + np.diag(-1*np.ones(nz-1),1)
        L0[-1,:] = 0
        self.L0 = numpy2tensor(L0,dtype=torch.float32).to(device)
        
        # horizontal constraint
        L1 = np.diag(1*np.ones(nx),0) + np.diag(-1*np.ones(nx-1),1)
        L1[-1,:] = 0
        self.L1 = numpy2tensor(L1,dtype=torch.float32).to(device)
        
    def forward(self,m):
        dz,dx  = self.dz/1000,self.dx/1000 # unit: km

        # vertical constraint
        m_norm_z = torch.matmul(self.L0, m) / dz

        # horizontal constraint
        m_norm_x = torch.matmul(self.L1, m.T).T / dx
        
        # update the alpha
        alphax = regular_StepLR(self.iter,self.step_size,self.alphax,self.gamma)
        alphaz = regular_StepLR(self.iter,self.step_size,self.alphaz,self.gamma)
        
        # misfit
        # misfit_norm = _l2_norm(alphax*m_norm_x) + _l2_norm(alphaz*m_norm_z)
        misfit_norm = torch.sqrt(torch.sum((alphax*m_norm_x*m_norm_x + alphaz*m_norm_z*m_norm_z)))
        
        self.iter += 1
        return misfit_norm