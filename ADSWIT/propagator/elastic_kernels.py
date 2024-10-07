'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-04-17 21:24:18
* LastEditors: LiuFeng
* LastEditTime: 2024-05-15 18:50:16
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
import time
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional,List

##########################################################################
#                          FD Coefficient     
##########################################################################
def DiffCoef(order,grid_scheme):
    C = np.zeros((1,order))
    if grid_scheme == 'r':
        B = np.insert(np.zeros(order-1),0,1/2).tolist()
        A = np.zeros((order,order))
        for i in range(1,order+1):
            for j in range(1,order+1):
                A[i-1,j-1] = j**(2*i-1)
    elif grid_scheme == 's':
        B = np.insert(np.zeros(order-1),0,1).tolist()
        A = np.zeros((order,order))
        for i in range(1,order+1):
            for j in range(1,order+1):
                A[i-1,j-1] = (2*j-1)**(2*i-1)
    C = np.linalg.inv(A) @ B
    return C

##########################################################################
#                          FD Operator     
##########################################################################
def DiffOperate(M):
    """Finite Difference Operator
    """
    NN = M//2
    fdc = DiffCoef(NN,'s')
    if M==4:
        Dxfm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj+1]-a[:,ii,:][:,:,jj])    +   fdc[1]*(a[:,ii,:][:,:,jj+2]-a[:,ii,:][:,:,jj-1])
        Dzfm = lambda a,ii,jj:fdc[0]*(a[:,ii+1,:][:,:,jj]-a[:,ii,:][:,:,jj])    +   fdc[1]*(a[:,ii+2,:][:,:,jj]-a[:,ii-1,:][:,:,jj])
        Dxbm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj]-a[:,ii,:][:,:,jj-1])    +   fdc[1]*(a[:,ii,:][:,:,jj+1]-a[:,ii,:][:,:,jj-2])
        Dzbm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj]-a[:,ii-1,:][:,:,jj])    +   fdc[1]*(a[:,ii+1,:][:,:,jj]-a[:,ii-2,:][:,:,jj])
    elif M == 6:
        Dxfm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj+1]-a[:,ii,:][:,:,jj])    +   fdc[1]*(a[:,ii,:][:,:,jj+2]-a[:,ii,:][:,:,jj-1]) \
                                + fdc[2]*(a[:,ii,:][:,:,jj+3]-a[:,ii,:][:,:,jj-2])
        Dzfm = lambda a,ii,jj:fdc[0]*(a[:,ii+1,:][:,:,jj]-a[:,ii,:][:,:,jj])    +   fdc[1]*(a[:,ii+2,:][:,:,jj]-a[:,ii-1,:][:,:,jj]) \
                                + fdc[2]*(a[:,ii+3,:][:,:,jj]-a[:,ii-2,:][:,:,jj])
        Dxbm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj]-a[:,ii,:][:,:,jj-1])    +   fdc[1]*(a[:,ii,:][:,:,jj+1]-a[:,ii,:][:,:,jj-2]) \
                                + fdc[2]*(a[:,ii,:][:,:,jj+2]-a[:,ii,:][:,:,jj-3])
        Dzbm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj]-a[:,ii-1,:][:,:,jj])    +   fdc[1]*(a[:,ii+1,:][:,:,jj]-a[:,ii-2,:][:,:,jj]) \
                                + fdc[2]*(a[:,ii+2,:][:,:,jj]-a[:,ii-3,:][:,:,jj])
    elif M == 8:
        Dxfm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj+1]-a[:,ii,:][:,:,jj])    +   fdc[1]*(a[:,ii,:][:,:,jj+2]-a[:,ii,:][:,:,jj-1]) \
                                + fdc[2]*(a[:,ii,:][:,:,jj+3]-a[:,ii,:][:,:,jj-2])  +   fdc[3]*(a[:,ii,:][:,:,jj+4]-a[:,ii,:][:,:,jj-3])
        Dzfm = lambda a,ii,jj:fdc[0]*(a[:,ii+1,:][:,:,jj]-a[:,ii,:][:,:,jj])    +   fdc[1]*(a[:,ii+2,:][:,:,jj]-a[:,ii-1,:][:,:,jj]) \
                                + fdc[2]*(a[:,ii+3,:][:,:,jj]-a[:,ii-2,:][:,:,jj])  +   fdc[3]*(a[:,ii+4,:][:,:,jj]-a[:,ii-3,:][:,:,jj])
        Dxbm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj]-a[:,ii,:][:,:,jj-1])    +   fdc[1]*(a[:,ii,:][:,:,jj+1]-a[:,ii,:][:,:,jj-2]) \
                                + fdc[2]*(a[:,ii,:][:,:,jj+2]-a[:,ii,:][:,:,jj-3])  +   fdc[3]*(a[:,ii,:][:,:,jj+3]-a[:,ii,:][:,:,jj-4])
        Dzbm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj]-a[:,ii-1,:][:,:,jj])    +   fdc[1]*(a[:,ii+1,:][:,:,jj]-a[:,ii-2,:][:,:,jj]) \
                                + fdc[2]*(a[:,ii+2,:][:,:,jj]-a[:,ii-3,:][:,:,jj])  +   fdc[3]*(a[:,ii+3,:][:,:,jj]-a[:,ii-4,:][:,:,jj])
    elif M == 10:
        Dxfm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj+1]-a[:,ii,:][:,:,jj])    +   fdc[1]*(a[:,ii,:][:,:,jj+2]-a[:,ii,:][:,:,jj-1]) \
                                + fdc[2]*(a[:,ii,:][:,:,jj+3]-a[:,ii,:][:,:,jj-2])  +   fdc[3]*(a[:,ii,:][:,:,jj+4]-a[:,ii,:][:,:,jj-3]) \
                                + fdc[4]*(a[:,ii,:][:,:,jj+5]-a[:,ii,:][:,:,jj-4])
        Dzfm = lambda a,ii,jj:fdc[0]*(a[:,ii+1,:][:,:,jj]-a[:,ii,:][:,:,jj])    +   fdc[1]*(a[:,ii+2,:][:,:,jj]-a[:,ii-1,:][:,:,jj]) \
                                + fdc[2]*(a[:,ii+3,:][:,:,jj]-a[:,ii-2,:][:,:,jj])  +   fdc[3]*(a[:,ii+4,:][:,:,jj]-a[:,ii-3,:][:,:,jj]) \
                                + fdc[4]*(a[:,ii+5,:][:,:,jj]-a[:,ii-4,:][:,:,jj])
        Dxbm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj]-a[:,ii,:][:,:,jj-1])    +   fdc[1]*(a[:,ii,:][:,:,jj+1]-a[:,ii,:][:,:,jj-2]) \
                                + fdc[2]*(a[:,ii,:][:,:,jj+2]-a[:,ii,:][:,:,jj-3])  +   fdc[3]*(a[:,ii,:][:,:,jj+3]-a[:,ii,:][:,:,jj-4]) \
                                + fdc[4]*(a[:,ii,:][:,:,jj+4]-a[:,ii,:][:,:,jj-5])
        Dzbm = lambda a,ii,jj:fdc[0]*(a[:,ii,:][:,:,jj]-a[:,ii-1,:][:,:,jj])    +   fdc[1]*(a[:,ii+1,:][:,:,jj]-a[:,ii-2,:][:,:,jj]) \
                                + fdc[2]*(a[:,ii+2,:][:,:,jj]-a[:,ii-3,:][:,:,jj])  +   fdc[3]*(a[:,ii+3,:][:,:,jj]-a[:,ii-4,:][:,:,jj]) \
                                + fdc[4]*(a[:,ii+4,:][:,:,jj]-a[:,ii-5,:][:,:,jj])
    return Dxfm,Dzfm,Dxbm,Dzbm
    
##########################################################################
#                          FD padding Coefficient     
##########################################################################
def pad_torchSingle(data,pml,fs_offset,free_surface=True,device='cpu'):
    data = data.clone()
    nz,nx = data.shape
    if free_surface:
        nx_pml = nx + 2*pml
        nz_pml = nz + pml   + fs_offset
    else:
        nx_pml = nx + 2*pml
        nz_pml = nz + 2*pml + fs_offset
    cc = torch.zeros((nz_pml,nx_pml)).to(device)
    if free_surface:
        cc[fs_offset:fs_offset+nz,pml:pml+nx]         = data
    else:
        cc[fs_offset+pml:fs_offset+pml+nz,pml:pml+nx] = data
    
    with torch.no_grad():
        if free_surface:
            cc[list(range(fs_offset))    ,pml:pml+nx] = torch.ones((fs_offset,nx)).to(device)*cc[[fs_offset],pml:pml+nx]             # top
        else:
            cc[list(range(fs_offset+pml)),pml:pml+nx] = torch.ones((fs_offset+pml,nx)).to(device)*cc[[fs_offset+pml],pml:pml+nx]     # top
        # padding
        cc[list(range(nz_pml-pml,nz_pml)),pml:pml+nx] = torch.ones((pml,nx)).to(device)*cc[[nz_pml-pml-1],pml:pml+nx] # bottom
        cc[:,list(range(0,pml))] = cc[:,[pml]]  #left
        cc[:,list(range(nx_pml-pml,nx_pml))] = cc[:,[nx_pml-pml-1]] # right
    return cc


##########################################################################
#                        step forward Modeling    
##########################################################################
def step_forward_PML(M:int,
                free_surface:bool,nx:int,nz:int,dx:float,dz:float,nabc:int,                 # basic settings
                src_x:np.array,src_z:np.array,src_n:int,dt:float,src_v:Tensor,MT:Tensor,    # source
                rcv_x:np.array,rcv_z:np.array,rcv_n:int,                                    # receiver
                bcx:Tensor,bcz:Tensor,                                                      # absobing bondary condition
                lam:Tensor,lamu:Tensor,
                C11:Tensor,C13:Tensor,C15:Tensor,C33:Tensor,C35:Tensor,C55:Tensor,          # elastic moduli parameters
                bx:Tensor,bz:Tensor,                                                        
                txx_x:Tensor,txx_z:Tensor,tzz_x:Tensor,tzz_z:Tensor,txz_x:Tensor,txz_z:Tensor,txx:Tensor,tzz:Tensor,txz:Tensor, # intermedia variable
                vx_x:Tensor,vx_z:Tensor,vz_x:Tensor,vz_z:Tensor,vx:Tensor,vz:Tensor,
                device="cpu",dtype=torch.float32
                ):
    """
    Description:
    --------------
        forward simulation within one time step
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Grid arrangement%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                                                                                           %
        %                        txx,tzz__________ vx ___________ txx,tzz------>                    %
        %                lbd,mu  |                        bh                            |           %
        %                             |                         |                             |     %
        %                             |                         |                             |     %
        %                             |                         |                             |     %
        %                        vz  |____________txz                           |                   %
        %                       bv  |                       muvh                        |           %
        %                             |                                                        |    %
        %                             |                                                        |    %
        %                             |                                                        |    %
        %                 txx,tzz  |____________________________|                                   %
        %                          |                                                                %
        %                         \|/                                                               %
        %                                                                                           %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Grid arrangement%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Prameters:
    --------------
        M (int)                         : order of the finite difference
        free_surface (bool)             : free-surface or not
        nx (int)                        : grids number along the X-axis
        nz (int)                        : grids number along the Z-axis
        dx (float)                      : grid spacing along the X-axis
        dz (float)                      : grid spacing along the Z-axis
        nabc (int)                      : the layer number of absolute bounrary condition
        src_x (ndarray)                 : source location in the X-axis
        src_z (ndarray)                 : source location in the Z-axis
        src_n (ndarray)                 : the number of the source
        dt (float)                      : time spacing (unit:s)
        src_v (Tensor)                  : wavelets for each source
        MT (Tensor)                     : moment tensor for each source
        rcv_x (ndarray)                 : receiver location in the X-axis
        rcv_z (ndarray)                 : receiver location in the Z-axis
        rcv_n (ndarray)                 : the number of the receiver
        bcx (Tensor)                    : boundary condition along the X-axis
        bcz (Tensor)                    : boundary condition along the Z-axis
        C11 (Tensor)                    : elastic moduli
        C13 (Tensor)                    : elastic moduli
        C14 (Tensor)                    : elastic moduli
        C33 (Tensor)                    : elastic moduli
        C35 (Tensor)                    : elastic moduli
        C55 (Tensor)                    : elastic moduli
        bx (Tensor)                     : 1/density along X-axis
        bz (Tensor)                     : 1/density along Z-axis
        txx_x (Tensor)                  : Stress Component : txx along X-axis
        txx_z (Tensor)                  : Stress Component : txx along Z-axis
        tzz_x (Tensor)                  : Stress Component : tzz along X-axis
        tzz_z (Tensor)                  : Stress Component : tzz along Z-axis
        txz_x (Tensor)                  : Stress Component : txz along X-axis
        txz_z (Tensor)                  : Stress Component : txz along Z-axis
        txx (Tensor)                    : Stress Component : txx
        tzz (Tensor)                    : Stress Component : tzz 
        txz (Tensor)                    : Stress Component : txz 
        vx_x (Tensor)                   : velocity Component : vx along X-axis 
        vx_z (Tensor)                   : velocity Component : vx along Z-axis
        vz_x (Tensor)                   : velocity Component : vz along X-axis 
        vz_z (Tensor)                   : velocity Component : vz along Z-axis 
        vx (Tensor)                     : velocity Component : vx 
        vz (Tensor)                     : velocity Component : vz
    
    returns:
    ------------------
        txx_x (Tensor)                  : Stress Component : txx along X-axis
        txx_z (Tensor)                  : Stress Component : txx along Z-axis
        tzz_x (Tensor)                  : Stress Component : tzz along X-axis
        tzz_z (Tensor)                  : Stress Component : tzz along Z-axis
        txz_x (Tensor)                  : Stress Component : txz along X-axis
        txz_z (Tensor)                  : Stress Component : txz along Z-axis
        txx (Tensor)                    : Stress Component : txx
        tzz (Tensor)                    : Stress Component : tzz 
        txz (Tensor)                    : Stress Component : txz 
        vx_x (Tensor)                   : velocity Component : vx along X-axis 
        vx_z (Tensor)                   : velocity Component : vx along Z-axis
        vz_x (Tensor)                   : velocity Component : vz along X-axis 
        vz_z (Tensor)                   : velocity Component : vz along Z-axis 
        vx (Tensor)                     : velocity Component : vx 
        vz (Tensor)                     : velocity Component : vz
        rcv_txx (Tensor)                : recorded txx on the receivers
        rcv_tzz (Tensor)                : recorded tzz on the receivers
        rcv_txz (Tensor)                : recorded txz on the receivers
        rcv_vx (Tensor)                 : recorded vx on the receivers
        rcv_vz (Tensor)                 : recorded vz on the receivers
        forward_wavefield_txx (Tensor)  : forward wavefield of txx
        forward_wavefield_tzz (Tensor)  : forward wavefield of tzz
        forward_wavefield_txz (Tensor)  : forward wavefield of txz
        forward_wavefield_vx (Tensor)   : forward wavefield of vx
        forward_wavefield_vz (Tensor)   : forward wavefield of vz
    """
    nt = src_v.shape[-1]
    # free surface offset
    fs_offset = M//2
    
    # forward simulation
    if free_surface:
        nx_pml = nx + 2*nabc
        nz_pml = nz +   nabc + fs_offset
    else:
        nx_pml = nx + 2*nabc
        nz_pml = nz + 2*nabc + fs_offset
    
    # copy the data
    vx,vz,txx,tzz,txz           = torch.ones_like(vx)*vx,    torch.ones_like(vz)*vz,    torch.ones_like(txx)*txx,    torch.ones_like(tzz)*tzz,    torch.ones_like(txz)*txz
    vx_x,vz_x,txx_x,tzz_x,txz_x = torch.ones_like(vx_x)*vx_x,torch.ones_like(vz_x)*vz_x,torch.ones_like(txx_x)*txx_x,torch.ones_like(tzz_x)*tzz_x,torch.ones_like(txz_x)*txz_x
    vx_z,vz_z,txx_z,tzz_z,txz_z = torch.ones_like(vx_z)*vx_z,torch.ones_like(vz_z)*vz_z,torch.ones_like(txx_z)*txx_z,torch.ones_like(tzz_z)*tzz_z,torch.ones_like(txz_z)*txz_z
    
    # recorded waveform
    rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz = torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device)
    forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz = torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device)
    forward_wavefield_vx,forward_wavefield_vz  = torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device)
    
    # Finite difference Order
    NN = M//2
    h = NN+1
    ii = np.arange(NN,nz_pml-NN)        # z-axis
    jj = np.arange(NN,nx_pml-NN)        # x-axis
    i_start = NN 
    i_end   = nz_pml-NN-1
    j_start = NN 
    j_end   = nx_pml-NN-1
    
    Dxfm,Dzfm,Dxbm,Dzbm = DiffOperate(M)
    
    # damping
    pmlxd = 1 + 0.5*dt*bcx[i_start:i_end+1,j_start:j_end+1]
    pmlxn = 1 - 0.5*dt*bcx[i_start:i_end+1,j_start:j_end+1]
    pmlzd = 1 + 0.5*dt*bcz[i_start:i_end+1,j_start:j_end+1]
    pmlzn = 1 - 0.5*dt*bcz[i_start:i_end+1,j_start:j_end+1]
    
    # moment tensor source implementation
    for t in range(nt):
        # stress component 
        dxbm_vx = Dxbm(vx,ii,jj)
        dxbm_vz = Dxbm(vz,ii,jj)
        dzbm_vx = Dzbm(vx,ii,jj)
        dzbm_vz = Dzbm(vz,ii,jj)
        dxfm_vx = Dxfm(vx,ii,jj)
        dzfm_vx = Dzfm(vx,ii,jj)
        dxfm_vz = Dxfm(vz,ii,jj)
        dzfm_vz = Dzfm(vz,ii,jj)
        txx_x[:,i_start:i_end+1,j_start:j_end+1] = (pmlxn*txx_x[:,i_start:i_end+1,j_start:j_end+1] + dt*(C11[i_start:i_end+1,j_start:j_end+1]*dxbm_vx+C15[i_start:i_end+1,j_start:j_end+1]*dxbm_vz)/dx)/pmlxd
        txx_z[:,i_start:i_end+1,j_start:j_end+1] = (pmlzn*txx_z[:,i_start:i_end+1,j_start:j_end+1] + dt*(C15[i_start:i_end+1,j_start:j_end+1]*dzbm_vx+C13[i_start:i_end+1,j_start:j_end+1]*dzbm_vz)/dz)/pmlzd
        tzz_x[:,i_start:i_end+1,j_start:j_end+1] = (pmlxn*tzz_x[:,i_start:i_end+1,j_start:j_end+1] + dt*(C13[i_start:i_end+1,j_start:j_end+1]*dxbm_vx+C35[i_start:i_end+1,j_start:j_end+1]*dxbm_vz)/dx)/pmlxd
        tzz_z[:,i_start:i_end+1,j_start:j_end+1] = (pmlzn*tzz_z[:,i_start:i_end+1,j_start:j_end+1] + dt*(C35[i_start:i_end+1,j_start:j_end+1]*dzbm_vx+C33[i_start:i_end+1,j_start:j_end+1]*dzbm_vz)/dz)/pmlzd
        txz_x[:,i_start:i_end+1,j_start:j_end+1] = (pmlxn*txz_x[:,i_start:i_end+1,j_start:j_end+1] + dt*(C15[i_start:i_end+1,j_start:j_end+1]*dxfm_vx+C55[i_start:i_end+1,j_start:j_end+1]*dxfm_vz)/dx)/pmlxd
        txz_z[:,i_start:i_end+1,j_start:j_end+1] = (pmlzn*txz_z[:,i_start:i_end+1,j_start:j_end+1] + dt*(C55[i_start:i_end+1,j_start:j_end+1]*dzfm_vx+C35[i_start:i_end+1,j_start:j_end+1]*dzfm_vz)/dz)/pmlzd
        # txx[:] = txx_x+txx_z
        # tzz[:] = tzz_x+tzz_z
        # txz[:] = txz_x+txz_z
        
        # add source
        if len(src_v.shape) == 1:
            txx_x[np.arange(src_n),src_z,src_x] = txx_x[np.arange(src_n),src_z,src_x] + (-MT[0,0]/2)*src_v[t]
            txx_z[np.arange(src_n),src_z,src_x] = txx_z[np.arange(src_n),src_z,src_x] + (-MT[0,0]/2)*src_v[t]
            tzz_x[np.arange(src_n),src_z,src_x] = tzz_x[np.arange(src_n),src_z,src_x] + (-MT[2,2]/2)*src_v[t]
            tzz_z[np.arange(src_n),src_z,src_x] = tzz_z[np.arange(src_n),src_z,src_x] + (-MT[2,2]/2)*src_v[t]
            txz_x[np.arange(src_n),src_z,src_x] = txz_x[np.arange(src_n),src_z,src_x] + (-MT[0,2]/2)*src_v[t]
            txz_z[np.arange(src_n),src_z,src_x] = txz_z[np.arange(src_n),src_z,src_x] + (-MT[0,2]/2)*src_v[t]
        else:
            txx_x[np.arange(src_n),src_z,src_x] = txx_x[np.arange(src_n),src_z,src_x] + (-MT[np.arange(src_n),0,0]/2)*src_v[np.arange(src_n),t]
            txx_z[np.arange(src_n),src_z,src_x] = txx_z[np.arange(src_n),src_z,src_x] + (-MT[np.arange(src_n),0,0]/2)*src_v[np.arange(src_n),t]
            tzz_x[np.arange(src_n),src_z,src_x] = tzz_x[np.arange(src_n),src_z,src_x] + (-MT[np.arange(src_n),2,2]/2)*src_v[np.arange(src_n),t]
            tzz_z[np.arange(src_n),src_z,src_x] = tzz_z[np.arange(src_n),src_z,src_x] + (-MT[np.arange(src_n),2,2]/2)*src_v[np.arange(src_n),t]
            txz_x[np.arange(src_n),src_z,src_x] = txz_x[np.arange(src_n),src_z,src_x] + (-MT[np.arange(src_n),0,2]/2)*src_v[np.arange(src_n),t]
            txz_z[np.arange(src_n),src_z,src_x] = txz_z[np.arange(src_n),src_z,src_x] + (-MT[np.arange(src_n),0,2]/2)*src_v[np.arange(src_n),t]
        txx[:] = txx_x + txx_z
        tzz[:] = tzz_x + tzz_z
        txz[:] = txz_x + txz_z
        
        # topFs with the assumption of weak anisotropy near the surface
        if free_surface:
            tzz[:,h-1,:] = 0
            tzz[:,h-2,:] = -tzz[:,h,:]
            txz[:,h-2,:] = -txz[:,h-1,:]
            txz[:,h-3,:] = -txz[:,h,:]
        
        # velociyt component
        vx_x[:,i_start:i_end+1,j_start:j_end+1] = (pmlxn*vx_x[:,i_start:i_end+1,j_start:j_end+1] + dt*bx[i_start:i_end+1,j_start:j_end+1]*Dxfm(txx,ii,jj)/dx)/pmlxd
        vx_z[:,i_start:i_end+1,j_start:j_end+1] = (pmlzn*vx_z[:,i_start:i_end+1,j_start:j_end+1] + dt*bx[i_start:i_end+1,j_start:j_end+1]*Dzbm(txz,ii,jj)/dz)/pmlzd
        vz_x[:,i_start:i_end+1,j_start:j_end+1] = (pmlxn*vz_x[:,i_start:i_end+1,j_start:j_end+1] + dt*bz[i_start:i_end+1,j_start:j_end+1]*Dxbm(txz,ii,jj)/dx)/pmlxd
        vz_z[:,i_start:i_end+1,j_start:j_end+1] = (pmlzn*vz_z[:,i_start:i_end+1,j_start:j_end+1] + dt*bz[i_start:i_end+1,j_start:j_end+1]*Dzfm(tzz,ii,jj)/dz)/pmlzd
        vx[:] = vx_x + vx_z
        vz[:] = vz_x + vz_z

        # topFS with the assumption of weak anisotropy near the surface
        if free_surface:
            # vz[:,h-2,j_start:j_end+1] = vz[:,h-1,j_start:j_end+1]   + (lam[h-1,j_start:j_end+1]/lamu[h-1,j_start:j_end+1])*(vx[:,h-1,j_start:j_end+1]-vx[:,h-1,j_start-1:j_end])
            # vx[:,h-2,j_start:j_end+1] = vz[:,h-2,j_start+1:j_end+2] - vz[:,h-2,j_start:j_end+1] + vz[:,h-1,j_start+1:j_end+2] - vz[:,h-1,j_start:j_end+1] + vx[:,h,j_start:j_end+1]
            # vz[:,h-3,j_start:j_end+1] = vz[:,h-2,j_start:j_end+1]   + (lam[h-1,j_start:j_end+1]/lamu[h-1,j_start:j_end+1])*(vx[:,h-2,j_start:j_end+1]-vx[:,h-2,j_start-1:j_end]) 
            vz[:,h-2,j_start:j_end+1] = vz[:,h-1,j_start:j_end+1]
            vx[:,h-2,j_start:j_end+1] = vz[:,h-2,j_start+1:j_end+2] - vz[:,h-2,j_start:j_end+1] + vz[:,h-1,j_start+1:j_end+2] - vz[:,h-1,j_start:j_end+1] + vx[:,h,j_start:j_end+1]
            vz[:,h-3,j_start:j_end+1] = vz[:,h-2,j_start:j_end+1]
        
        # -----------------------------------------------------------
        #                   Receiver Observation
        # -----------------------------------------------------------
        rcv_txx[:,t,list(range(rcv_n))] = txx[:,rcv_z,rcv_x]
        rcv_tzz[:,t,list(range(rcv_n))] = tzz[:,rcv_z,rcv_x]
        rcv_txz[:,t,list(range(rcv_n))] = txz[:,rcv_z,rcv_x]
        rcv_vx[:,t,list(range(rcv_n))]  =  vx[:,rcv_z,rcv_x]
        rcv_vz[:,t,list(range(rcv_n))]  =  vz[:,rcv_z,rcv_x]
        
        with torch.no_grad():
            if free_surface:
                forward_wavefield_txx = torch.sum(txx*txx,dim=0)[fs_offset:fs_offset+nz,nabc:nabc+nx]
                forward_wavefield_tzz = torch.sum(tzz*tzz,dim=0)[fs_offset:fs_offset+nz,nabc:nabc+nx]
                forward_wavefield_txz = torch.sum(txz*txz,dim=0)[fs_offset:fs_offset+nz,nabc:nabc+nx]
                forward_wavefield_vx  =   torch.sum(vx*vx,dim=0)[fs_offset:fs_offset+nz,nabc:nabc+nx]
                forward_wavefield_vz  =   torch.sum(vz*vz,dim=0)[fs_offset:fs_offset+nz,nabc:nabc+nx]
            else:
                forward_wavefield_txx = torch.sum(txx*txx,dim=0)[fs_offset+nabc:fs_offset+nabc+nz,nabc:nabc+nx]
                forward_wavefield_tzz = torch.sum(tzz*tzz,dim=0)[fs_offset+nabc:fs_offset+nabc+nz,nabc:nabc+nx]
                forward_wavefield_txz = torch.sum(txz*txz,dim=0)[fs_offset+nabc:fs_offset+nabc+nz,nabc:nabc+nx]
                forward_wavefield_vx  =   torch.sum(vx*vx,dim=0)[fs_offset+nabc:fs_offset+nabc+nz,nabc:nabc+nx]
                forward_wavefield_vz  =   torch.sum(vz*vz,dim=0)[fs_offset+nabc:fs_offset+nabc+nz,nabc:nabc+nx]
    return txx_x,txx_z,tzz_x,tzz_z,txz_x,txz_z,txx,tzz,txz,vx_x,vx_z,vz_x,vz_z,vx,vz,\
            rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz,\
            forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz


##########################################################################
#                step forward Modeling :damping mode   
##########################################################################
def step_forward_ABL(M:int,
                free_surface:bool,nx:int,nz:int,dx:float,dz:float,nabc:int,                 # basic settings
                src_x:np.array,src_z:np.array,src_n:int,dt:float,src_v:Tensor,MT:Tensor,    # source
                rcv_x:np.array,rcv_z:np.array,rcv_n:int,                                    # receiver
                damp:Tensor,                                                                # absobing bondary condition
                lam:Tensor,lamu:Tensor,
                C11:Tensor,C13:Tensor,C15:Tensor,C33:Tensor,C35:Tensor,C55:Tensor,          # elastic moduli parameters
                bx:Tensor,bz:Tensor,                                                        
                txx:Tensor,tzz:Tensor,txz:Tensor,                                           # intermedia variable
                vx:Tensor,vz:Tensor,
                device="cpu",dtype=torch.float32
                ):
    """
    Description:
    --------------
        forward simulation within one time step
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Grid arrangement%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                                                                                           %
        %                        txx,tzz__________ vx ___________ txx,tzz------>                    %
        %                lbd,mu  |                        bh                            |           %
        %                             |                         |                             |     %
        %                             |                         |                             |     %
        %                             |                         |                             |     %
        %                        vz  |____________txz                           |                   %
        %                       bv  |                       muvh                        |           %
        %                             |                                                        |    %
        %                             |                                                        |    %
        %                             |                                                        |    %
        %                 txx,tzz  |____________________________|                                   %
        %                          |                                                                %
        %                         \|/                                                               %
        %                                                                                           %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Grid arrangement%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Prameters:
    --------------
        M (int)                         : order of the finite difference
        free_surface (bool)             : free-surface or not
        nx (int)                        : grids number along the X-axis
        nz (int)                        : grids number along the Z-axis
        dx (float)                      : grid spacing along the X-axis
        dz (float)                      : grid spacing along the Z-axis
        nabc (int)                      : the layer number of absolute bounrary condition
        src_x (ndarray)                 : source location in the X-axis
        src_z (ndarray)                 : source location in the Z-axis
        src_n (ndarray)                 : the number of the source
        dt (float)                      : time spacing (unit:s)
        src_v (Tensor)                  : wavelets for each source
        MT (Tensor)                     : moment tensor for each source
        rcv_x (ndarray)                 : receiver location in the X-axis
        rcv_z (ndarray)                 : receiver location in the Z-axis
        rcv_n (ndarray)                 : the number of the receiver
        damp (Tensor)                   : boundary condition along both X and Z axis
        C11 (Tensor)                    : elastic moduli
        C13 (Tensor)                    : elastic moduli
        C14 (Tensor)                    : elastic moduli
        C33 (Tensor)                    : elastic moduli
        C35 (Tensor)                    : elastic moduli
        C55 (Tensor)                    : elastic moduli
        bx (Tensor)                     : 1/density along X-axis
        bz (Tensor)                     : 1/density along Z-axis
        txx (Tensor)                    : Stress Component : txx
        tzz (Tensor)                    : Stress Component : tzz 
        txz (Tensor)                    : Stress Component : txz 
        vx (Tensor)                     : velocity Component : vx 
        vz (Tensor)                     : velocity Component : vz
    
    returns:
    ------------------
        txx_x (Tensor)                  : Stress Component : txx along X-axis
        txx_z (Tensor)                  : Stress Component : txx along Z-axis
        tzz_x (Tensor)                  : Stress Component : tzz along X-axis
        tzz_z (Tensor)                  : Stress Component : tzz along Z-axis
        txz_x (Tensor)                  : Stress Component : txz along X-axis
        txz_z (Tensor)                  : Stress Component : txz along Z-axis
        txx (Tensor)                    : Stress Component : txx
        tzz (Tensor)                    : Stress Component : tzz 
        txz (Tensor)                    : Stress Component : txz 
        vx_x (Tensor)                   : velocity Component : vx along X-axis 
        vx_z (Tensor)                   : velocity Component : vx along Z-axis
        vz_x (Tensor)                   : velocity Component : vz along X-axis 
        vz_z (Tensor)                   : velocity Component : vz along Z-axis 
        vx (Tensor)                     : velocity Component : vx 
        vz (Tensor)                     : velocity Component : vz
        rcv_txx (Tensor)                : recorded txx on the receivers
        rcv_tzz (Tensor)                : recorded tzz on the receivers
        rcv_txz (Tensor)                : recorded txz on the receivers
        rcv_vx (Tensor)                 : recorded vx on the receivers
        rcv_vz (Tensor)                 : recorded vz on the receivers
        forward_wavefield_txx (Tensor)  : forward wavefield of txx
        forward_wavefield_tzz (Tensor)  : forward wavefield of tzz
        forward_wavefield_txz (Tensor)  : forward wavefield of txz
        forward_wavefield_vx (Tensor)   : forward wavefield of vx
        forward_wavefield_vz (Tensor)   : forward wavefield of vz
    """
    nt = src_v.shape[-1]
    # free surface offset
    fs_offset = M//2
    
    # forward simulation
    if free_surface:
        nx_pml = nx + 2*nabc
        nz_pml = nz +   nabc + fs_offset
    else:
        nx_pml = nx + 2*nabc
        nz_pml = nz + 2*nabc + fs_offset
    
    # copy the data
    vx,vz,txx,tzz,txz           = torch.ones_like(vx)*vx,    torch.ones_like(vz)*vz,    torch.ones_like(txx)*txx,    torch.ones_like(tzz)*tzz,    torch.ones_like(txz)*txz
    
    # recorded waveform
    rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz = torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device)
    forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz = torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device)
    forward_wavefield_vx,forward_wavefield_vz  = torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device)
    
    # Finite difference Order
    NN = M//2
    h = NN+1
    ii = np.arange(NN,nz_pml-NN)        # z-axis
    jj = np.arange(NN,nx_pml-NN)        # x-axis
    i_start = NN 
    i_end   = nz_pml-NN-1
    j_start = NN 
    j_end   = nx_pml-NN-1
    
    Dxfm,Dzfm,Dxbm,Dzbm = DiffOperate(M)
    
    # moment tensor source implementation
    for t in range(nt):
        # Stress Component
        dxbm_vx = Dxbm(vx,ii,jj)
        dxbm_vz = Dxbm(vz,ii,jj)
        dzbm_vx = Dzbm(vx,ii,jj)
        dzbm_vz = Dzbm(vz,ii,jj)
        dxfm_vx = Dxfm(vx,ii,jj)
        dzfm_vx = Dzfm(vx,ii,jj)
        dxfm_vz = Dxfm(vz,ii,jj)
        dzfm_vz = Dzfm(vz,ii,jj)
        txx[:,i_start:i_end+1,j_start:j_end+1] = txx[:,i_start:i_end+1,j_start:j_end+1] + dt*((C11[i_start:i_end+1,j_start:j_end+1]*dxbm_vx + C15[i_start:i_end+1,j_start:j_end+1]*dxbm_vz)/dx
                                                                                            + (C15[i_start:i_end+1,j_start:j_end+1]*dzbm_vx + C13[i_start:i_end+1,j_start:j_end+1]*dzbm_vz)/dz)
        tzz[:,i_start:i_end+1,j_start:j_end+1] = tzz[:,i_start:i_end+1,j_start:j_end+1] + dt*((C13[i_start:i_end+1,j_start:j_end+1]*dxbm_vx + C35[i_start:i_end+1,j_start:j_end+1]*dxbm_vz)/dx
                                                                                            + (C35[i_start:i_end+1,j_start:j_end+1]*dzbm_vx + C33[i_start:i_end+1,j_start:j_end+1]*dzbm_vz)/dz)
        txz[:,i_start:i_end+1,j_start:j_end+1] = txz[:,i_start:i_end+1,j_start:j_end+1] + dt*((C15[i_start:i_end+1,j_start:j_end+1]*dxfm_vx + C55[i_start:i_end+1,j_start:j_end+1]*dxfm_vz)/dx
                                                                                            + (C55[i_start:i_end+1,j_start:j_end+1]*dzfm_vx + C35[i_start:i_end+1,j_start:j_end+1]*dzfm_vz)/dz)
        
        # add source
        if len(src_v.shape) == 1:
            txx[np.arange(src_n),src_z,src_x] = txx[np.arange(src_n),src_z,src_x] + (-MT[0,0]/3)*src_v[t]
            tzz[np.arange(src_n),src_z,src_x] = tzz[np.arange(src_n),src_z,src_x] + (-MT[2,2]/3)*src_v[t]
            txz[np.arange(src_n),src_z,src_x] = txz[np.arange(src_n),src_z,src_x] + (-MT[0,2]/3)*src_v[t]
        else:
            txx[np.arange(src_n),src_z,src_x] = txx[np.arange(src_n),src_z,src_x] + (-MT[np.arange(src_n),0,0]/3)*src_v[np.arange(src_n),t]
            tzz[np.arange(src_n),src_z,src_x] = tzz[np.arange(src_n),src_z,src_x] + (-MT[np.arange(src_n),2,2]/3)*src_v[np.arange(src_n),t]
            txz[np.arange(src_n),src_z,src_x] = txz[np.arange(src_n),src_z,src_x] + (-MT[np.arange(src_n),0,2]/3)*src_v[np.arange(src_n),t]
    
        # topFs with the assumption of weak anisotropy near the surface
        if free_surface:
            tzz[:,h-1,:] = 0
            tzz[:,h-2,:] = -tzz[:,h,:]
            txz[:,h-2,:] = -txz[:,h-1,:]
            txz[:,h-3,:] = -txz[:,h,:]
        
        # velocity component
        vx[:,i_start:i_end+1,j_start:j_end+1] = vx[:,i_start:i_end+1,j_start:j_end+1] + dt*bx[i_start:i_end+1,j_start:j_end+1]*(Dxfm(txx,ii,jj)/dx + Dzbm(txz,ii,jj)/dz)
        vz[:,i_start:i_end+1,j_start:j_end+1] = vz[:,i_start:i_end+1,j_start:j_end+1] + dt*bz[i_start:i_end+1,j_start:j_end+1]*(Dxbm(txz,ii,jj)/dx + Dzfm(tzz,ii,jj)/dz)

        # topFS with the assumption of weak anisotropy near the surface
        if free_surface:
            # vz[:,h-2,j_start:j_end+1] = vz[:,h-1,j_start:j_end+1]   + (lam[h-1,j_start:j_end+1]/lamu[h-1,j_start:j_end+1])*(vx[:,h-1,j_start:j_end+1]-vx[:,h-1,j_start-1:j_end])
            # vx[:,h-2,j_start:j_end+1] = vz[:,h-2,j_start+1:j_end+2] - vz[:,h-2,j_start:j_end+1] + vz[:,h-1,j_start+1:j_end+2] - vz[:,h-1,j_start:j_end+1] + vx[:,h,j_start:j_end+1]
            # vz[:,h-3,j_start:j_end+1] = vz[:,h-2,j_start:j_end+1]   + (lam[h-1,j_start:j_end+1]/lamu[h-1,j_start:j_end+1])*(vx[:,h-2,j_start:j_end+1]-vx[:,h-2,j_start-1:j_end]) 
            vz[:,h-2,j_start:j_end+1] = vz[:,h-1,j_start:j_end+1]
            vx[:,h-2,j_start:j_end+1] = vz[:,h-2,j_start+1:j_end+2] - vz[:,h-2,j_start:j_end+1] + vz[:,h-1,j_start+1:j_end+2] - vz[:,h-1,j_start:j_end+1] + vx[:,h,j_start:j_end+1]
            vz[:,h-3,j_start:j_end+1] = vz[:,h-2,j_start:j_end+1]

        vx = damp*vx
        vz = damp*vz
            
        rcv_txx[:,t,list(range(rcv_n))] = txx[:,rcv_z,rcv_x]
        rcv_tzz[:,t,list(range(rcv_n))] = tzz[:,rcv_z,rcv_x]
        rcv_txz[:,t,list(range(rcv_n))] = txz[:,rcv_z,rcv_x]
        rcv_vx[:,t,list(range(rcv_n))]  =  vx[:,rcv_z,rcv_x]
        rcv_vz[:,t,list(range(rcv_n))]  =  vz[:,rcv_z,rcv_x]
        
        with torch.no_grad():
            if free_surface:
                forward_wavefield_txx = torch.sum(txx*txx,dim=0)[fs_offset:fs_offset+nz,nabc:nabc+nx].detach()
                forward_wavefield_tzz = torch.sum(tzz*tzz,dim=0)[fs_offset:fs_offset+nz,nabc:nabc+nx].detach()
                forward_wavefield_txz = torch.sum(txz*txz,dim=0)[fs_offset:fs_offset+nz,nabc:nabc+nx].detach()
                forward_wavefield_vx  =   torch.sum(vx*vx,dim=0)[fs_offset:fs_offset+nz,nabc:nabc+nx].detach()
                forward_wavefield_vz  =   torch.sum(vz*vz,dim=0)[fs_offset:fs_offset+nz,nabc:nabc+nx].detach()
            else:
                forward_wavefield_txx = torch.sum(txx*txx,dim=0)[fs_offset+nabc:fs_offset+nabc+nz,nabc:nabc+nx].detach()
                forward_wavefield_tzz = torch.sum(tzz*tzz,dim=0)[fs_offset+nabc:fs_offset+nabc+nz,nabc:nabc+nx].detach()
                forward_wavefield_txz = torch.sum(txz*txz,dim=0)[fs_offset+nabc:fs_offset+nabc+nz,nabc:nabc+nx].detach()
                forward_wavefield_vx  =   torch.sum(vx*vx,dim=0)[fs_offset+nabc:fs_offset+nabc+nz,nabc:nabc+nx].detach()
                forward_wavefield_vz  =   torch.sum(vz*vz,dim=0)[fs_offset+nabc:fs_offset+nabc+nz,nabc:nabc+nx].detach()
        
    return txx,tzz,txz,vx,vz,rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz,\
            forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz



##########################################################################
#                       forward Modeling    
##########################################################################
def forward_kernel( nx:int,nz:int,dx:float,dz:float,nt:int,dt:float,
                    nabc:int,free_surface:bool,                                         # Model settings
                    src_x:np.array,src_z:np.array,src_n:int,src_v:Tensor,MT:Tensor,     # Source
                    rcv_x:np.array,rcv_z:np.array,rcv_n:int,                            # Receiver
                    abc_type:str,bcx:Tensor,bcz:Tensor,damp:Tensor,                     # PML/ABL
                    lamu:Tensor,lam:Tensor,bx:Tensor,bz:Tensor,                         # lame constant
                    CC:List[Tensor],                                                    # elastic moduli
                    fd_order=4,n_segments = 1,                                          # Finite Difference
                    device='cpu',dtype=torch.float32
                ):
    """ Forward simulation of Elastic Waveform Equation

    Prameters:
    --------------
        nx (int)                        : grid number along the X-axis
        nz (int)                        : grid number along the Z-axis
        dx (float)                      : grid spacing along the X-axis
        dz (float)                      : grid spacing along the Z-axis
        nt (int)                        : number of time points for recording waveforms 
        dt (float)                      : time spacing (unit:s)
        nabc (int)                      : number of absorbing boundary condition
        free_surface (bool)             : free-surface or not
        src_x (ndarray)                 : source location in the X-axis
        src_z (ndarray)                 : source location in the Z-axis
        src_n (ndarray)                 : the number of the source
        src_v (Tensor)                  : wavelets for each source
        MT (Tensor)                     : moment tensor for each source
        rcv_x (ndarray)                 : receiver location in the X-axis
        rcv_z (ndarray)                 : receiver location in the Z-axis
        rcv_n (ndarray)                 : the number of the receiver
        bcx (Tensor)                    : boundary condition along the X-axis
        bcz (Tensor)                    : boundary condition along the Z-axis
        damp (Tensor)                   : boundary condition along both the X and Z-axis
        abc_type (str)                  : boundary condition types: ABL or PML
        lamu (Tensor)                   : Lame parameters : rho*vp^2
        lam (Tensor)                    : Lame parameters : rho*vp^2 - 2*rho*vs^2
        bx (Tensor)                     : 1/density in X-axis
        bz (Tensor)                     : 1/density in Z-axis
        CC (list of Tensors)            : 21 elastic moduli
        fd_order (int)                  : order of the finite difference
        checkpoint_segments             : segments of the checkpoints for saving memory
        device (str)                    : device, Default "cpu"
        dtype (types)                   : dtypes, Default torch.float32
    
    Returns
    ---------------
        record_waveforms (dict)
            txx (Tensors)                   : recorded txx on the receivers                         
            tzz (Tensors)                   : recorded txx on the receivers
            txz (Tensors)                   : recorded txx on the receivers
            vx (Tensors)                    : recorded txx on the receivers
            vz (Tensors)                    : recorded txx on the receivers
            forward_wavefield_txx (Tensor)  : forward wavefield of txx
            forward_wavefield_tzz (Tensor)  : forward wavefield of tzz
            forward_wavefield_txz (Tensor)  : forward wavefield of txz
            forward_wavefield_vx (Tensor)   : forward wavefield of vx
            forward_wavefield_vz (Tensor)   : forward wavefield of vz
    """
    # free surface offset
    fs_offset = fd_order//2
    
    # forward simulation
    if free_surface:
        nx_pml = nx + 2*nabc
        nz_pml = nz +   nabc + fs_offset
    else:
        nx_pml = nx + 2*nabc
        nz_pml = nz + 2*nabc + fs_offset
    
    # padding input parameter
    lamu = pad_torchSingle(lamu,nabc,fs_offset,free_surface,device)
    lam  = pad_torchSingle(lam ,nabc,fs_offset,free_surface,device)
    bx   = pad_torchSingle(bx  ,nabc,fs_offset,free_surface,device)
    bz   = pad_torchSingle(bz  ,nabc,fs_offset,free_surface,device)
    
    [C11,_,C13,_,C15,_,_,_,_,_,_,C33,_,C35,_,_,_,_,C55,_,_] = CC
    C11 = pad_torchSingle(C11,nabc,fs_offset,free_surface,device)
    C13 = pad_torchSingle(C13,nabc,fs_offset,free_surface,device)
    C15 = pad_torchSingle(C15,nabc,fs_offset,free_surface,device)
    C33 = pad_torchSingle(C33,nabc,fs_offset,free_surface,device)
    C35 = pad_torchSingle(C35,nabc,fs_offset,free_surface,device)
    C55 = pad_torchSingle(C55,nabc,fs_offset,free_surface,device)

    # padding the absorbing boundary condition
    if abc_type.lower() in ["pml"]:
        bcx = pad_torchSingle(bcx,0,fs_offset,free_surface,device)
        bcz = pad_torchSingle(bcz,0,fs_offset,free_surface,device)
    else:
        damp = pad_torchSingle(damp,0,fs_offset,free_surface,device)

    # padding source and receiver
    src_x = src_x + nabc
    src_z = src_z + fs_offset if free_surface else src_z+nabc + fs_offset
    rcv_x = rcv_x + nabc
    rcv_z = rcv_z + fs_offset if free_surface else rcv_z+nabc + fs_offset
    
    # some intermediate variables
    vx,vz,txx,tzz,txz           = torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device)
    vx_x,vz_x,txx_x,tzz_x,txz_x = torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device)
    vx_z,vz_z,txx_z,tzz_z,txz_z = torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype).to(device)
    rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz = torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device)

    forward_wavefield_txx = torch.zeros((nz,nx),dtype=dtype).to(device)
    forward_wavefield_tzz = torch.zeros((nz,nx),dtype=dtype).to(device)
    forward_wavefield_txz = torch.zeros((nz,nx),dtype=dtype).to(device)
    forward_wavefield_vx  = torch.zeros((nz,nx),dtype=dtype).to(device)
    forward_wavefield_vz  = torch.zeros((nz,nx),dtype=dtype).to(device)
    
    # checkpoints for saving memory
    k = 0
    for i, chunk in enumerate(torch.chunk(src_v,n_segments,dim=-1)):
        if abc_type.lower() in ["pml"]:
            txx_x,txx_z,tzz_x,tzz_z,txz_x,txz_z,txx,tzz,txz,\
            vx_x,vx_z,vz_x,vz_z,vx,vz,\
            rcv_txx_temp,rcv_tzz_temp,rcv_txz_temp,rcv_vx_temp,rcv_vz_temp,\
            forward_wavefield_txx_temp,forward_wavefield_tzz_temp,forward_wavefield_txz_temp,forward_wavefield_vx_temp,forward_wavefield_vz_temp \
                                                                            = checkpoint(step_forward_PML,
                                                                                fd_order,
                                                                                free_surface,nx,nz,dx,dz,nabc,
                                                                                src_x,src_z,src_n,dt,chunk,MT,
                                                                                rcv_x,rcv_z,rcv_n,
                                                                                bcx,bcz,
                                                                                lam,lamu,
                                                                                C11,C13,C15,C33,C35,C55,
                                                                                bx,bz,
                                                                                txx_x,txx_z,tzz_x,tzz_z,txz_x,txz_z,txx,tzz,txz,
                                                                                vx_x,vx_z,vz_x,vz_z,vx,vz,
                                                                                device,dtype)
        else:
            txx,tzz,txz,vx,vz,\
            rcv_txx_temp,rcv_tzz_temp,rcv_txz_temp,rcv_vx_temp,rcv_vz_temp,\
            forward_wavefield_txx_temp,forward_wavefield_tzz_temp,forward_wavefield_txz_temp,forward_wavefield_vx_temp,forward_wavefield_vz_temp \
                                                                            = checkpoint(step_forward_ABL,
                                                                                fd_order,
                                                                                free_surface,nx,nz,dx,dz,nabc,
                                                                                src_x,src_z,src_n,dt,chunk,MT,
                                                                                rcv_x,rcv_z,rcv_n,
                                                                                damp,
                                                                                lam,lamu,
                                                                                C11,C13,C15,C33,C35,C55,
                                                                                bx,bz,
                                                                                txx,tzz,txz,
                                                                                vx,vz,
                                                                                device,dtype)
        # save the waveform recorded on receiver
        rcv_txx[:,k:k+chunk.shape[-1]] = rcv_txx_temp
        rcv_tzz[:,k:k+chunk.shape[-1]] = rcv_tzz_temp
        rcv_txz[:,k:k+chunk.shape[-1]] = rcv_txz_temp
        rcv_vx[:,k:k+chunk.shape[-1]]  =  rcv_vx_temp
        rcv_vz[:,k:k+chunk.shape[-1]]  =  rcv_vz_temp
        
        # save the forward wavefield
        with torch.no_grad():
            forward_wavefield_txx += forward_wavefield_txx_temp
            forward_wavefield_tzz += forward_wavefield_tzz_temp
            forward_wavefield_txz += forward_wavefield_txz_temp
            forward_wavefield_vx  += forward_wavefield_vx_temp
            forward_wavefield_vz  += forward_wavefield_vz_temp
        k+=chunk.shape[-1]

    record_waveform = {
        "txx":rcv_txx,
        "tzz":rcv_tzz,
        "txz":rcv_txz,
        "vx" :rcv_vx,
        "vz" :rcv_vz,
        "forward_wavefield_txx":forward_wavefield_txx,
        "forward_wavefield_tzz":forward_wavefield_tzz,
        "forward_wavefield_txz":forward_wavefield_txz,
        "forward_wavefield_vx":forward_wavefield_vx,
        "forward_wavefield_vz":forward_wavefield_vz,
    }
    return record_waveform