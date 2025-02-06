'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-17 21:24:18
* LastEditors: LiuFeng
* LastEditTime: 2024-05-15 18:50:16
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
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
@torch.jit.script
def DiffCoef(order: int, grid_scheme: str) -> torch.Tensor:
    """
    Calculate the differential coefficients for a given order and grid scheme.

    Parameters:
        order (int): The order of the coefficients.
        grid_scheme (str): The grid scheme, either 'r' for rational or 's' for symmetric.

    Returns:
        torch.Tensor: The calculated differential coefficients.
    """
    # Initialize the coefficient array
    C = torch.zeros(order)

    # Determine the coefficient matrix and right-hand side based on the grid scheme
    if grid_scheme == 'r':
        B = torch.cat((torch.tensor([1 / 2]), torch.zeros(order - 1)))
        A = torch.empty((order, order), dtype=torch.float32)

        for i in range(order):
            for j in range(order):
                A[i, j] = (j + 1) ** (2 * (i + 1) - 1)

    elif grid_scheme == 's':
        B = torch.cat((torch.tensor([1.0]), torch.zeros(order - 1)))
        A = torch.empty((order, order), dtype=torch.float32)

        for i in range(order):
            for j in range(order):
                A[i, j] = (2 * (j + 1) - 1) ** (2 * (i + 1) - 1)

    else:
        raise ValueError("Invalid grid scheme. Use 'r' for rational or 's' for symmetric.")

    # Calculate the differential coefficients using PyTorch
    C = torch.linalg.inv(A) @ B

    return C



##########################################################################
#                          FD Operator     
##########################################################################

@torch.jit.script
def Dxfm_4(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return fdc[0] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj]) + \
           fdc[1] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 1])
            
@torch.jit.script
def Dzfm_4(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return fdc[0] * (a[:, ii + 1, :][:, :, jj] - a[:, ii, :][:, :, jj]) + \
           fdc[1] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 1, :][:, :, jj])

@torch.jit.script
def Dxbm_4(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii, :][:, :, jj - 1]) + \
           fdc[1] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj - 2])

@torch.jit.script
def Dzbm_4(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) + \
           fdc[1] * (a[:, ii + 1, :][:, :, jj] - a[:, ii - 2, :][:, :, jj])

@torch.jit.script
def Dxfm_6(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return fdc[0] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj]) + \
           fdc[1] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 1]) + \
           fdc[2] * (a[:, ii, :][:, :, jj + 3] - a[:, ii, :][:, :, jj - 2])

@torch.jit.script
def Dzfm_6(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii + 1, :][:, :, jj] - a[:, ii, :][:, :, jj]) + \
            fdc[1] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) + \
            fdc[2] * (a[:, ii + 3, :][:, :, jj] - a[:, ii - 2, :][:, :, jj])

@torch.jit.script
def Dxbm_6(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii, :][:, :, jj - 1]) + \
            fdc[1] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj - 2]) + \
            fdc[2] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 3])

@torch.jit.script
def Dzbm_6(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) + \
            fdc[1] * (a[:, ii + 1, :][:, :, jj] - a[:, ii - 2, :][:, :, jj]) + \
            fdc[2] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 3, :][:, :, jj])

@torch.jit.script
def Dxfm_8(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj]) + \
            fdc[1] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 1]) + \
            fdc[2] * (a[:, ii, :][:, :, jj + 3] - a[:, ii, :][:, :, jj - 2]) + \
            fdc[3] * (a[:, ii, :][:, :, jj + 4] - a[:, ii, :][:, :, jj - 3])

@torch.jit.script
def Dzfm_8(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii + 1, :][:, :, jj] - a[:, ii, :][:, :, jj]) + \
            fdc[1] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) + \
            fdc[2] * (a[:, ii + 3, :][:, :, jj] - a[:, ii - 2, :][:, :, jj]) + \
            fdc[3] * (a[:, ii + 4, :][:, :, jj] - a[:, ii - 3, :][:, :, jj])

@torch.jit.script
def Dxbm_8(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii, :][:, :, jj - 1]) + \
            fdc[1] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj - 2]) + \
            fdc[2] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 3]) + \
            fdc[3] * (a[:, ii, :][:, :, jj + 3] - a[:, ii, :][:, :, jj - 4])

@torch.jit.script
def Dzbm_8(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) + \
            fdc[1] * (a[:, ii + 1, :][:, :, jj] - a[:, ii - 2, :][:, :, jj]) + \
            fdc[2] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 3, :][:, :, jj]) + \
            fdc[3] * (a[:, ii + 3, :][:, :, jj] - a[:, ii - 4, :][:, :, jj])

@torch.jit.script
def Dxfm_10(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj]) + \
            fdc[1] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 1]) + \
            fdc[2] * (a[:, ii, :][:, :, jj + 3] - a[:, ii, :][:, :, jj - 2]) + \
            fdc[3] * (a[:, ii, :][:, :, jj + 4] - a[:, ii, :][:, :, jj - 3]) + \
            fdc[4] * (a[:, ii, :][:, :, jj + 5] - a[:, ii, :][:, :, jj - 4])

@torch.jit.script
def Dzfm_10(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii + 1, :][:, :, jj] - a[:, ii, :][:, :, jj]) + \
            fdc[1] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) + \
            fdc[2] * (a[:, ii + 3, :][:, :, jj] - a[:, ii - 2, :][:, :, jj]) + \
            fdc[3] * (a[:, ii + 4, :][:, :, jj] - a[:, ii - 3, :][:, :, jj]) + \
            fdc[4] * (a[:, ii + 5, :][:, :, jj] - a[:, ii - 4, :][:, :, jj])

@torch.jit.script
def Dxbm_10(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii, :][:, :, jj - 1]) + \
            fdc[1] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj - 2]) + \
            fdc[2] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 3]) + \
            fdc[3] * (a[:, ii, :][:, :, jj + 3] - a[:, ii, :][:, :, jj - 4]) + \
            fdc[4] * (a[:, ii, :][:, :, jj + 4] - a[:, ii, :][:, :, jj - 5])

@torch.jit.script
def Dzbm_10(a: torch.Tensor, fdc: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return  fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) + \
            fdc[1] * (a[:, ii + 1, :][:, :, jj] - a[:, ii - 2, :][:, :, jj]) + \
            fdc[2] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 3, :][:, :, jj]) + \
            fdc[3] * (a[:, ii + 3, :][:, :, jj] - a[:, ii - 4, :][:, :, jj]) + \
            fdc[4] * (a[:, ii + 4, :][:, :, jj] - a[:, ii - 5, :][:, :, jj])


##########################################################################
#                          FD padding Coefficient     
##########################################################################
@torch.jit.script
def pad_torchSingle(data: torch.Tensor, pml: int, fs_offset: int, free_surface: bool = True, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Description:
        Pad a 2D tensor with specified padding layers (PML) and optional free surface condition.

    Parameters:
        - data (torch.Tensor): Input tensor to be padded, with shape (nz, nx).
        - pml (int): Number of padding layers to apply (Perfectly Matched Layer).
        - fs_offset (int): Offset for the free surface condition.
        - free_surface (bool): Flag indicating whether to apply free surface conditions. Default is True.
        - device (torch.device): Device to allocate the padded tensor on. Default is CPU.

    Returns:
        - torch.Tensor: Padded tensor with shape (nz_pml, nx_pml).
    """
    nz, nx = data.shape
    
    # Calculate new dimensions
    nx_pml = nx + 2 * pml
    nz_pml = nz + (pml + fs_offset if free_surface else 2 * pml + fs_offset)
    
    # Initialize padded tensor
    cc = torch.zeros((nz_pml, nx_pml), device=device)
    
    if free_surface:
        # Copy data to padded tensor with free surface condition
        cc[fs_offset:fs_offset + nz, pml:pml + nx] = data
        cc[:fs_offset, pml:pml + nx] = data[0, :].unsqueeze(0).expand(fs_offset, -1)  # Top padding
    else:
        # Copy data to padded tensor without free surface
        cc[fs_offset + pml:fs_offset + pml + nz, pml:pml + nx] = data
        cc[:fs_offset + pml, pml:pml + nx] = data[0, :].unsqueeze(0).expand(fs_offset + pml, -1)  # Top padding

    # Bottom padding
    cc[nz_pml - pml:, pml:pml + nx] = data[-1, :].unsqueeze(0).expand(pml, -1)
    
    # Left and right padding
    cc[:, :pml] = cc[:, pml:pml + 1].expand(-1, pml)  # Left padding
    cc[:, nx_pml - pml:] = cc[:, nx_pml - pml - 1:nx_pml - pml].expand(-1, pml)  # Right padding

    return cc

##########################################################################
#                        step forward Modeling    
##########################################################################
@torch.jit.script
def step_forward_PML_4order(M:int,
                free_surface:bool,nx:int,nz:int,dx:float,dz:float,nabc:int,                 # basic settings
                src_x:Tensor,src_z:Tensor,src_n:int,dt:float,src_v:Tensor,MT:Tensor,        # source
                rcv_x:Tensor,rcv_z:Tensor,rcv_n:int,                                        # receiver
                bcx:Tensor,bcz:Tensor,                                                      # absobing bondary condition
                lam:Tensor,lamu:Tensor,
                C11:Tensor,C13:Tensor,C15:Tensor,C33:Tensor,C35:Tensor,C55:Tensor,          # elastic moduli parameters
                bx:Tensor,bz:Tensor,                                                        
                txx_x:Tensor,txx_z:Tensor,tzz_x:Tensor,tzz_z:Tensor,txz_x:Tensor,txz_z:Tensor,txx:Tensor,tzz:Tensor,txz:Tensor, # intermedia variable
                vx_x:Tensor,vx_z:Tensor,vz_x:Tensor,vz_z:Tensor,vx:Tensor,vz:Tensor,
                device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
                ):
    """Description:
        Simulates the time-stepping of the wavefield in a 2D elastic medium using finite difference methods with Perfectly Matched Layer (PML) boundary conditions.
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
        - M (int): The finite difference order. Typically 4, 6, 8, or 10, determining the accuracy of spatial derivatives.
        - free_surface (bool): Indicates whether to impose a free surface condition at the top boundary.
        - nx (int), nz (int): Number of grid points in the x and z directions, respectively, defining the simulation domain.
        - dx (float), dz (float): Spatial step sizes in the x and z directions.
        - nabc (int): The number of grid points used for the PML absorbing boundary.
        - src_x (Tensor), src_z (Tensor), src_n (int): Source positions in x and z directions, and the number of sources.
        - dt (float): Time step size.
        - src_v (Tensor): Source time function values.
        - MT (Tensor): Moment tensor for the source mechanism.
        - rcv_x (Tensor), rcv_z (Tensor), rcv_n (int): Receiver positions in x and z directions, and the number of receivers.
        - bcx (Tensor), bcz (Tensor): Damping profiles for the absorbing boundary conditions in x and z directions.
        - lam (Tensor), lamu (Tensor): Lame parameters (for P-wave and S-wave velocities).
        - C11, C13, C15, C33, C35, C55 (Tensor): Elastic moduli for the anisotropic medium.
        - bx (Tensor), bz (Tensor): Density-related coefficients in x and z directions.
        - txx_x, txx_z, tzz_x, tzz_z, txz_x, txz_z, txx, tzz, txz (Tensor): Stress tensors and their derivatives in x and z directions.
        - vx_x, vx_z, vz_x, vz_z, vx, vz (Tensor): Velocity components and their derivatives in x and z directions.
        - device (torch.device): The computing device (CPU or GPU).
        - dtype (torch.dtype): The data type (e.g., float32) used for the computations.

    Returns:
    --------------
        A tuple containing the updated stress and velocity tensors as well as recorded receiver waveforms and forward wavefields.
    """
    nt = src_v.shape[-1]    # Number of time steps
    fs_offset = M // 2      # Finite difference stencil offset

    # PML dimensions with free surface condition
    nx_pml = nx + 2 * nabc
    nz_pml = nz + (nabc + fs_offset if free_surface else 2 * nabc + fs_offset)
    
    # Clone tensors to avoid in-place modification
    vx, vz, txx, tzz, txz           = vx.clone()    , vz.clone()    , txx.clone(), tzz.clone(), txz.clone()
    vx_x, vz_x, txx_x, tzz_x, txz_x = vx_x.clone()  , vz_x.clone()  , txx_x.clone(), tzz_x.clone(), txz_x.clone()
    vx_z, vz_z, txx_z, tzz_z, txz_z = vx_z.clone()  , vz_z.clone()  , txx_z.clone(), tzz_z.clone(), txz_z.clone()

    # Initialize recorded waveforms
    rcv_txx, rcv_tzz, rcv_txz, rcv_vx, rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    forward_wavefield_txx, forward_wavefield_tzz, forward_wavefield_txz = torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vx, forward_wavefield_vz = torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device)
    
    # Finite difference order and indexing
    NN = M // 2
    fdc = DiffCoef(NN, 's')
    h = NN + 1
    i_start = NN 
    i_end = nz_pml - NN
    j_start = NN 
    j_end = nx_pml - NN
    idx_i = slice(i_start, i_end)
    idx_j = slice(j_start, j_end)
    idx_j_plus1  = slice(j_start+1,j_end+1)
    idx_j_minus1 = slice(j_start-1,j_end-1)
    ii = torch.arange(i_start, i_end, dtype=torch.long, device=device)  # z-axis range
    jj = torch.arange(j_start, j_end, dtype=torch.long, device=device)  # x-axis range
    
    # Damping factors for PML
    pmlxd = 1 + 0.5 * dt * bcx[idx_i, idx_j]
    pmlxn = 1 - 0.5 * dt * bcx[idx_i, idx_j]
    pmlzd = 1 + 0.5 * dt * bcz[idx_i, idx_j]
    pmlzn = 1 - 0.5 * dt * bcz[idx_i, idx_j]
    
    # Precompute constants for the forward loop
    dt_dx = dt / dx
    dt_dz = dt / dz
    pmlx_inv = 1.0 / pmlxd
    pmlz_inv = 1.0 / pmlzd
    
    # Source and receiver indices
    src_idx = list(range(src_n))
    rcv_idx = list(range(rcv_n))
    half_MT = MT[src_idx] / 3  # Half of the moment tensor
    
    # Adjust offset for free surface condition
    offset = fs_offset if free_surface else fs_offset + nabc
    
    # Finite difference operators for different axes
    Dxfm = Dxfm_4
    Dzfm = Dzfm_4
    Dxbm = Dxbm_4
    Dzbm = Dzbm_4
    
    # moment tensor source implementation
    for t in range(nt):
        
        # Compute stress components
        dxbm_vx = Dxbm(vx, fdc, ii, jj)
        dxbm_vz = Dxbm(vz, fdc, ii, jj)
        dzbm_vx = Dzbm(vx, fdc, ii, jj)
        dzbm_vz = Dzbm(vz, fdc, ii, jj)
        dxfm_vx = Dxfm(vx, fdc, ii, jj)
        dzfm_vx = Dzfm(vx, fdc, ii, jj)
        dxfm_vz = Dxfm(vz, fdc, ii, jj)
        dzfm_vz = Dzfm(vz, fdc, ii, jj)

        # Update stress fields
        txx_x[:, idx_i, idx_j] = (pmlxn * txx_x[:, idx_i, idx_j] + dt_dx * (C11[idx_i, idx_j] * dxbm_vx + C15[idx_i, idx_j] * dxbm_vz)) * pmlx_inv
        txx_z[:, idx_i, idx_j] = (pmlzn * txx_z[:, idx_i, idx_j] + dt_dz * (C15[idx_i, idx_j] * dzbm_vx + C13[idx_i, idx_j] * dzbm_vz)) * pmlz_inv
        tzz_x[:, idx_i, idx_j] = (pmlxn * tzz_x[:, idx_i, idx_j] + dt_dx * (C13[idx_i, idx_j] * dxbm_vx + C35[idx_i, idx_j] * dxbm_vz)) * pmlx_inv
        tzz_z[:, idx_i, idx_j] = (pmlzn * tzz_z[:, idx_i, idx_j] + dt_dz * (C35[idx_i, idx_j] * dzbm_vx + C33[idx_i, idx_j] * dzbm_vz)) * pmlz_inv
        txz_x[:, idx_i, idx_j] = (pmlxn * txz_x[:, idx_i, idx_j] + dt_dx * (C15[idx_i, idx_j] * dxfm_vx + C55[idx_i, idx_j] * dxfm_vz)) * pmlx_inv
        txz_z[:, idx_i, idx_j] = (pmlzn * txz_z[:, idx_i, idx_j] + dt_dz * (C55[idx_i, idx_j] * dzfm_vx + C35[idx_i, idx_j] * dzfm_vz)) * pmlz_inv

        # Add moment tensor source
        if len(src_v.shape) == 1:
            txx_x[src_idx, src_z, src_x] += -half_MT[0, 0] * src_v[t]
            txx_z[src_idx, src_z, src_x] += -half_MT[0, 0] * src_v[t]
            tzz_x[src_idx, src_z, src_x] += -half_MT[2, 2] * src_v[t]
            tzz_z[src_idx, src_z, src_x] += -half_MT[2, 2] * src_v[t]
            txz_x[src_idx, src_z, src_x] += -half_MT[0, 2] * src_v[t]
            txz_z[src_idx, src_z, src_x] += -half_MT[0, 2] * src_v[t]
        else:
            txx_x[src_idx, src_z, src_x] += -half_MT[:, 0, 0] * src_v[src_idx, t]
            txx_z[src_idx, src_z, src_x] += -half_MT[:, 0, 0] * src_v[src_idx, t]
            tzz_x[src_idx, src_z, src_x] += -half_MT[:, 2, 2] * src_v[src_idx, t]
            tzz_z[src_idx, src_z, src_x] += -half_MT[:, 2, 2] * src_v[src_idx, t]
            txz_x[src_idx, src_z, src_x] += -half_MT[:, 0, 2] * src_v[src_idx, t]
            txz_z[src_idx, src_z, src_x] += -half_MT[:, 0, 2] * src_v[src_idx, t]

        # Combine x and z components of stress
        txx[:] = txx_x + txx_z
        tzz[:] = tzz_x + tzz_z
        txz[:] = txz_x + txz_z

        # Apply free surface boundary conditions
        if free_surface:
            tzz[:, h-1, :] = 0
            tzz[:, h-2, :] = -tzz[:, h, :]
            txz[:, h-2, :] = -txz[:, h-1, :]
            txz[:, h-3, :] = -txz[:, h, :]

        # Compute velocity components
        dxfm_txx = Dxfm(txx, fdc, ii, jj)
        dzbm_txz = Dzbm(txz, fdc, ii, jj)
        dxbm_txz = Dxbm(txz, fdc, ii, jj)
        dzfm_tzz = Dzfm(tzz, fdc, ii, jj)
        vx_x[:, idx_i, idx_j] = (pmlxn * vx_x[:, idx_i, idx_j] + dt * bx[idx_i, idx_j] * dxfm_txx / dx) / pmlxd
        vx_z[:, idx_i, idx_j] = (pmlzn * vx_z[:, idx_i, idx_j] + dt * bx[idx_i, idx_j] * dzbm_txz / dz) / pmlzd
        vz_x[:, idx_i, idx_j] = (pmlxn * vz_x[:, idx_i, idx_j] + dt * bz[idx_i, idx_j] * dxbm_txz / dx) / pmlxd
        vz_z[:, idx_i, idx_j] = (pmlzn * vz_z[:, idx_i, idx_j] + dt * bz[idx_i, idx_j] * dzfm_tzz / dz) / pmlzd
        vx[:] = vx_x + vx_z
        vz[:] = vz_x + vz_z

        # Apply free surface boundary conditions for velocity
        # if free_surface:
        #     vz[:, h-2, idx_j] = vz[:, h-1, idx_j]       + (lam[h-1, idx_j]/lamu[h-1, idx_j])*(vx[:, h-1, idx_j] - vx[:, h-1, idx_j_minus1])
        #     vx[:, h-2, idx_j] = vz[:, h-2, idx_j_plus1] - vz[:, h-2, idx_j] + vz[:, h-1, idx_j_plus1] - vz[:, h-1, idx_j] + vx[:, h, idx_j]
        #     vz[:, h-3, idx_j] = vz[:, h-2, idx_j]       + (lam[h-1, idx_j]/lamu[h-1, idx_j])*(vx[:, h-2, idx_j] - vx[:, h-2, idx_j_minus1])
        
        if free_surface:
            vz[:, h-2, idx_j] = vz[:, h-1, idx_j]
            vx[:, h-2, idx_j] = vz[:, h-2, idx_j_plus1] - vz[:, h-2, idx_j] + vz[:, h-1, idx_j_plus1] - vz[:, h-1, idx_j] + vx[:, h, idx_j]
            vz[:, h-3, idx_j] = vz[:, h-2, idx_j]

        # -----------------------------------------------------------
        #                   Receiver Observation
        # -----------------------------------------------------------
        rcv_txx[:, t, rcv_idx] = txx[:, rcv_z, rcv_x]
        rcv_tzz[:, t, rcv_idx] = tzz[:, rcv_z, rcv_x]
        rcv_txz[:, t, rcv_idx] = txz[:, rcv_z, rcv_x]
        rcv_vx[:, t, rcv_idx] = vx[:, rcv_z, rcv_x]
        rcv_vz[:, t, rcv_idx] = vz[:, rcv_z, rcv_x]

        # Store forward wavefields for visualization or further processing
        forward_wavefield_txx += torch.sum(txx * txx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_tzz += torch.sum(tzz * tzz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_txz += torch.sum(txz * txz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vx  += torch.sum(vx * vx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vz  += torch.sum(vz * vz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()

    return txx_x,txx_z,tzz_x,tzz_z,txz_x,txz_z,txx,tzz,txz,\
            vx_x,vx_z,vz_x,vz_z,vx,vz,\
            rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz,\
            forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz

@torch.jit.script
def step_forward_PML_6order(M:int,
                free_surface:bool,nx:int,nz:int,dx:float,dz:float,nabc:int,                 # basic settings
                src_x:Tensor,src_z:Tensor,src_n:int,dt:float,src_v:Tensor,MT:Tensor,        # source
                rcv_x:Tensor,rcv_z:Tensor,rcv_n:int,                                        # receiver
                bcx:Tensor,bcz:Tensor,                                                      # absobing bondary condition
                lam:Tensor,lamu:Tensor,
                C11:Tensor,C13:Tensor,C15:Tensor,C33:Tensor,C35:Tensor,C55:Tensor,          # elastic moduli parameters
                bx:Tensor,bz:Tensor,                                                        
                txx_x:Tensor,txx_z:Tensor,tzz_x:Tensor,tzz_z:Tensor,txz_x:Tensor,txz_z:Tensor,txx:Tensor,tzz:Tensor,txz:Tensor, # intermedia variable
                vx_x:Tensor,vx_z:Tensor,vz_x:Tensor,vz_z:Tensor,vx:Tensor,vz:Tensor,
                device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
                ):
    nt = src_v.shape[-1]    # Number of time steps
    fs_offset = M // 2      # Finite difference stencil offset

    # PML dimensions with free surface condition
    nx_pml = nx + 2 * nabc
    nz_pml = nz + (nabc + fs_offset if free_surface else 2 * nabc + fs_offset)
    
    # Clone tensors to avoid in-place modification
    vx, vz, txx, tzz, txz           = vx.clone()    , vz.clone()    , txx.clone(), tzz.clone(), txz.clone()
    vx_x, vz_x, txx_x, tzz_x, txz_x = vx_x.clone()  , vz_x.clone()  , txx_x.clone(), tzz_x.clone(), txz_x.clone()
    vx_z, vz_z, txx_z, tzz_z, txz_z = vx_z.clone()  , vz_z.clone()  , txx_z.clone(), tzz_z.clone(), txz_z.clone()

    # Initialize recorded waveforms
    rcv_txx, rcv_tzz, rcv_txz, rcv_vx, rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    forward_wavefield_txx, forward_wavefield_tzz, forward_wavefield_txz = torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vx, forward_wavefield_vz = torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device)
    
    # Finite difference order and indexing
    NN = M // 2
    fdc = DiffCoef(NN, 's')
    h = NN + 1
    i_start = NN 
    i_end = nz_pml - NN
    j_start = NN 
    j_end = nx_pml - NN
    idx_i = slice(i_start, i_end)
    idx_j = slice(j_start, j_end)
    idx_j_plus1 = slice(j_start+1,j_end+1)
    idx_j_minus1 = slice(j_start-1,j_end-1)
    ii = torch.arange(i_start, i_end, dtype=torch.long, device=device)  # z-axis range
    jj = torch.arange(j_start, j_end, dtype=torch.long, device=device)  # x-axis range
    
    # Damping factors for PML
    pmlxd = 1 + 0.5 * dt * bcx[idx_i, idx_j]
    pmlxn = 1 - 0.5 * dt * bcx[idx_i, idx_j]
    pmlzd = 1 + 0.5 * dt * bcz[idx_i, idx_j]
    pmlzn = 1 - 0.5 * dt * bcz[idx_i, idx_j]
    
    # Precompute constants for the forward loop
    dt_dx = dt / dx
    dt_dz = dt / dz
    pmlx_inv = 1.0 / pmlxd
    pmlz_inv = 1.0 / pmlzd
    
    # Source and receiver indices
    src_idx = list(range(src_n))
    rcv_idx = list(range(rcv_n))
    half_MT = MT[src_idx] / 2  # Half of the moment tensor
    
    # Adjust offset for free surface condition
    offset = fs_offset if free_surface else fs_offset + nabc
    
    # Finite difference operators for different axes
    Dxfm = Dxfm_6
    Dzfm = Dzfm_6
    Dxbm = Dxbm_6
    Dzbm = Dzbm_6
    
    # moment tensor source implementation
    for t in range(nt):
        
        # Compute stress components
        dxbm_vx = Dxbm(vx, fdc, ii, jj)
        dxbm_vz = Dxbm(vz, fdc, ii, jj)
        dzbm_vx = Dzbm(vx, fdc, ii, jj)
        dzbm_vz = Dzbm(vz, fdc, ii, jj)
        dxfm_vx = Dxfm(vx, fdc, ii, jj)
        dzfm_vx = Dzfm(vx, fdc, ii, jj)
        dxfm_vz = Dxfm(vz, fdc, ii, jj)
        dzfm_vz = Dzfm(vz, fdc, ii, jj)

        # Update stress fields
        txx_x[:, idx_i, idx_j] = (pmlxn * txx_x[:, idx_i, idx_j] + dt_dx * (C11[idx_i, idx_j] * dxbm_vx + C15[idx_i, idx_j] * dxbm_vz)) * pmlx_inv
        txx_z[:, idx_i, idx_j] = (pmlzn * txx_z[:, idx_i, idx_j] + dt_dz * (C15[idx_i, idx_j] * dzbm_vx + C13[idx_i, idx_j] * dzbm_vz)) * pmlz_inv
        tzz_x[:, idx_i, idx_j] = (pmlxn * tzz_x[:, idx_i, idx_j] + dt_dx * (C13[idx_i, idx_j] * dxbm_vx + C35[idx_i, idx_j] * dxbm_vz)) * pmlx_inv
        tzz_z[:, idx_i, idx_j] = (pmlzn * tzz_z[:, idx_i, idx_j] + dt_dz * (C35[idx_i, idx_j] * dzbm_vx + C33[idx_i, idx_j] * dzbm_vz)) * pmlz_inv
        txz_x[:, idx_i, idx_j] = (pmlxn * txz_x[:, idx_i, idx_j] + dt_dx * (C15[idx_i, idx_j] * dxfm_vx + C55[idx_i, idx_j] * dxfm_vz)) * pmlx_inv
        txz_z[:, idx_i, idx_j] = (pmlzn * txz_z[:, idx_i, idx_j] + dt_dz * (C55[idx_i, idx_j] * dzfm_vx + C35[idx_i, idx_j] * dzfm_vz)) * pmlz_inv

        # Add moment tensor source
        if len(src_v.shape) == 1:
            txx_x[src_idx, src_z, src_x] += -half_MT[0, 0] * src_v[t]
            txx_z[src_idx, src_z, src_x] += -half_MT[0, 0] * src_v[t]
            tzz_x[src_idx, src_z, src_x] += -half_MT[2, 2] * src_v[t]
            tzz_z[src_idx, src_z, src_x] += -half_MT[2, 2] * src_v[t]
            txz_x[src_idx, src_z, src_x] += -half_MT[0, 2] * src_v[t]
            txz_z[src_idx, src_z, src_x] += -half_MT[0, 2] * src_v[t]
        else:
            txx_x[src_idx, src_z, src_x] += -half_MT[:, 0, 0] * src_v[src_idx, t]
            txx_z[src_idx, src_z, src_x] += -half_MT[:, 0, 0] * src_v[src_idx, t]
            tzz_x[src_idx, src_z, src_x] += -half_MT[:, 2, 2] * src_v[src_idx, t]
            tzz_z[src_idx, src_z, src_x] += -half_MT[:, 2, 2] * src_v[src_idx, t]
            txz_x[src_idx, src_z, src_x] += -half_MT[:, 0, 2] * src_v[src_idx, t]
            txz_z[src_idx, src_z, src_x] += -half_MT[:, 0, 2] * src_v[src_idx, t]

        # Combine x and z components of stress
        txx[:] = txx_x + txx_z
        tzz[:] = tzz_x + tzz_z
        txz[:] = txz_x + txz_z

        # Apply free surface boundary conditions
        if free_surface:
            tzz[:, h-1, :] = 0
            tzz[:, h-2, :] = -tzz[:, h, :]
            txz[:, h-2, :] = -txz[:, h-1, :]
            txz[:, h-3, :] = -txz[:, h, :]

        # Compute velocity components
        dxfm_txx = Dxfm(txx, fdc, ii, jj)
        dzbm_txz = Dzbm(txz, fdc, ii, jj)
        dxbm_txz = Dxbm(txz, fdc, ii, jj)
        dzfm_tzz = Dzfm(tzz, fdc, ii, jj)
        vx_x[:, idx_i, idx_j] = (pmlxn * vx_x[:, idx_i, idx_j] + dt * bx[idx_i, idx_j] * dxfm_txx / dx) / pmlxd
        vx_z[:, idx_i, idx_j] = (pmlzn * vx_z[:, idx_i, idx_j] + dt * bx[idx_i, idx_j] * dzbm_txz / dz) / pmlzd
        vz_x[:, idx_i, idx_j] = (pmlxn * vz_x[:, idx_i, idx_j] + dt * bz[idx_i, idx_j] * dxbm_txz / dx) / pmlxd
        vz_z[:, idx_i, idx_j] = (pmlzn * vz_z[:, idx_i, idx_j] + dt * bz[idx_i, idx_j] * dzfm_tzz / dz) / pmlzd
        vx[:] = vx_x + vx_z
        vz[:] = vz_x + vz_z

        # Apply free surface boundary conditions for velocity
        # if free_surface:
        #     vz[:, h-2, idx_j] = vz[:, h-1, idx_j]       + (lam[h-1, idx_j]/lamu[h-1, idx_j])*(vx[:, h-1, idx_j] - vx[:, h-1, idx_j_minus1])
        #     vx[:, h-2, idx_j] = vz[:, h-2, idx_j_plus1] - vz[:, h-2, idx_j] + vz[:, h-1, idx_j_plus1] - vz[:, h-1, idx_j] + vx[:, h, idx_j]
        #     vz[:, h-3, idx_j] = vz[:, h-2, idx_j]       + (lam[h-1, idx_j]/lamu[h-1, idx_j])*(vx[:, h-2, idx_j] - vx[:, h-2, idx_j_minus1])
        
        if free_surface:
            vz[:, h-2, idx_j] = vz[:, h-1, idx_j]
            vx[:, h-2, idx_j] = vz[:, h-2, idx_j_plus1] - vz[:, h-2, idx_j] + vz[:, h-1, idx_j_plus1] - vz[:, h-1, idx_j] + vx[:, h, idx_j]
            vz[:, h-3, idx_j] = vz[:, h-2, idx_j]

        # -----------------------------------------------------------
        #                   Receiver Observation
        # -----------------------------------------------------------
        rcv_txx[:, t, rcv_idx] = txx[:, rcv_z, rcv_x]
        rcv_tzz[:, t, rcv_idx] = tzz[:, rcv_z, rcv_x]
        rcv_txz[:, t, rcv_idx] = txz[:, rcv_z, rcv_x]
        rcv_vx[:, t, rcv_idx] = vx[:, rcv_z, rcv_x]
        rcv_vz[:, t, rcv_idx] = vz[:, rcv_z, rcv_x]

        # Store forward wavefields for visualization or further processing
        forward_wavefield_txx += torch.sum(txx * txx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_tzz += torch.sum(tzz * tzz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_txz += torch.sum(txz * txz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vx  += torch.sum(vx * vx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vz  += torch.sum(vz * vz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        
    return txx_x,txx_z,tzz_x,tzz_z,txz_x,txz_z,txx,tzz,txz,\
            vx_x,vx_z,vz_x,vz_z,vx,vz,\
            rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz,\
            forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz

@torch.jit.script
def step_forward_PML_8order(M:int,
                free_surface:bool,nx:int,nz:int,dx:float,dz:float,nabc:int,                 # basic settings
                src_x:Tensor,src_z:Tensor,src_n:int,dt:float,src_v:Tensor,MT:Tensor,        # source
                rcv_x:Tensor,rcv_z:Tensor,rcv_n:int,                                        # receiver
                bcx:Tensor,bcz:Tensor,                                                      # absobing bondary condition
                lam:Tensor,lamu:Tensor,
                C11:Tensor,C13:Tensor,C15:Tensor,C33:Tensor,C35:Tensor,C55:Tensor,          # elastic moduli parameters
                bx:Tensor,bz:Tensor,                                                        
                txx_x:Tensor,txx_z:Tensor,tzz_x:Tensor,tzz_z:Tensor,txz_x:Tensor,txz_z:Tensor,txx:Tensor,tzz:Tensor,txz:Tensor, # intermedia variable
                vx_x:Tensor,vx_z:Tensor,vz_x:Tensor,vz_z:Tensor,vx:Tensor,vz:Tensor,
                device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
                ):
    nt = src_v.shape[-1]    # Number of time steps
    fs_offset = M // 2      # Finite difference stencil offset

    # PML dimensions with free surface condition
    nx_pml = nx + 2 * nabc
    nz_pml = nz + (nabc + fs_offset if free_surface else 2 * nabc + fs_offset)
    
    # Clone tensors to avoid in-place modification
    vx, vz, txx, tzz, txz           = vx.clone()    , vz.clone()    , txx.clone(), tzz.clone(), txz.clone()
    vx_x, vz_x, txx_x, tzz_x, txz_x = vx_x.clone()  , vz_x.clone()  , txx_x.clone(), tzz_x.clone(), txz_x.clone()
    vx_z, vz_z, txx_z, tzz_z, txz_z = vx_z.clone()  , vz_z.clone()  , txx_z.clone(), tzz_z.clone(), txz_z.clone()

    # Initialize recorded waveforms
    rcv_txx, rcv_tzz, rcv_txz, rcv_vx, rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    forward_wavefield_txx, forward_wavefield_tzz, forward_wavefield_txz = torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vx, forward_wavefield_vz = torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device)
    
    # Finite difference order and indexing
    NN = M // 2
    fdc = DiffCoef(NN, 's')
    h = NN + 1
    i_start = NN 
    i_end = nz_pml - NN
    j_start = NN 
    j_end = nx_pml - NN
    idx_i = slice(i_start, i_end)
    idx_j = slice(j_start, j_end)
    idx_j_plus1 = slice(j_start+1,j_end+1)
    idx_j_minus1 = slice(j_start-1,j_end-1)
    ii = torch.arange(i_start, i_end, dtype=torch.long, device=device)  # z-axis range
    jj = torch.arange(j_start, j_end, dtype=torch.long, device=device)  # x-axis range
    
    # Damping factors for PML
    pmlxd = 1 + 0.5 * dt * bcx[idx_i, idx_j]
    pmlxn = 1 - 0.5 * dt * bcx[idx_i, idx_j]
    pmlzd = 1 + 0.5 * dt * bcz[idx_i, idx_j]
    pmlzn = 1 - 0.5 * dt * bcz[idx_i, idx_j]
    
    # Precompute constants for the forward loop
    dt_dx = dt / dx
    dt_dz = dt / dz
    pmlx_inv = 1.0 / pmlxd
    pmlz_inv = 1.0 / pmlzd
    
    # Source and receiver indices
    src_idx = list(range(src_n))
    rcv_idx = list(range(rcv_n))
    half_MT = MT[src_idx] / 2  # Half of the moment tensor
    
    # Adjust offset for free surface condition
    offset = fs_offset if free_surface else fs_offset + nabc
    
    # Finite difference operators for different axes
    Dxfm = Dxfm_8
    Dzfm = Dzfm_8
    Dxbm = Dxbm_8
    Dzbm = Dzbm_8
    
    # moment tensor source implementation
    for t in range(nt):
        
        # Compute stress components
        dxbm_vx = Dxbm(vx, fdc, ii, jj)
        dxbm_vz = Dxbm(vz, fdc, ii, jj)
        dzbm_vx = Dzbm(vx, fdc, ii, jj)
        dzbm_vz = Dzbm(vz, fdc, ii, jj)
        dxfm_vx = Dxfm(vx, fdc, ii, jj)
        dzfm_vx = Dzfm(vx, fdc, ii, jj)
        dxfm_vz = Dxfm(vz, fdc, ii, jj)
        dzfm_vz = Dzfm(vz, fdc, ii, jj)

        # Update stress fields
        txx_x[:, idx_i, idx_j] = (pmlxn * txx_x[:, idx_i, idx_j] + dt_dx * (C11[idx_i, idx_j] * dxbm_vx + C15[idx_i, idx_j] * dxbm_vz)) * pmlx_inv
        txx_z[:, idx_i, idx_j] = (pmlzn * txx_z[:, idx_i, idx_j] + dt_dz * (C15[idx_i, idx_j] * dzbm_vx + C13[idx_i, idx_j] * dzbm_vz)) * pmlz_inv
        tzz_x[:, idx_i, idx_j] = (pmlxn * tzz_x[:, idx_i, idx_j] + dt_dx * (C13[idx_i, idx_j] * dxbm_vx + C35[idx_i, idx_j] * dxbm_vz)) * pmlx_inv
        tzz_z[:, idx_i, idx_j] = (pmlzn * tzz_z[:, idx_i, idx_j] + dt_dz * (C35[idx_i, idx_j] * dzbm_vx + C33[idx_i, idx_j] * dzbm_vz)) * pmlz_inv
        txz_x[:, idx_i, idx_j] = (pmlxn * txz_x[:, idx_i, idx_j] + dt_dx * (C15[idx_i, idx_j] * dxfm_vx + C55[idx_i, idx_j] * dxfm_vz)) * pmlx_inv
        txz_z[:, idx_i, idx_j] = (pmlzn * txz_z[:, idx_i, idx_j] + dt_dz * (C55[idx_i, idx_j] * dzfm_vx + C35[idx_i, idx_j] * dzfm_vz)) * pmlz_inv

        # Add moment tensor source
        if len(src_v.shape) == 1:
            txx_x[src_idx, src_z, src_x] += -half_MT[0, 0] * src_v[t]
            txx_z[src_idx, src_z, src_x] += -half_MT[0, 0] * src_v[t]
            tzz_x[src_idx, src_z, src_x] += -half_MT[2, 2] * src_v[t]
            tzz_z[src_idx, src_z, src_x] += -half_MT[2, 2] * src_v[t]
            txz_x[src_idx, src_z, src_x] += -half_MT[0, 2] * src_v[t]
            txz_z[src_idx, src_z, src_x] += -half_MT[0, 2] * src_v[t]
        else:
            txx_x[src_idx, src_z, src_x] += -half_MT[:, 0, 0] * src_v[src_idx, t]
            txx_z[src_idx, src_z, src_x] += -half_MT[:, 0, 0] * src_v[src_idx, t]
            tzz_x[src_idx, src_z, src_x] += -half_MT[:, 2, 2] * src_v[src_idx, t]
            tzz_z[src_idx, src_z, src_x] += -half_MT[:, 2, 2] * src_v[src_idx, t]
            txz_x[src_idx, src_z, src_x] += -half_MT[:, 0, 2] * src_v[src_idx, t]
            txz_z[src_idx, src_z, src_x] += -half_MT[:, 0, 2] * src_v[src_idx, t]

        # Combine x and z components of stress
        txx[:] = txx_x + txx_z
        tzz[:] = tzz_x + tzz_z
        txz[:] = txz_x + txz_z

        # Apply free surface boundary conditions
        if free_surface:
            tzz[:, h-1, :] = 0
            tzz[:, h-2, :] = -tzz[:, h, :]
            txz[:, h-2, :] = -txz[:, h-1, :]
            txz[:, h-3, :] = -txz[:, h, :]

        # Compute velocity components
        dxfm_txx = Dxfm(txx, fdc, ii, jj)
        dzbm_txz = Dzbm(txz, fdc, ii, jj)
        dxbm_txz = Dxbm(txz, fdc, ii, jj)
        dzfm_tzz = Dzfm(tzz, fdc, ii, jj)
        vx_x[:, idx_i, idx_j] = (pmlxn * vx_x[:, idx_i, idx_j] + dt * bx[idx_i, idx_j] * dxfm_txx / dx) / pmlxd
        vx_z[:, idx_i, idx_j] = (pmlzn * vx_z[:, idx_i, idx_j] + dt * bx[idx_i, idx_j] * dzbm_txz / dz) / pmlzd
        vz_x[:, idx_i, idx_j] = (pmlxn * vz_x[:, idx_i, idx_j] + dt * bz[idx_i, idx_j] * dxbm_txz / dx) / pmlxd
        vz_z[:, idx_i, idx_j] = (pmlzn * vz_z[:, idx_i, idx_j] + dt * bz[idx_i, idx_j] * dzfm_tzz / dz) / pmlzd
        vx[:] = vx_x + vx_z
        vz[:] = vz_x + vz_z

        # Apply free surface boundary conditions for velocity
        # if free_surface:
        #     vz[:, h-2, idx_j] = vz[:, h-1, idx_j]       + (lam[h-1, idx_j]/lamu[h-1, idx_j])*(vx[:, h-1, idx_j] - vx[:, h-1, idx_j_minus1])
        #     vx[:, h-2, idx_j] = vz[:, h-2, idx_j_plus1] - vz[:, h-2, idx_j] + vz[:, h-1, idx_j_plus1] - vz[:, h-1, idx_j] + vx[:, h, idx_j]
        #     vz[:, h-3, idx_j] = vz[:, h-2, idx_j]       + (lam[h-1, idx_j]/lamu[h-1, idx_j])*(vx[:, h-2, idx_j] - vx[:, h-2, idx_j_minus1])
        
        if free_surface:
            vz[:, h-2, idx_j] = vz[:, h-1, idx_j]
            vx[:, h-2, idx_j] = vz[:, h-2, idx_j_plus1] - vz[:, h-2, idx_j] + vz[:, h-1, idx_j_plus1] - vz[:, h-1, idx_j] + vx[:, h, idx_j]
            vz[:, h-3, idx_j] = vz[:, h-2, idx_j]

        # -----------------------------------------------------------
        #                   Receiver Observation
        # -----------------------------------------------------------
        rcv_txx[:, t, rcv_idx] = txx[:, rcv_z, rcv_x]
        rcv_tzz[:, t, rcv_idx] = tzz[:, rcv_z, rcv_x]
        rcv_txz[:, t, rcv_idx] = txz[:, rcv_z, rcv_x]
        rcv_vx[:, t, rcv_idx] = vx[:, rcv_z, rcv_x]
        rcv_vz[:, t, rcv_idx] = vz[:, rcv_z, rcv_x]

        # Store forward wavefields for visualization or further processing
        forward_wavefield_txx += torch.sum(txx * txx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_tzz += torch.sum(tzz * tzz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_txz += torch.sum(txz * txz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vx  += torch.sum(vx * vx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vz  += torch.sum(vz * vz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        
    return txx_x,txx_z,tzz_x,tzz_z,txz_x,txz_z,txx,tzz,txz,\
            vx_x,vx_z,vz_x,vz_z,vx,vz,\
            rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz,\
            forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz

@torch.jit.script
def step_forward_PML_10order(M:int,
                free_surface:bool,nx:int,nz:int,dx:float,dz:float,nabc:int,                 # basic settings
                src_x:Tensor,src_z:Tensor,src_n:int,dt:float,src_v:Tensor,MT:Tensor,        # source
                rcv_x:Tensor,rcv_z:Tensor,rcv_n:int,                                        # receiver
                bcx:Tensor,bcz:Tensor,                                                      # absobing bondary condition
                lam:Tensor,lamu:Tensor,
                C11:Tensor,C13:Tensor,C15:Tensor,C33:Tensor,C35:Tensor,C55:Tensor,          # elastic moduli parameters
                bx:Tensor,bz:Tensor,                                                        
                txx_x:Tensor,txx_z:Tensor,tzz_x:Tensor,tzz_z:Tensor,txz_x:Tensor,txz_z:Tensor,txx:Tensor,tzz:Tensor,txz:Tensor, # intermedia variable
                vx_x:Tensor,vx_z:Tensor,vz_x:Tensor,vz_z:Tensor,vx:Tensor,vz:Tensor,
                device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
                ):
    nt = src_v.shape[-1]    # Number of time steps
    fs_offset = M // 2      # Finite difference stencil offset

    # PML dimensions with free surface condition
    nx_pml = nx + 2 * nabc
    nz_pml = nz + (nabc + fs_offset if free_surface else 2 * nabc + fs_offset)
    
    # Clone tensors to avoid in-place modification
    vx, vz, txx, tzz, txz           = vx.clone()    , vz.clone()    , txx.clone(), tzz.clone(), txz.clone()
    vx_x, vz_x, txx_x, tzz_x, txz_x = vx_x.clone()  , vz_x.clone()  , txx_x.clone(), tzz_x.clone(), txz_x.clone()
    vx_z, vz_z, txx_z, tzz_z, txz_z = vx_z.clone()  , vz_z.clone()  , txx_z.clone(), tzz_z.clone(), txz_z.clone()

    # Initialize recorded waveforms
    rcv_txx, rcv_tzz, rcv_txz, rcv_vx, rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device), torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    forward_wavefield_txx, forward_wavefield_tzz, forward_wavefield_txz = torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vx, forward_wavefield_vz = torch.zeros((nz, nx), dtype=dtype, device=device), torch.zeros((nz, nx), dtype=dtype, device=device)
    
    # Finite difference order and indexing
    NN = M // 2
    fdc = DiffCoef(NN, 's')
    h = NN + 1
    i_start = NN 
    i_end = nz_pml - NN
    j_start = NN 
    j_end = nx_pml - NN
    idx_i = slice(i_start, i_end)
    idx_j = slice(j_start, j_end)
    idx_j_plus1 = slice(j_start+1,j_end+1)
    idx_j_minus1 = slice(j_start-1,j_end-1)
    ii = torch.arange(i_start, i_end, dtype=torch.long, device=device)  # z-axis range
    jj = torch.arange(j_start, j_end, dtype=torch.long, device=device)  # x-axis range
    
    # Damping factors for PML
    pmlxd = 1 + 0.5 * dt * bcx[idx_i, idx_j]
    pmlxn = 1 - 0.5 * dt * bcx[idx_i, idx_j]
    pmlzd = 1 + 0.5 * dt * bcz[idx_i, idx_j]
    pmlzn = 1 - 0.5 * dt * bcz[idx_i, idx_j]
    
    # Precompute constants for the forward loop
    dt_dx = dt / dx
    dt_dz = dt / dz
    pmlx_inv = 1.0 / pmlxd
    pmlz_inv = 1.0 / pmlzd
    
    # Source and receiver indices
    src_idx = list(range(src_n))
    rcv_idx = list(range(rcv_n))
    half_MT = MT[src_idx] / 2  # Half of the moment tensor
    
    # Adjust offset for free surface condition
    offset = fs_offset if free_surface else fs_offset + nabc
    
    # Finite difference operators for different axes
    Dxfm = Dxfm_10
    Dzfm = Dzfm_10
    Dxbm = Dxbm_10
    Dzbm = Dzbm_10
    
    # moment tensor source implementation
    for t in range(nt):
        
        # Compute stress components
        dxbm_vx = Dxbm(vx, fdc, ii, jj)
        dxbm_vz = Dxbm(vz, fdc, ii, jj)
        dzbm_vx = Dzbm(vx, fdc, ii, jj)
        dzbm_vz = Dzbm(vz, fdc, ii, jj)
        dxfm_vx = Dxfm(vx, fdc, ii, jj)
        dzfm_vx = Dzfm(vx, fdc, ii, jj)
        dxfm_vz = Dxfm(vz, fdc, ii, jj)
        dzfm_vz = Dzfm(vz, fdc, ii, jj)

        # Update stress fields
        txx_x[:, idx_i, idx_j] = (pmlxn * txx_x[:, idx_i, idx_j] + dt_dx * (C11[idx_i, idx_j] * dxbm_vx + C15[idx_i, idx_j] * dxbm_vz)) * pmlx_inv
        txx_z[:, idx_i, idx_j] = (pmlzn * txx_z[:, idx_i, idx_j] + dt_dz * (C15[idx_i, idx_j] * dzbm_vx + C13[idx_i, idx_j] * dzbm_vz)) * pmlz_inv
        tzz_x[:, idx_i, idx_j] = (pmlxn * tzz_x[:, idx_i, idx_j] + dt_dx * (C13[idx_i, idx_j] * dxbm_vx + C35[idx_i, idx_j] * dxbm_vz)) * pmlx_inv
        tzz_z[:, idx_i, idx_j] = (pmlzn * tzz_z[:, idx_i, idx_j] + dt_dz * (C35[idx_i, idx_j] * dzbm_vx + C33[idx_i, idx_j] * dzbm_vz)) * pmlz_inv
        txz_x[:, idx_i, idx_j] = (pmlxn * txz_x[:, idx_i, idx_j] + dt_dx * (C15[idx_i, idx_j] * dxfm_vx + C55[idx_i, idx_j] * dxfm_vz)) * pmlx_inv
        txz_z[:, idx_i, idx_j] = (pmlzn * txz_z[:, idx_i, idx_j] + dt_dz * (C55[idx_i, idx_j] * dzfm_vx + C35[idx_i, idx_j] * dzfm_vz)) * pmlz_inv

        # Add moment tensor source
        if len(src_v.shape) == 1:
            txx_x[src_idx, src_z, src_x] += -half_MT[0, 0] * src_v[t]
            txx_z[src_idx, src_z, src_x] += -half_MT[0, 0] * src_v[t]
            tzz_x[src_idx, src_z, src_x] += -half_MT[2, 2] * src_v[t]
            tzz_z[src_idx, src_z, src_x] += -half_MT[2, 2] * src_v[t]
            txz_x[src_idx, src_z, src_x] += -half_MT[0, 2] * src_v[t]
            txz_z[src_idx, src_z, src_x] += -half_MT[0, 2] * src_v[t]
        else:
            txx_x[src_idx, src_z, src_x] += -half_MT[:, 0, 0] * src_v[src_idx, t]
            txx_z[src_idx, src_z, src_x] += -half_MT[:, 0, 0] * src_v[src_idx, t]
            tzz_x[src_idx, src_z, src_x] += -half_MT[:, 2, 2] * src_v[src_idx, t]
            tzz_z[src_idx, src_z, src_x] += -half_MT[:, 2, 2] * src_v[src_idx, t]
            txz_x[src_idx, src_z, src_x] += -half_MT[:, 0, 2] * src_v[src_idx, t]
            txz_z[src_idx, src_z, src_x] += -half_MT[:, 0, 2] * src_v[src_idx, t]

        # Combine x and z components of stress
        txx[:] = txx_x + txx_z
        tzz[:] = tzz_x + tzz_z
        txz[:] = txz_x + txz_z

        # Apply free surface boundary conditions
        if free_surface:
            tzz[:, h-1, :] = 0
            tzz[:, h-2, :] = -tzz[:, h, :]
            txz[:, h-2, :] = -txz[:, h-1, :]
            txz[:, h-3, :] = -txz[:, h, :]

        # Compute velocity components
        dxfm_txx = Dxfm(txx, fdc, ii, jj)
        dzbm_txz = Dzbm(txz, fdc, ii, jj)
        dxbm_txz = Dxbm(txz, fdc, ii, jj)
        dzfm_tzz = Dzfm(tzz, fdc, ii, jj)
        vx_x[:, idx_i, idx_j] = (pmlxn * vx_x[:, idx_i, idx_j] + dt * bx[idx_i, idx_j] * dxfm_txx / dx) / pmlxd
        vx_z[:, idx_i, idx_j] = (pmlzn * vx_z[:, idx_i, idx_j] + dt * bx[idx_i, idx_j] * dzbm_txz / dz) / pmlzd
        vz_x[:, idx_i, idx_j] = (pmlxn * vz_x[:, idx_i, idx_j] + dt * bz[idx_i, idx_j] * dxbm_txz / dx) / pmlxd
        vz_z[:, idx_i, idx_j] = (pmlzn * vz_z[:, idx_i, idx_j] + dt * bz[idx_i, idx_j] * dzfm_tzz / dz) / pmlzd
        vx[:] = vx_x + vx_z
        vz[:] = vz_x + vz_z

        # Apply free surface boundary conditions for velocity
        # if free_surface:
        #     vz[:, h-2, idx_j] = vz[:, h-1, idx_j]       + (lam[h-1, idx_j]/lamu[h-1, idx_j])*(vx[:, h-1, idx_j] - vx[:, h-1, idx_j_minus1])
        #     vx[:, h-2, idx_j] = vz[:, h-2, idx_j_plus1] - vz[:, h-2, idx_j] + vz[:, h-1, idx_j_plus1] - vz[:, h-1, idx_j] + vx[:, h, idx_j]
        #     vz[:, h-3, idx_j] = vz[:, h-2, idx_j]       + (lam[h-1, idx_j]/lamu[h-1, idx_j])*(vx[:, h-2, idx_j] - vx[:, h-2, idx_j_minus1])
        
        if free_surface:
            vz[:, h-2, idx_j] = vz[:, h-1, idx_j]
            vx[:, h-2, idx_j] = vz[:, h-2, idx_j_plus1] - vz[:, h-2, idx_j] + vz[:, h-1, idx_j_plus1] - vz[:, h-1, idx_j] + vx[:, h, idx_j]
            vz[:, h-3, idx_j] = vz[:, h-2, idx_j]

        # -----------------------------------------------------------
        #                   Receiver Observation
        # -----------------------------------------------------------
        rcv_txx[:, t, rcv_idx] = txx[:, rcv_z, rcv_x]
        rcv_tzz[:, t, rcv_idx] = tzz[:, rcv_z, rcv_x]
        rcv_txz[:, t, rcv_idx] = txz[:, rcv_z, rcv_x]
        rcv_vx[:, t, rcv_idx] = vx[:, rcv_z, rcv_x]
        rcv_vz[:, t, rcv_idx] = vz[:, rcv_z, rcv_x]

        # Store forward wavefields for visualization or further processing
        forward_wavefield_txx += torch.sum(txx * txx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_tzz += torch.sum(tzz * tzz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_txz += torch.sum(txz * txz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vx  += torch.sum(vx * vx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vz  += torch.sum(vz * vz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        
    return txx_x,txx_z,tzz_x,tzz_z,txz_x,txz_z,txx,tzz,txz,\
            vx_x,vx_z,vz_x,vz_z,vx,vz,\
            rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz,\
            forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz


##########################################################################
#                step forward Modeling :damping mode   
##########################################################################
@torch.jit.script
def step_forward_ABL_4order(M:int,
                free_surface:bool,nx:int,nz:int,dx:float,dz:float,nabc:int,                 # basic settings
                src_x:Tensor,src_z:Tensor,src_n:int,dt:float,src_v:Tensor,MT:Tensor,        # source
                rcv_x:Tensor,rcv_z:Tensor,rcv_n:int,                                        # receiver
                damp:Tensor,                                                                # absobing bondary condition
                lam:Tensor,lamu:Tensor,
                C11:Tensor,C13:Tensor,C15:Tensor,C33:Tensor,C35:Tensor,C55:Tensor,          # elastic moduli parameters
                bx:Tensor,bz:Tensor,                                                        
                txx:Tensor,tzz:Tensor,txz:Tensor,                                           # intermedia variable
                vx:Tensor,vz:Tensor,
                device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
                ):
    """Description:
    --------------
        Perform a single time step in the simulation of wave propagation using a 4th order finite difference method.
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
    
    Parameters:
    --------------
        - M (int): Order of the finite difference stencil.
        - free_surface (bool): Indicates whether free surface conditions should be applied.
        - nx (int): Number of grid points in the x-direction.
        - nz (int): Number of grid points in the z-direction.
        - dx (float): Grid spacing in the x-direction.
        - dz (float): Grid spacing in the z-direction.
        - nabc (int): Number of absorbing boundary condition layers.
        - src_x (Tensor): Source x-coordinates.
        - src_z (Tensor): Source z-coordinates.
        - src_n (int): Number of sources.
        - dt (float): Time step size.
        - src_v (Tensor): Source time function values.
        - MT (Tensor): Moment tensor representing the source.
        - rcv_x (Tensor): Receiver x-coordinates.
        - rcv_z (Tensor): Receiver z-coordinates.
        - rcv_n (int): Number of receivers.
        - damp (Tensor): Damping tensor for absorbing boundary condition.
        - lam (Tensor): First Lamé parameter.
        - lamu (Tensor): Second Lamé parameter.
        - C11, C13, C15, C33, C35, C55 (Tensor): Elastic moduli parameters.
        - bx (Tensor): Coefficient for x-direction velocity update.
        - bz (Tensor): Coefficient for z-direction velocity update.
        - txx (Tensor): Stress tensor component (xx).
        - tzz (Tensor): Stress tensor component (zz).
        - txz (Tensor): Stress tensor component (xz).
        - vx (Tensor): Velocity component in the x-direction.
        - vz (Tensor): Velocity component in the z-direction.
        - device (torch.device): Device to allocate tensors on (default is CPU).
        - dtype (torch.dtype): Data type for the tensors (default is float32).

    Returns:
    - Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: 
      Updated stress tensors (txx, tzz, txz), updated velocity tensors (vx, vz),
      recorded waveforms (rcv_txx, rcv_tzz, rcv_txz, rcv_vx, rcv_vz),
      and forward wavefields (forward_wavefield_txx, forward_wavefield_tzz, forward_wavefield_txz, forward_wavefield_vx, forward_wavefield_vz).
    """
    nt = src_v.shape[-1]    # Number of time steps
    fs_offset = M // 2      # Finite difference stencil offset
    
    # Configure ABL dimensions based on boundary conditions
    nx_pml = nx + 2 * nabc
    nz_pml = nz + (nabc + fs_offset if free_surface else 2 * nabc + fs_offset)
    
    # Clone tensors to avoid in-place modification
    vx, vz, txx, tzz, txz = vx.clone(), vz.clone(), txx.clone(), tzz.clone(), txz.clone()
    
    # Initialize recorded waveforms
    rcv_txx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_tzz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_txz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx  = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz  = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    forward_wavefield_txx = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_tzz = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_txz = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vx  = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vz  = torch.zeros((nz, nx), dtype=dtype, device=device)
    
    # Finite difference order and indexing
    NN = M // 2
    fdc = DiffCoef(NN, 's')
    h       = NN + 1
    i_start = NN 
    i_end   = nz_pml - NN
    j_start = NN 
    j_end   = nx_pml - NN
    idx_i   = slice(i_start, i_end)
    idx_j   = slice(j_start, j_end)
    idx_j_plus1 = slice(j_start+1,j_end+1)
    idx_j_minus1 = slice(j_start-1,j_end-1)
    ii      = torch.arange(i_start, i_end, dtype=torch.long, device=device)  # z-axis range
    jj      = torch.arange(j_start, j_end, dtype=torch.long, device=device)  # x-axis range
    
    # Finite difference operators for different axes
    Dxfm = Dxfm_4
    Dzfm = Dzfm_4
    Dxbm = Dxbm_4
    Dzbm = Dzbm_4
    
    # Source and receiver indices
    src_idx = list(range(src_n))
    rcv_idx = list(range(rcv_n))
    scaling_factor = -1 / 2
    
    # Adjust offset for free surface condition
    offset = fs_offset if free_surface else fs_offset + nabc
    
    # moment tensor source implementation
    for t in range(nt):
        # Compute stress components
        dxbm_vx = Dxbm(vx, fdc, ii, jj)
        dxbm_vz = Dxbm(vz, fdc, ii, jj)
        dzbm_vx = Dzbm(vx, fdc, ii, jj)
        dzbm_vz = Dzbm(vz, fdc, ii, jj)
        dxfm_vx = Dxfm(vx, fdc, ii, jj)
        dzfm_vx = Dzfm(vx, fdc, ii, jj)
        dxfm_vz = Dxfm(vz, fdc, ii, jj)
        dzfm_vz = Dzfm(vz, fdc, ii, jj)
        
        txx[:, idx_i, idx_j] = txx[:, idx_i, idx_j] + dt*((C11[idx_i,idx_j]*dxbm_vx + C15[idx_i,idx_j]*dxbm_vz)/dx
                                                        + (C15[idx_i,idx_j]*dzbm_vx + C13[idx_i,idx_j]*dzbm_vz)/dz)
        tzz[:, idx_i, idx_j] = tzz[:, idx_i, idx_j] + dt*((C13[idx_i,idx_j]*dxbm_vx + C35[idx_i,idx_j]*dxbm_vz)/dx
                                                        + (C35[idx_i,idx_j]*dzbm_vx + C33[idx_i,idx_j]*dzbm_vz)/dz)
        txz[:, idx_i, idx_j] = txz[:, idx_i, idx_j] + dt*((C15[idx_i,idx_j]*dxfm_vx + C55[idx_i,idx_j]*dxfm_vz)/dx
                                                        + (C55[idx_i,idx_j]*dzfm_vx + C35[idx_i,idx_j]*dzfm_vz)/dz)
        
        # Add source
        if src_v.ndim == 1:
            txx[src_idx, src_z, src_x] += scaling_factor * MT[0, 0] * src_v[t]
            tzz[src_idx, src_z, src_x] += scaling_factor * MT[2, 2] * src_v[t]
            txz[src_idx, src_z, src_x] += scaling_factor * MT[0, 2] * src_v[t]
        else:
            txx[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 0, 0] * src_v[src_idx, t]
            tzz[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 2, 2] * src_v[src_idx, t]
            txz[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 0, 2] * src_v[src_idx, t]

        # topFs with the assumption of weak anisotropy near the surface
        if free_surface:
            tzz[:,h-1,:] = 0
            tzz[:,h-2,:] = -tzz[:,h,:]
            txz[:,h-2,:] = -txz[:,h-1,:]
            txz[:,h-3,:] = -txz[:,h,:]
        
        # Update velocity components
        dxfm_txx = Dxfm(txx, fdc, ii, jj)
        dzbm_txz = Dzbm(txz, fdc, ii, jj)
        dxbm_txz = Dxbm(txz, fdc, ii, jj)
        dzfm_tzz = Dzfm(tzz, fdc, ii, jj)
        vx[:, idx_i, idx_j] += dt * bx[idx_i, idx_j] * (dxfm_txx / dx + dzbm_txz / dz)
        vz[:, idx_i, idx_j] += dt * bz[idx_i, idx_j] * (dxbm_txz / dx + dzfm_tzz / dz)

        # Apply free surface boundary conditions
        if free_surface:
            vz[:, h - 2, idx_j] = vz[:, h - 1, idx_j]
            vx[:, h - 2, idx_j] = vz[:, h - 2, idx_j_plus1] - vz[:, h - 2, idx_j] + vz[:, h - 1, idx_j_plus1] - vz[:, h - 1, idx_j] + vx[:, h, idx_j]
            vz[:, h - 3, idx_j] = vz[:, h - 2, idx_j]

        # Apply damping
        vx *= damp
        vz *= damp
        
        # Record receiver waveforms
        rcv_txx[:, t, rcv_idx] = txx[:, rcv_z, rcv_x]
        rcv_tzz[:, t, rcv_idx] = tzz[:, rcv_z, rcv_x]
        rcv_txz[:, t, rcv_idx] = txz[:, rcv_z, rcv_x]
        rcv_vx[:, t, rcv_idx] = vx[:, rcv_z, rcv_x]
        rcv_vz[:, t, rcv_idx] = vz[:, rcv_z, rcv_x]
        
        # Store forward wavefields for visualization or further processing
        forward_wavefield_txx += torch.sum(txx * txx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_tzz += torch.sum(tzz * tzz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_txz += torch.sum(txz * txz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vx  += torch.sum(vx * vx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vz  += torch.sum(vz * vz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()

    return txx,tzz,txz,vx,vz,rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz,\
            forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz

@torch.jit.script
def step_forward_ABL_6order(M:int,
                free_surface:bool,nx:int,nz:int,dx:float,dz:float,nabc:int,                 # basic settings
                src_x:Tensor,src_z:Tensor,src_n:int,dt:float,src_v:Tensor,MT:Tensor,        # source
                rcv_x:Tensor,rcv_z:Tensor,rcv_n:int,                                        # receiver
                damp:Tensor,                                                                # absobing bondary condition
                lam:Tensor,lamu:Tensor,
                C11:Tensor,C13:Tensor,C15:Tensor,C33:Tensor,C35:Tensor,C55:Tensor,          # elastic moduli parameters
                bx:Tensor,bz:Tensor,                                                        
                txx:Tensor,tzz:Tensor,txz:Tensor,                                           # intermedia variable
                vx:Tensor,vz:Tensor,
                device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
                ):
    nt = src_v.shape[-1]    # Number of time steps
    fs_offset = M // 2      # Finite difference stencil offset
    
    # Configure ABL dimensions based on boundary conditions
    nx_pml = nx + 2 * nabc
    nz_pml = nz + (nabc + fs_offset if free_surface else 2 * nabc + fs_offset)
    
    # Clone tensors to avoid in-place modification
    vx, vz, txx, tzz, txz = vx.clone(), vz.clone(), txx.clone(), tzz.clone(), txz.clone()
    
    # Initialize recorded waveforms
    rcv_txx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_tzz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_txz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    forward_wavefield_txx = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_tzz = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_txz = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vx = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vz = torch.zeros((nz, nx), dtype=dtype, device=device)
    
    # Finite difference order and indexing
    NN = M // 2
    fdc = DiffCoef(NN, 's')
    h = NN + 1
    i_start = NN 
    i_end = nz_pml - NN
    j_start = NN 
    j_end = nx_pml - NN
    idx_i = slice(i_start, i_end)
    idx_j = slice(j_start, j_end)
    ii = torch.arange(i_start, i_end, dtype=torch.long, device=device)  # z-axis range
    jj = torch.arange(j_start, j_end, dtype=torch.long, device=device)  # x-axis range
    
    # Finite difference operators for different axes
    Dxfm = Dxfm_6
    Dzfm = Dzfm_6
    Dxbm = Dxbm_6
    Dzbm = Dzbm_6
    
    # Source and receiver indices
    src_idx = list(range(src_n))
    rcv_idx = list(range(rcv_n))
    scaling_factor = -1 / 2
    
    # Adjust offset for free surface condition
    offset = fs_offset if free_surface else fs_offset + nabc
    
    # moment tensor source implementation
    for t in range(nt):
        # Compute stress components
        dxbm_vx = Dxbm(vx, fdc, ii, jj)
        dxbm_vz = Dxbm(vz, fdc, ii, jj)
        dzbm_vx = Dzbm(vx, fdc, ii, jj)
        dzbm_vz = Dzbm(vz, fdc, ii, jj)
        dxfm_vx = Dxfm(vx, fdc, ii, jj)
        dzfm_vx = Dzfm(vx, fdc, ii, jj)
        dxfm_vz = Dxfm(vz, fdc, ii, jj)
        dzfm_vz = Dzfm(vz, fdc, ii, jj)
        
        txx[:, idx_i, idx_j] = txx[:, idx_i, idx_j] + dt*((C11[idx_i,idx_j]*dxbm_vx + C15[idx_i,idx_j]*dxbm_vz)/dx
                                                        + (C15[idx_i,idx_j]*dzbm_vx + C13[idx_i,idx_j]*dzbm_vz)/dz)
        tzz[:, idx_i, idx_j] = tzz[:, idx_i, idx_j] + dt*((C13[idx_i,idx_j]*dxbm_vx + C35[idx_i,idx_j]*dxbm_vz)/dx
                                                        + (C35[idx_i,idx_j]*dzbm_vx + C33[idx_i,idx_j]*dzbm_vz)/dz)
        txz[:, idx_i, idx_j] = txz[:, idx_i, idx_j] + dt*((C15[idx_i,idx_j]*dxfm_vx + C55[idx_i,idx_j]*dxfm_vz)/dx
                                                        + (C55[idx_i,idx_j]*dzfm_vx + C35[idx_i,idx_j]*dzfm_vz)/dz)
        
        # Add source
        if src_v.ndim == 1:
            txx[src_idx, src_z, src_x] += scaling_factor * MT[0, 0] * src_v[t]
            tzz[src_idx, src_z, src_x] += scaling_factor * MT[2, 2] * src_v[t]
            txz[src_idx, src_z, src_x] += scaling_factor * MT[0, 2] * src_v[t]
        else:
            txx[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 0, 0] * src_v[src_idx, t]
            tzz[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 2, 2] * src_v[src_idx, t]
            txz[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 0, 2] * src_v[src_idx, t]

        # topFs with the assumption of weak anisotropy near the surface
        if free_surface:
            tzz[:,h-1,:] = 0
            tzz[:,h-2,:] = -tzz[:,h,:]
            txz[:,h-2,:] = -txz[:,h-1,:]
            txz[:,h-3,:] = -txz[:,h,:]
        
        # Update velocity components
        dxfm_txx = Dxfm(txx, fdc, ii, jj)
        dzbm_txz = Dzbm(txz, fdc, ii, jj)
        dxbm_txz = Dxbm(txz, fdc, ii, jj)
        dzfm_tzz = Dzfm(tzz, fdc, ii, jj)
        vx[:, idx_i, idx_j] += dt * bx[idx_i, idx_j] * (dxfm_txx / dx + dzbm_txz / dz)
        vz[:, idx_i, idx_j] += dt * bz[idx_i, idx_j] * (dxbm_txz / dx + dzfm_tzz / dz)

        # Apply free surface boundary conditions
        if free_surface:
            vz[:, h - 2, idx_j] = vz[:, h - 1, idx_j]
            vx[:, h - 2, idx_j] = vz[:, h - 2, j_start + 1:j_end + 1] - vz[:, h - 2, idx_j] + vz[:, h - 1, j_start + 1:j_end + 1] - vz[:, h - 1, idx_j] + vx[:, h, idx_j]
            vz[:, h - 3, idx_j] = vz[:, h - 2, idx_j]

        # Apply damping
        vx *= damp
        vz *= damp
        
        # Record receiver waveforms
        rcv_txx[:, t, rcv_idx] = txx[:, rcv_z, rcv_x]
        rcv_tzz[:, t, rcv_idx] = tzz[:, rcv_z, rcv_x]
        rcv_txz[:, t, rcv_idx] = txz[:, rcv_z, rcv_x]
        rcv_vx[:, t, rcv_idx] = vx[:, rcv_z, rcv_x]
        rcv_vz[:, t, rcv_idx] = vz[:, rcv_z, rcv_x]
        
        # Store forward wavefields for visualization or further processing
        forward_wavefield_txx += torch.sum(txx * txx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_tzz += torch.sum(tzz * tzz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_txz += torch.sum(txz * txz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vx  += torch.sum(vx * vx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vz  += torch.sum(vz * vz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()

    return txx,tzz,txz,vx,vz,rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz,\
            forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz

@torch.jit.script
def step_forward_ABL_8order(M:int,
                free_surface:bool,nx:int,nz:int,dx:float,dz:float,nabc:int,                 # basic settings
                src_x:Tensor,src_z:Tensor,src_n:int,dt:float,src_v:Tensor,MT:Tensor,        # source
                rcv_x:Tensor,rcv_z:Tensor,rcv_n:int,                                        # receiver
                damp:Tensor,                                                                # absobing bondary condition
                lam:Tensor,lamu:Tensor,
                C11:Tensor,C13:Tensor,C15:Tensor,C33:Tensor,C35:Tensor,C55:Tensor,          # elastic moduli parameters
                bx:Tensor,bz:Tensor,                                                        
                txx:Tensor,tzz:Tensor,txz:Tensor,                                           # intermedia variable
                vx:Tensor,vz:Tensor,
                device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
                ):
    nt = src_v.shape[-1]    # Number of time steps
    fs_offset = M // 2      # Finite difference stencil offset
    
    # Configure ABL dimensions based on boundary conditions
    nx_pml = nx + 2 * nabc
    nz_pml = nz + (nabc + fs_offset if free_surface else 2 * nabc + fs_offset)
    
    # Clone tensors to avoid in-place modification
    vx, vz, txx, tzz, txz = vx.clone(), vz.clone(), txx.clone(), tzz.clone(), txz.clone()
    
    # Initialize recorded waveforms
    rcv_txx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_tzz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_txz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    forward_wavefield_txx = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_tzz = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_txz = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vx = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vz = torch.zeros((nz, nx), dtype=dtype, device=device)
    
    # Finite difference order and indexing
    NN = M // 2
    fdc = DiffCoef(NN, 's')
    h = NN + 1
    i_start = NN 
    i_end = nz_pml - NN
    j_start = NN 
    j_end = nx_pml - NN
    idx_i = slice(i_start, i_end)
    idx_j = slice(j_start, j_end)
    ii = torch.arange(i_start, i_end, dtype=torch.long, device=device)  # z-axis range
    jj = torch.arange(j_start, j_end, dtype=torch.long, device=device)  # x-axis range
    
    # Finite difference operators for different axes
    Dxfm = Dxfm_8
    Dzfm = Dzfm_8
    Dxbm = Dxbm_8
    Dzbm = Dzbm_8
    
    # Source and receiver indices
    src_idx = list(range(src_n))
    rcv_idx = list(range(rcv_n))
    scaling_factor = -1 / 2
    
    # Adjust offset for free surface condition
    offset = fs_offset if free_surface else fs_offset + nabc
    
    # moment tensor source implementation
    for t in range(nt):
        # Compute stress components
        dxbm_vx = Dxbm(vx, fdc, ii, jj)
        dxbm_vz = Dxbm(vz, fdc, ii, jj)
        dzbm_vx = Dzbm(vx, fdc, ii, jj)
        dzbm_vz = Dzbm(vz, fdc, ii, jj)
        dxfm_vx = Dxfm(vx, fdc, ii, jj)
        dzfm_vx = Dzfm(vx, fdc, ii, jj)
        dxfm_vz = Dxfm(vz, fdc, ii, jj)
        dzfm_vz = Dzfm(vz, fdc, ii, jj)
        
        txx[:, idx_i, idx_j] = txx[:, idx_i, idx_j] + dt*((C11[idx_i,idx_j]*dxbm_vx + C15[idx_i,idx_j]*dxbm_vz)/dx
                                                        + (C15[idx_i,idx_j]*dzbm_vx + C13[idx_i,idx_j]*dzbm_vz)/dz)
        tzz[:, idx_i, idx_j] = tzz[:, idx_i, idx_j] + dt*((C13[idx_i,idx_j]*dxbm_vx + C35[idx_i,idx_j]*dxbm_vz)/dx
                                                        + (C35[idx_i,idx_j]*dzbm_vx + C33[idx_i,idx_j]*dzbm_vz)/dz)
        txz[:, idx_i, idx_j] = txz[:, idx_i, idx_j] + dt*((C15[idx_i,idx_j]*dxfm_vx + C55[idx_i,idx_j]*dxfm_vz)/dx
                                                        + (C55[idx_i,idx_j]*dzfm_vx + C35[idx_i,idx_j]*dzfm_vz)/dz)
        
        # Add source
        if src_v.ndim == 1:
            txx[src_idx, src_z, src_x] += scaling_factor * MT[0, 0] * src_v[t]
            tzz[src_idx, src_z, src_x] += scaling_factor * MT[2, 2] * src_v[t]
            txz[src_idx, src_z, src_x] += scaling_factor * MT[0, 2] * src_v[t]
        else:
            txx[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 0, 0] * src_v[src_idx, t]
            tzz[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 2, 2] * src_v[src_idx, t]
            txz[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 0, 2] * src_v[src_idx, t]

        # topFs with the assumption of weak anisotropy near the surface
        if free_surface:
            tzz[:,h-1,:] = 0
            tzz[:,h-2,:] = -tzz[:,h,:]
            txz[:,h-2,:] = -txz[:,h-1,:]
            txz[:,h-3,:] = -txz[:,h,:]
        
        # Update velocity components
        dxfm_txx = Dxfm(txx, fdc, ii, jj)
        dzbm_txz = Dzbm(txz, fdc, ii, jj)
        dxbm_txz = Dxbm(txz, fdc, ii, jj)
        dzfm_tzz = Dzfm(tzz, fdc, ii, jj)
        vx[:, idx_i, idx_j] += dt * bx[idx_i, idx_j] * (dxfm_txx / dx + dzbm_txz / dz)
        vz[:, idx_i, idx_j] += dt * bz[idx_i, idx_j] * (dxbm_txz / dx + dzfm_tzz / dz)

        # Apply free surface boundary conditions
        if free_surface:
            vz[:, h - 2, idx_j] = vz[:, h - 1, idx_j]
            vx[:, h - 2, idx_j] = vz[:, h - 2, j_start + 1:j_end + 1] - vz[:, h - 2, idx_j] + vz[:, h - 1, j_start + 1:j_end + 1] - vz[:, h - 1, idx_j] + vx[:, h, idx_j]
            vz[:, h - 3, idx_j] = vz[:, h - 2, idx_j]

        # Apply damping
        vx *= damp
        vz *= damp
        
        # Record receiver waveforms
        rcv_txx[:, t, rcv_idx] = txx[:, rcv_z, rcv_x]
        rcv_tzz[:, t, rcv_idx] = tzz[:, rcv_z, rcv_x]
        rcv_txz[:, t, rcv_idx] = txz[:, rcv_z, rcv_x]
        rcv_vx[:, t, rcv_idx] = vx[:, rcv_z, rcv_x]
        rcv_vz[:, t, rcv_idx] = vz[:, rcv_z, rcv_x]
        
        # Store forward wavefields for visualization or further processing
        forward_wavefield_txx += torch.sum(txx * txx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_tzz += torch.sum(tzz * tzz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_txz += torch.sum(txz * txz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vx += torch.sum(vx * vx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vz += torch.sum(vz * vz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()

    return txx,tzz,txz,vx,vz,rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz,\
            forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz

@torch.jit.script
def step_forward_ABL_10order(M:int,
                free_surface:bool,nx:int,nz:int,dx:float,dz:float,nabc:int,                 # basic settings
                src_x:Tensor,src_z:Tensor,src_n:int,dt:float,src_v:Tensor,MT:Tensor,        # source
                rcv_x:Tensor,rcv_z:Tensor,rcv_n:int,                                        # receiver
                damp:Tensor,                                                                # absobing bondary condition
                lam:Tensor,lamu:Tensor,
                C11:Tensor,C13:Tensor,C15:Tensor,C33:Tensor,C35:Tensor,C55:Tensor,          # elastic moduli parameters
                bx:Tensor,bz:Tensor,                                                        
                txx:Tensor,tzz:Tensor,txz:Tensor,                                           # intermedia variable
                vx:Tensor,vz:Tensor,
                device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
                ):
    nt = src_v.shape[-1]    # Number of time steps
    fs_offset = M // 2      # Finite difference stencil offset
    
    # Configure ABL dimensions based on boundary conditions
    nx_pml = nx + 2 * nabc
    nz_pml = nz + (nabc + fs_offset if free_surface else 2 * nabc + fs_offset)
    
    # Clone tensors to avoid in-place modification
    vx, vz, txx, tzz, txz = vx.clone(), vz.clone(), txx.clone(), tzz.clone(), txz.clone()
    
    # Initialize recorded waveforms
    rcv_txx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_tzz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_txz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    forward_wavefield_txx = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_tzz = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_txz = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vx = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vz = torch.zeros((nz, nx), dtype=dtype, device=device)
    
    # Finite difference order and indexing
    NN = M // 2
    fdc = DiffCoef(NN, 's')
    h = NN + 1
    i_start = NN 
    i_end = nz_pml - NN
    j_start = NN 
    j_end = nx_pml - NN
    idx_i = slice(i_start, i_end)
    idx_j = slice(j_start, j_end)
    ii = torch.arange(i_start, i_end, dtype=torch.long, device=device)  # z-axis range
    jj = torch.arange(j_start, j_end, dtype=torch.long, device=device)  # x-axis range
    
    # Finite difference operators for different axes
    Dxfm = Dxfm_10
    Dzfm = Dzfm_10
    Dxbm = Dxbm_10
    Dzbm = Dzbm_10
    
    # Source and receiver indices
    src_idx = list(range(src_n))
    rcv_idx = list(range(rcv_n))
    scaling_factor = -1 / 2
    
    # Adjust offset for free surface condition
    offset = fs_offset if free_surface else fs_offset + nabc
    
    # moment tensor source implementation
    for t in range(nt):
        # Compute stress components
        dxbm_vx = Dxbm(vx, fdc, ii, jj)
        dxbm_vz = Dxbm(vz, fdc, ii, jj)
        dzbm_vx = Dzbm(vx, fdc, ii, jj)
        dzbm_vz = Dzbm(vz, fdc, ii, jj)
        dxfm_vx = Dxfm(vx, fdc, ii, jj)
        dzfm_vx = Dzfm(vx, fdc, ii, jj)
        dxfm_vz = Dxfm(vz, fdc, ii, jj)
        dzfm_vz = Dzfm(vz, fdc, ii, jj)
        
        txx[:, idx_i, idx_j] = txx[:, idx_i, idx_j] + dt*((C11[idx_i,idx_j]*dxbm_vx + C15[idx_i,idx_j]*dxbm_vz)/dx
                                                        + (C15[idx_i,idx_j]*dzbm_vx + C13[idx_i,idx_j]*dzbm_vz)/dz)
        tzz[:, idx_i, idx_j] = tzz[:, idx_i, idx_j] + dt*((C13[idx_i,idx_j]*dxbm_vx + C35[idx_i,idx_j]*dxbm_vz)/dx
                                                        + (C35[idx_i,idx_j]*dzbm_vx + C33[idx_i,idx_j]*dzbm_vz)/dz)
        txz[:, idx_i, idx_j] = txz[:, idx_i, idx_j] + dt*((C15[idx_i,idx_j]*dxfm_vx + C55[idx_i,idx_j]*dxfm_vz)/dx
                                                        + (C55[idx_i,idx_j]*dzfm_vx + C35[idx_i,idx_j]*dzfm_vz)/dz)
        
        # Add source
        if src_v.ndim == 1:
            txx[src_idx, src_z, src_x] += scaling_factor * MT[0, 0] * src_v[t]
            tzz[src_idx, src_z, src_x] += scaling_factor * MT[2, 2] * src_v[t]
            txz[src_idx, src_z, src_x] += scaling_factor * MT[0, 2] * src_v[t]
        else:
            txx[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 0, 0] * src_v[src_idx, t]
            tzz[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 2, 2] * src_v[src_idx, t]
            txz[src_idx, src_z, src_x] += scaling_factor * MT[src_idx, 0, 2] * src_v[src_idx, t]

        # topFs with the assumption of weak anisotropy near the surface
        if free_surface:
            tzz[:,h-1,:] = 0
            tzz[:,h-2,:] = -tzz[:,h,:]
            txz[:,h-2,:] = -txz[:,h-1,:]
            txz[:,h-3,:] = -txz[:,h,:]
        
        # Update velocity components
        dxfm_txx = Dxfm(txx, fdc, ii, jj)
        dzbm_txz = Dzbm(txz, fdc, ii, jj)
        dxbm_txz = Dxbm(txz, fdc, ii, jj)
        dzfm_tzz = Dzfm(tzz, fdc, ii, jj)
        vx[:, idx_i, idx_j] += dt * bx[idx_i, idx_j] * (dxfm_txx / dx + dzbm_txz / dz)
        vz[:, idx_i, idx_j] += dt * bz[idx_i, idx_j] * (dxbm_txz / dx + dzfm_tzz / dz)

        # Apply free surface boundary conditions
        if free_surface:
            vz[:, h - 2, idx_j] = vz[:, h - 1, idx_j]
            vx[:, h - 2, idx_j] = vz[:, h - 2, j_start + 1:j_end + 1] - vz[:, h - 2, idx_j] + vz[:, h - 1, j_start + 1:j_end + 1] - vz[:, h - 1, idx_j] + vx[:, h, idx_j]
            vz[:, h - 3, idx_j] = vz[:, h - 2, idx_j]

        # Apply damping
        vx *= damp
        vz *= damp
        
        # Record receiver waveforms
        rcv_txx[:, t, rcv_idx] = txx[:, rcv_z, rcv_x]
        rcv_tzz[:, t, rcv_idx] = tzz[:, rcv_z, rcv_x]
        rcv_txz[:, t, rcv_idx] = txz[:, rcv_z, rcv_x]
        rcv_vx[:, t, rcv_idx] = vx[:, rcv_z, rcv_x]
        rcv_vz[:, t, rcv_idx] = vz[:, rcv_z, rcv_x]
        
        # Store forward wavefields for visualization or further processing
        forward_wavefield_txx += torch.sum(txx * txx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_tzz += torch.sum(tzz * tzz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_txz += torch.sum(txz * txz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vx += torch.sum(vx * vx, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()
        forward_wavefield_vz += torch.sum(vz * vz, dim=0)[offset:offset + nz, nabc:nabc + nx].detach()

    return txx,tzz,txz,vx,vz,rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz,\
            forward_wavefield_txx,forward_wavefield_tzz,forward_wavefield_txz,forward_wavefield_vx,forward_wavefield_vz


##########################################################################
#                       forward Modeling    
##########################################################################
def forward_kernel( nx:int,nz:int,dx:float,dz:float,nt:int,dt:float,
                    nabc:int,free_surface:bool,                                         # Model settings
                    src_x:Tensor,src_z:Tensor,src_n:int,src_v:Tensor,MT:Tensor,         # Source
                    rcv_x:Tensor,rcv_z:Tensor,rcv_n:int,                                # Receiver
                    abc_type:str,bcx:Tensor,bcz:Tensor,damp:Tensor,                     # PML/ABL
                    lamu:Tensor,lam:Tensor,bx:Tensor,bz:Tensor,                         # lame constant
                    CC:List[Tensor],                                                    # elastic moduli
                    fd_order=4,n_segments = 1,                                          # Finite Difference
                    device: torch.device = torch.device("cpu"), dtype=torch.float32
                ):
    # free surface offset
    fs_offset = fd_order//2
    
    # forward simulation
    nx_pml = nx + 2 * nabc
    nz_pml = nz + (nabc + fs_offset if free_surface else 2 * nabc + fs_offset)
    
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
    vx,vz,txx,tzz,txz           = torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device)
    vx_x,vz_x,txx_x,tzz_x,txz_x = torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device)
    vx_z,vz_z,txx_z,tzz_z,txz_z = torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device),torch.zeros((src_n,nz_pml,nx_pml),dtype=dtype,device=device)
    rcv_txx,rcv_tzz,rcv_txz,rcv_vx,rcv_vz = torch.zeros((src_n,nt,rcv_n),dtype=dtype,device=device),torch.zeros((src_n,nt,rcv_n),dtype=dtype,device=device),torch.zeros((src_n,nt,rcv_n),dtype=dtype,device=device),torch.zeros((src_n,nt,rcv_n),dtype=dtype,device=device),torch.zeros((src_n,nt,rcv_n),dtype=dtype,device=device)

    forward_wavefield_txx = torch.zeros((nz,nx),dtype=dtype,device=device)
    forward_wavefield_tzz = torch.zeros((nz,nx),dtype=dtype,device=device)
    forward_wavefield_txz = torch.zeros((nz,nx),dtype=dtype,device=device)
    forward_wavefield_vx  = torch.zeros((nz,nx),dtype=dtype,device=device)
    forward_wavefield_vz  = torch.zeros((nz,nx),dtype=dtype,device=device)
    
    # checkpoints for saving memory
    k = 0
    for i, chunk in enumerate(torch.chunk(src_v,n_segments,dim=-1)):
        if abc_type.lower() in ["pml"]:
            txx_x,txx_z,tzz_x,tzz_z,txz_x,txz_z,txx,tzz,txz,\
            vx_x,vx_z,vz_x,vz_z,vx,vz,\
            rcv_txx_temp,rcv_tzz_temp,rcv_txz_temp,rcv_vx_temp,rcv_vz_temp,\
            forward_wavefield_txx_temp,forward_wavefield_tzz_temp,forward_wavefield_txz_temp,forward_wavefield_vx_temp,forward_wavefield_vz_temp \
                                                                            = checkpoint(step_forward_PML_10order if fd_order == 10 else step_forward_PML_8order if fd_order == 8 else step_forward_PML_6order if fd_order == 6 else step_forward_PML_4order,
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
                                                                                device,dtype,
                                                                                use_reentrant=True)
        else:
            txx,tzz,txz,vx,vz,\
            rcv_txx_temp,rcv_tzz_temp,rcv_txz_temp,rcv_vx_temp,rcv_vz_temp,\
            forward_wavefield_txx_temp,forward_wavefield_tzz_temp,forward_wavefield_txz_temp,forward_wavefield_vx_temp,forward_wavefield_vz_temp \
                                                                            = checkpoint(step_forward_ABL_10order if fd_order == 10 else step_forward_ABL_8order if fd_order == 8 else step_forward_ABL_6order if fd_order == 6 else step_forward_ABL_4order,
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
                                                                                device,dtype,
                                                                                use_reentrant=True)
        # save the waveform recorded on receiver
        rcv_txx[:,k:k+chunk.shape[-1]] = rcv_txx_temp
        rcv_tzz[:,k:k+chunk.shape[-1]] = rcv_tzz_temp
        rcv_txz[:,k:k+chunk.shape[-1]] = rcv_txz_temp
        rcv_vx[:,k:k+chunk.shape[-1]]  =  rcv_vx_temp
        rcv_vz[:,k:k+chunk.shape[-1]]  =  rcv_vz_temp
        
        # save the forward wavefield
        forward_wavefield_txx += forward_wavefield_txx_temp.detach()
        forward_wavefield_tzz += forward_wavefield_tzz_temp.detach()
        forward_wavefield_txz += forward_wavefield_txz_temp.detach()
        forward_wavefield_vx  += forward_wavefield_vx_temp.detach()
        forward_wavefield_vz  += forward_wavefield_vz_temp.detach()
        
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