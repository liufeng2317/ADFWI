'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-06-27 19:15:22
* LastEditors: LiuFeng
* LastEditTime: 2023-12-12 18:34:52
* FilePath: /Acoustic_AD/TorchInversion/kernels.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def pad_torch(v:torch.Tensor,pml:int,nx:int,ny:int,ns:int,device:str):
    nx_pml = nx+2*pml
    ny_pml = ny+2*pml
    cc = torch.zeros((ns,nx_pml,ny_pml)).to(device)
    cc[:,pml:nx_pml-pml,pml:ny_pml-pml] = v
    with torch.no_grad():
        cc[:,list(range(0,pml)),pml:pml+ny] = torch.ones_like(cc[:,list(range(0,pml)),pml:pml+ny])*cc[:,[pml],pml:pml+ny]
        cc[:,list(range(nx_pml-pml,nx_pml)),pml:pml+ny] = torch.ones_like(cc[:,list(range(nx_pml-pml,nx_pml)),pml:pml+ny])*cc[:,[nx_pml-pml-1],pml:pml+ny]
        cc[:,:,list(range(0,pml))] = cc[:,:,[pml]]
        cc[:,:,list(range(ny_pml-pml,ny_pml))] = cc[:,:,[ny_pml-pml-1]]
    return cc

def pad_torchSingle(v:torch.Tensor,pml:int,nx:int,ny:int,ns:int,device:str):
    nx_pml = nx+2*pml
    ny_pml = ny+2*pml
    cc = torch.zeros((nx_pml,ny_pml)).to(device)
    cc[pml:nx_pml-pml,pml:ny_pml-pml] = v
    with torch.no_grad():
        cc[list(range(0,pml)),pml:pml+ny] = torch.ones_like(cc[list(range(pml)),pml:pml+ny])*cc[[pml],pml:pml+ny]
        cc[list(range(nx_pml-pml,nx_pml)),pml:pml+ny] = torch.ones_like(cc[list(range(nx_pml-pml,nx_pml)),pml:pml+ny])*cc[[nx_pml-pml-1],pml:pml+ny]
        cc[:,list(range(0,pml))] = cc[:,[pml]]
        cc[:,list(range(ny_pml-pml,ny_pml))] = cc[:,[ny_pml-pml-1]]
    return cc

def acoustic_FM2_kernel(nx:int, ny:int, dx:float, dy:float,
                        nt:int, dt:float, pml:int, fs:int,
                        nx_pml:int, ny_pml:int, damp_global:torch.Tensor,
                        src_x:torch.Tensor, src_y:torch.Tensor, src_n:int,st:torch.Tensor,
                        rcv_x:torch.Tensor, rcv_y:torch.Tensor, rcv_n:int,
                        v:torch.Tensor, rho:torch.Tensor, device:str):
    ###################################################################################
    c = pad_torchSingle(v,pml,nx,ny,src_n,device=device)
    den = pad_torchSingle(rho,pml,nx,ny,src_n,device=device)
    damp_global = damp_global
    
    # Second order staggered grid finite difference forward modeling（Acoutic wave）
    p = torch.zeros((src_n,nx_pml,ny_pml),dtype=torch.float32).to(device=device)
    u = torch.zeros((src_n,nx_pml,ny_pml-1),dtype=torch.float32).to(device=device)
    w = torch.zeros((src_n,nx_pml-1,ny_pml),dtype=torch.float32).to(device=device)

    # record waveform
    csg = torch.zeros((src_n,nt,rcv_n),dtype=torch.float32).to(device=device)
    forw = torch.zeros((nx,ny),dtype=torch.float32).to(device)

    # 2-order staggred grid FD coef 
    c1_staggered = 9.0/8.0
    c2_staggered = -1.0/24.0
    
    # parameter for waveform simulation
    alpha1 = den*c*c*dt/dx  ## ！
    kappa1 = damp_global*dt
    
    alpha2 = dt/(den*dx)
    kappa2 = torch.zeros_like(damp_global).to(device)
    kappa2[:,1:ny_pml-2] = 0.5*(damp_global[:,1:ny_pml-2]+damp_global[:,2:ny_pml-1])*dt
    
    kappa3 = torch.zeros_like(damp_global).to(device)
    kappa3[pml:nx_pml-2,:] = 0.5*(damp_global[pml:nx_pml-2,:]+damp_global[pml+1:nx_pml-1,:])*dt
    for it in range(1,nt):
        # Update the pressure        
        p[:,pml+1:nx_pml-2,2:ny_pml-2] = \
            (1.0-kappa1[pml+1:nx_pml-2,2:ny_pml-2])*p[:,pml+1:nx_pml-2,2:ny_pml-2] - \
            alpha1[pml+1:nx_pml-2,2:ny_pml-2]*(\
            c1_staggered*(u[:,pml+1:nx_pml-2,2:ny_pml-2] - u[:,pml+1:nx_pml-2,1:ny_pml-3] + w[:,pml+1:nx_pml-2,2:ny_pml-2] - w[:,pml:nx_pml-3,2:ny_pml-2]) + \
            c2_staggered*(u[:,pml+1:nx_pml-2,3:ny_pml-1] - u[:,pml+1:nx_pml-2,0:ny_pml-4] + w[:,pml+2:nx_pml-1,2:ny_pml-2] - w[:,pml-1:nx_pml-4,2:ny_pml-2]))

        # Add source
        if len(st.shape) == 1:
            p[list(range(src_n)),src_x,src_y] = p[list(range(src_n)),src_x,src_y] + dt*st[it]
        else:
            p[list(range(src_n)),src_x,src_y] = p[list(range(src_n)),src_x,src_y] + dt*st[:,it]
            
        # Free surface
        if fs==1:
            p[:,pml-1,:] = -p[:,pml+1,:]
        
        # Update horizontal particle velocity: u
        u[:,pml:nx_pml-1,1:ny_pml-2] = \
            (1.0 - kappa2[pml:nx_pml-1,1:ny_pml-2])*u[:,pml:nx_pml-1,1:ny_pml-2] - \
            alpha2[pml:nx_pml-1,1:ny_pml-2]*( \
            c1_staggered*(p[:,pml:nx_pml-1,2:ny_pml-1] - p[:,pml:nx_pml-1,1:ny_pml-2]) + 
            c2_staggered*(p[:,pml:nx_pml-1,3:ny_pml] - p[:,pml:nx_pml-1,0:ny_pml-3]))

        # Update verticle particle velocity: w
        w[:,pml:nx_pml-2,1:ny_pml-1] = \
            (1.0 - kappa3[pml:nx_pml-2,1:ny_pml-1])*w[:,pml:nx_pml-2,1:ny_pml-1] - \
            alpha2[pml:nx_pml-2,1:ny_pml-1]*(\
                c1_staggered*(p[:,pml+1:nx_pml-1,1:ny_pml-1] - p[:,pml:nx_pml-2,1:ny_pml-1]) +\
                c2_staggered*(p[:,pml+2:nx_pml,1:ny_pml-1] - p[:,pml-1:nx_pml-3,1:ny_pml-1])
            )
        
        # 
        if fs == 1:
            w[:,pml-1,list(range(0,ny_pml))] = w[:,pml,list(range(0,ny_pml))]
        
        # Output pressure seismogram
        igx = pml + rcv_x[list(range(0,rcv_n))]
        igz = pml + rcv_y[list(range(0,rcv_n))]
        mask = (igx == pml)
        igx[mask] = igx[mask]+1
        csg[:,it,list(range(0,rcv_n))] = p[:,igx,igz]
        
        with torch.no_grad():
            forw = forw + torch.sum(p*p,dim=0)[pml:pml+nx,pml:pml+ny]
        
        # 暂存波场
        # with torch.no_grad():
        #     if it%20==0:
        #         plt.figure()
        #         plt.imshow(p[0,:,:].cpu().detach().numpy())
        #         plt.savefig("/media/liufeng/a0b205ec-bfb3-473f-a6f0-0680c5da64ba/project/004_inversion/ADInversion/Acoustic/Acoustic_AD/TestADinversion/data/01_test_gradient/snapshot/{}.png".format(it),bbox_inches='tight')
        #         plt.close()
            
    return csg,forw