import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import numpy as np

def pad_torch(v:torch.Tensor,pml:int,nz:int,nx:int,ns:int,device:str):
    nz_pml = nz+2*pml
    nx_pml = nx+2*pml
    cc = torch.zeros((ns,nz_pml,nx_pml)).to(device)
    cc[:,pml:nz_pml-pml,pml:nx_pml-pml] = v
    with torch.no_grad():
        cc[:,list(range(0,pml)),pml:pml+nx] = torch.ones_like(cc[:,list(range(0,pml)),pml:pml+nx])*cc[:,[pml],pml:pml+nx]
        cc[:,list(range(nz_pml-pml,nz_pml)),pml:pml+nx] = torch.ones_like(cc[:,list(range(nz_pml-pml,nz_pml)),pml:pml+nx])*cc[:,[nz_pml-pml-1],pml:pml+nx]
        cc[:,:,list(range(0,pml))] = cc[:,:,[pml]]
        cc[:,:,list(range(nx_pml-pml,nx_pml))] = cc[:,:,[nx_pml-pml-1]]
    return cc

def pad_torchSingle(v:torch.Tensor,pml:int,nz:int,nx:int,ns:int,device:str):
    nz_pml = nz+2*pml
    nx_pml = nx+2*pml
    cc = torch.zeros((nz_pml,nx_pml)).to(device)
    cc[pml:nz_pml-pml,pml:nx_pml-pml] = v
    with torch.no_grad():
        cc[list(range(0,pml)),pml:pml+nx] = torch.ones_like(cc[list(range(pml)),pml:pml+nx])*cc[[pml],pml:pml+nx]
        cc[list(range(nz_pml-pml,nz_pml)),pml:pml+nx] = torch.ones_like(cc[list(range(nz_pml-pml,nz_pml)),pml:pml+nx])*cc[[nz_pml-pml-1],pml:pml+nx]
        cc[:,list(range(0,pml))] = cc[:,[pml]]
        cc[:,list(range(nx_pml-pml,nx_pml))] = cc[:,[nx_pml-pml-1]]
    return cc

def step_forward(nx:int,nz:int,dx:float,dz:float,dt:float,
                nabc:int,free_surface:bool,                               # Model settings
                src_x:np.array,src_z:np.array,src_n:int,src_v:Tensor,     # Source
                rcv_x:np.array,rcv_z:np.array,rcv_n:int,                  # Receiver
                kappa1:Tensor,alpha1:Tensor,kappa2:Tensor,alpha2:Tensor,kappa3:Tensor,c1_staggered:float,c2_staggered:float,
                p:Tensor,u:Tensor,w:Tensor,
                device="cpu",dtype=torch.float32,
                ):
    """
    Description
    --------------
        Forward Simulation with one time step for 2-order Acoustic Waveform Equation 
    
    Prameters:
    --------------
        free_surface (bool)             : free-surface or not
        nx (int)                        : grids number along the X-axis
        nz (int)                        : grids number along the Z-axis
        dx (float)                      : grids spacing along the X-axis
        dz (float)                      : grids spacing along the Z-axis
        dt (float)                      : time spacing (unit:s)
        src_x (ndarray)                 : source location in the X-axis
        src_z (ndarray)                 : source location in the Z-axis
        src_n (ndarray)                 : the number of the source
        src_v (Tensor)                  : wavelets for each source
        rcv_x (ndarray)                 : receiver location in the X-axis
        rcv_z (ndarray)                 : receiver location in the Z-axis
        rcv_n (ndarray)                 : the number of the receiver
        damp (Tensor)                   : boundary condition along the X and Z 
        kappa1 (Tensor)                 : temp variable for forward simulation
        alpha1 (Tensor)                 : temp variable for forward simulation
        kappa2 (Tensor)                 : temp variable for forward simulation
        alpha2 (Tensor)                 : temp variable for forward simulation
        kappa3 (Tensor)                 : temp variable for forward simulation
        c1_staggered (float)            : 2nd-order finite difference coefficient
        c2_staggered (float)            : 2nd-order finite difference coefficient
        p (Tensor)                      : Pressure
        u (Tensor)                      : vertical velocity (vx)
        w (Tensor)                      : horizontal velocity (vz)
        device (str)                    : device
        dtype (torch dtypes)            : dtypes for tensors
    
    returns:
    ------------------
        p (Tensor)                      : Pressure
        u (Tensor)                      : vertical velocity (vx)
        w (Tensor)                      : horizontal velocity (vz)
        rcv_p (Tensor)                  : recorded p(pressure) on the receivers
        rcv_u (Tensor)                  : recorded u(velocity component) on the receivers
        rcv_w (Tensor)                  : recorded w(velocity component) on the receivers
        forward_wavefield_p (Tensor)    : forward wavefield of p(pressure) 
        forward_wavefield_u (Tensor)    : forward wavefield of u(velocity component) 
        forward_wavefield_w (Tensor)    : forward wavefield of w(velocity component) 
    """
    nt = src_v.shape[-1]
    if free_surface:
        free_surface_start = nabc
    else:
        free_surface_start = 1
    nx_pml = nx + 2*nabc
    nz_pml = nz + 2*nabc
    
    p,u,w = torch.ones_like(p)*p,torch.ones_like(u)*u,torch.ones_like(w)*w
    
    rcv_p,rcv_u,rcv_w = torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device),torch.zeros((src_n,nt,rcv_n),dtype=dtype).to(device)
    
    forward_wavefield_p = torch.zeros((nz,nx),dtype=torch.float32).to(device)
    forward_wavefield_u = torch.zeros((nz,nx),dtype=torch.float32).to(device)
    forward_wavefield_w = torch.zeros((nz,nx),dtype=torch.float32).to(device)
    
    for it in range(nt):
        # Update the pressure       
        p[:,free_surface_start+1:nz_pml-2,2:nx_pml-2] = \
            (1.0-kappa1[free_surface_start+1:nz_pml-2,2:nx_pml-2])*p[:,free_surface_start+1:nz_pml-2,2:nx_pml-2] - \
            alpha1[free_surface_start+1:nz_pml-2,2:nx_pml-2]*(\
            c1_staggered*(u[:,free_surface_start+1:nz_pml-2,2:nx_pml-2] - u[:,free_surface_start+1:nz_pml-2,1:nx_pml-3] + w[:,free_surface_start+1:nz_pml-2,2:nx_pml-2] - w[:,free_surface_start:nz_pml-3,2:nx_pml-2]) + \
            c2_staggered*(u[:,free_surface_start+1:nz_pml-2,3:nx_pml-1] - u[:,free_surface_start+1:nz_pml-2,0:nx_pml-4] + w[:,free_surface_start+2:nz_pml-1,2:nx_pml-2] - w[:,free_surface_start-1:nz_pml-4,2:nx_pml-2]))
        
        # Add source
        if len(src_v.shape) == 1:
            p[list(range(src_n)),src_z,src_x] = p[list(range(src_n)),src_z,src_x] + dt*src_v[it]
        else:
            p[list(range(src_n)),src_z,src_x] = p[list(range(src_n)),src_z,src_x] + dt*src_v[:,it]
            
        # Free surface
        if free_surface:
            p[:,free_surface_start-1,:] = -p[:,free_surface_start+1,:]
        
        # Update horizontal particle velocity: u
        u[:,free_surface_start:nz_pml-1,1:nx_pml-2] = \
            (1.0 - kappa2[free_surface_start:nz_pml-1,1:nx_pml-2])*u[:,free_surface_start:nz_pml-1,1:nx_pml-2] - \
            alpha2[free_surface_start:nz_pml-1,1:nx_pml-2]*( \
            c1_staggered*(p[:,free_surface_start:nz_pml-1,2:nx_pml-1] - p[:,free_surface_start:nz_pml-1,1:nx_pml-2]) + 
            c2_staggered*(p[:,free_surface_start:nz_pml-1,3:nx_pml]   - p[:,free_surface_start:nz_pml-1,0:nx_pml-3]))

        # Update verticle particle velocity: w
        w[:,free_surface_start:nz_pml-2,1:nx_pml-1] = \
            (1.0 - kappa3[free_surface_start:nz_pml-2,1:nx_pml-1])*w[:,free_surface_start:nz_pml-2,1:nx_pml-1] - \
            alpha2[free_surface_start:nz_pml-2,1:nx_pml-1]*(\
                c1_staggered*(p[:,free_surface_start+1:nz_pml-1,1:nx_pml-1] - p[:,free_surface_start:nz_pml-2  ,1:nx_pml-1]) +\
                c2_staggered*(p[:,free_surface_start+2:nz_pml,1:nx_pml-1]   - p[:,free_surface_start-1:nz_pml-3,1:nx_pml-1])
            )
        # Free surface
        if free_surface:
            w[:,free_surface_start-1,list(range(0,nx_pml))] = w[:,free_surface_start,list(range(0,nx_pml))]
        
        # Output pressure seismogram
        rcv_p[:,it,list(range(rcv_n))] = p[:,rcv_z,rcv_x]
        rcv_u[:,it,list(range(rcv_n))] = u[:,rcv_z,rcv_x]
        rcv_w[:,it,list(range(rcv_n))] = w[:,rcv_z,rcv_x]
        with torch.no_grad():
            forward_wavefield_p = forward_wavefield_p + torch.sum(p*p,dim=0)[nabc:nabc+nz,nabc:nabc+nx]
            forward_wavefield_u = forward_wavefield_u + torch.sum(u*u,dim=0)[nabc:nabc+nz,nabc:nabc+nx]
            forward_wavefield_w = forward_wavefield_w + torch.sum(w*w,dim=0)[nabc:nabc+nz,nabc:nabc+nx]
    return p,u,w,rcv_p,rcv_u,rcv_w,forward_wavefield_p,forward_wavefield_u,forward_wavefield_w
        
def forward_kernel(nx:int,nz:int,dx:float,dz:float,nt:int,dt:float,
                    nabc:int,free_surface:bool,                               # Model settings
                    src_x:np.array,src_z:np.array,src_n:int,src_v:Tensor,     # Source
                    rcv_x:np.array,rcv_z:np.array,rcv_n:int,                  # Receiver
                    damp:Tensor,                                              # PML
                    v:Tensor,rho:Tensor,                                                    # velocity model
                    checkpoint_segments = 1,                                           # Finite Difference
                    device='cpu',dtype=torch.float32
                    ):
    """ Forward simulation of Acoustic Waveform Equation

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
        rcv_x (ndarray)                 : receiver location in the X-axis
        rcv_z (ndarray)                 : receiver location in the Z-axis
        rcv_n (ndarray)                 : the number of the receiver
        damp (Tensor)                   : boundary condition along both the X and Z-axis
        v (Tensor)                      : P-wave velocity (km/s)
        rho (Tensor)                    : density (kg/m^3)
        checkpoint_segments             : segments of the checkpoints for saving memory
        device (str)                    : device, Default "cpu"
        dtype (types)                   : dtypes, Default torch.float32
    
    Returns
    ---------------
        record_waveforms (dict)
            rcv_p (Tensor)                  : recorded p(pressure) on the receivers
            rcv_u (Tensor)                  : recorded u(velocity component) on the receivers
            rcv_w (Tensor)                  : recorded w(velocity component) on the receivers
            forward_wavefield_p (Tensor)    : forward wavefield of p(pressure) 
            forward_wavefield_u (Tensor)    : forward wavefield of u(velocity component) 
            forward_wavefield_w (Tensor)    : forward wavefield of w(velocity component) 
    """
    ###################################################################################
    c   = pad_torchSingle(v,nabc,nz,nx,src_n,device=device)
    den = pad_torchSingle(rho,nabc,nz,nx,src_n,device=device)
    
    if free_surface:
        free_surface_start = nabc
    else:
        free_surface_start = 1
    
    nx_pml = nx + 2*nabc
    nz_pml = nz + 2*nabc
    
    src_x = src_x + nabc
    src_z = src_z + nabc
    
    rcv_x = rcv_x + nabc
    rcv_z = rcv_z + nabc
    
    # Second order staggered grid finite difference forward modeling（Acoutic wave）
    p = torch.zeros((src_n,nz_pml,nx_pml),dtype=torch.float32).to(device=device)
    u = torch.zeros((src_n,nz_pml,nx_pml-1),dtype=torch.float32).to(device=device)
    w = torch.zeros((src_n,nz_pml-1,nx_pml),dtype=torch.float32).to(device=device)

    # record waveform
    rcv_p = torch.zeros((src_n,nt,rcv_n),dtype=torch.float32).to(device=device)
    rcv_u = torch.zeros((src_n,nt,rcv_n),dtype=torch.float32).to(device=device)
    rcv_w = torch.zeros((src_n,nt,rcv_n),dtype=torch.float32).to(device=device)
    forward_wavefield_p = torch.zeros((nz,nx),dtype=torch.float32).to(device)
    forward_wavefield_u = torch.zeros((nz,nx),dtype=torch.float32).to(device)
    forward_wavefield_w = torch.zeros((nz,nx),dtype=torch.float32).to(device)

    # 2-order staggred grid FD coef 
    c1_staggered =  9.0/8.0
    c2_staggered = -1.0/24.0
    
    # parameter for waveform simulation
    alpha1 = den*c*c*dt/dz  ## ！
    kappa1 = damp*dt
    
    alpha2 = dt/(den*dz)
    kappa2 = torch.zeros_like(damp).to(device)
    kappa2[:,1:nx_pml-2] = 0.5*(damp[:,1:nx_pml-2]+damp[:,2:nx_pml-1])*dt
    
    kappa3 = torch.zeros_like(damp).to(device)
    kappa3[free_surface_start:nz_pml-2,:] = 0.5*(damp[free_surface_start:nz_pml-2,:]+damp[free_surface_start+1:nz_pml-1,:])*dt
    
    
    k = 0
    for i, chunk in enumerate(torch.chunk(src_v,checkpoint_segments,dim=-1)):
        # step forward
        p,u,w,\
        rcv_p_temp,rcv_u_temp,rcv_w_temp,\
        forward_wavefield_p_temp,forward_wavefield_u_temp,forward_wavefield_w_temp = \
            checkpoint(step_forward,
                        nx,nz,dx,dz,dt,
                        nabc,free_surface,
                        src_x,src_z,src_n,chunk,
                        rcv_x,rcv_z,rcv_n,
                        kappa1,alpha1,kappa2,alpha2,kappa3,c1_staggered,c2_staggered,
                        p,u,w,
                        device,dtype)

        # save the waveform recorded on receiver
        rcv_p[:,k:k+chunk.shape[-1]] = rcv_p_temp
        rcv_u[:,k:k+chunk.shape[-1]] = rcv_u_temp
        rcv_w[:,k:k+chunk.shape[-1]] = rcv_w_temp

        # save the forward wavefield
        with torch.no_grad():
            forward_wavefield_p += forward_wavefield_p_temp
            forward_wavefield_u += forward_wavefield_u_temp
            forward_wavefield_w += forward_wavefield_w_temp
        k+=chunk.shape[-1]
    
    record_waveform = {
        "p":rcv_p,
        "u":rcv_u,
        "w":rcv_w,
        "forward_wavefield_p":forward_wavefield_p,
        "forward_wavefield_u":forward_wavefield_u,
        "forward_wavefield_w":forward_wavefield_w,
    }
    return record_waveform