import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Tuple,Dict

@torch.jit.script
def pad_torchSingle(v: torch.Tensor, pml: int, nz: int, nx: int, ns: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    nz_pml = nz + 2 * pml
    nx_pml = nx + 2 * pml
    cc = torch.zeros((nz_pml, nx_pml), device=device)
    
    # Copy the original tensor to the appropriate position
    cc[pml:nz_pml - pml, pml:nx_pml - pml] = v

    # Handle the top boundary
    cc[:pml, pml:pml + nx] = cc[pml, pml:pml + nx].expand(pml, -1)
    
    # Handle the bottom boundary
    cc[nz_pml - pml:nz_pml, pml:pml + nx] = cc[nz_pml - pml - 1, pml:pml + nx].expand(pml, -1)

    # Handle the left boundary
    cc[:, :pml] = cc[:, [pml]].expand(-1, pml)

    # Handle the right boundary
    cc[:, nx_pml - pml:nx_pml] = cc[:, [nx_pml - pml - 1]].expand(-1, pml)

    return cc


@torch.jit.script
def step_forward(nx: int, nz: int, dx: float, dz: float, dt: float,
                 nabc: int, free_surface: bool,                               # Model settings
                 src_x: torch.Tensor, src_z: torch.Tensor, src_n: int, src_v: torch.Tensor,     # Source
                 rcv_x: torch.Tensor, rcv_z: torch.Tensor, rcv_n: int,                  # Receiver
                 kappa1: torch.Tensor, alpha1: torch.Tensor, kappa2: torch.Tensor, alpha2: torch.Tensor,
                 kappa3: torch.Tensor, c1_staggered: float, c2_staggered: float,
                 p: torch.Tensor, u: torch.Tensor, w: torch.Tensor,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Description
    --------------
        Forward Simulation with one time step for 2-order Acoustic Waveform Equation 

    Parameters:
    --------------
        free_surface (bool)             : whether there is a free-surface
        nx (int)                        : number of grid points along the X-axis
        nz (int)                        : number of grid points along the Z-axis
        dx (float)                      : grid spacing along the X-axis
        dz (float)                      : grid spacing along the Z-axis
        dt (float)                      : time spacing (unit: s)
        src_x (Tensor)                  : source location in the X-axis
        src_z (Tensor)                  : source location in the Z-axis
        src_n (Tensor)                  : the number of sources
        src_v (Tensor)                  : wavelets for each source
        rcv_x (Tensor)                  : receiver location in the X-axis
        rcv_z (Tensor)                  : receiver location in the Z-axis
        rcv_n (Tensor)                  : the number of receivers
        kappa1 (Tensor)                 : temporary variable for forward simulation
        alpha1 (Tensor)                 : temporary variable for forward simulation
        kappa2 (Tensor)                 : temporary variable for forward simulation
        alpha2 (Tensor)                 : temporary variable for forward simulation
        kappa3 (Tensor)                 : temporary variable for forward simulation
        c1_staggered (float)            : 2nd-order finite difference coefficient
        c2_staggered (float)            : 2nd-order finite difference coefficient
        p (Tensor)                      : pressure
        u (Tensor)                      : vertical velocity (vx)
        w (Tensor)                      : horizontal velocity (vz)
        device (str)                    : device type
        dtype (torch.dtype)             : data type for tensors
    
    Returns:
    ------------------
        p (Tensor)                      : pressure
        u (Tensor)                      : vertical velocity (vx)
        w (Tensor)                      : horizontal velocity (vz)
        rcv_p (Tensor)                  : recorded pressure at receivers
        rcv_u (Tensor)                  : recorded vertical velocity at receivers
        rcv_w (Tensor)                  : recorded horizontal velocity at receivers
        forward_wavefield_p (Tensor)    : forward wavefield of pressure
        forward_wavefield_u (Tensor)    : forward wavefield of vertical velocity
        forward_wavefield_w (Tensor)    : forward wavefield of horizontal velocity
    """
    p = p.clone()
    u = u.clone()
    w = w.clone()
    
    nt = src_v.shape[-1]
    free_surface_start = nabc if free_surface else 1
    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc

    # Initialize recorded values
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_u = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_w = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    # Initialize forward wavefield
    forward_wavefield_p = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_u = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_w = torch.zeros((nz, nx), dtype=dtype, device=device)

    for it in range(nt):
        # Update pressure
        p[:, free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2] = (
            (1.0 - kappa1[free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2]) * 
            p[:, free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2] - 
            alpha1[free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2] * (
                c1_staggered * (u[:, free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2] -
                                u[:, free_surface_start + 1:nz_pml - 2, 1:nx_pml - 3] +
                                w[:, free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2] -
                                w[:, free_surface_start:nz_pml - 3, 2:nx_pml - 2]) +
                c2_staggered * (u[:, free_surface_start + 1:nz_pml - 2, 3:nx_pml - 1] -
                                u[:, free_surface_start + 1:nz_pml - 2, 0:nx_pml - 4] +
                                w[:, free_surface_start + 2:nz_pml - 1, 2:nx_pml - 2] -
                                w[:, free_surface_start - 1:nz_pml - 4, 2:nx_pml - 2])
            )
        )

        # Add source
        src_update = dt * (src_v[it] if len(src_v.shape) == 1 else src_v[:, it])
        p[torch.arange(src_n), src_z, src_x] = p[torch.arange(src_n), src_z, src_x] + src_update

        # Free surface handling
        if free_surface:
            p[:, free_surface_start - 1, :] = -p[:, free_surface_start + 1, :]

        # Update horizontal particle velocity: u
        u[:, free_surface_start:nz_pml - 1, 1:nx_pml - 2] = (
            (1.0 - kappa2[free_surface_start:nz_pml - 1, 1:nx_pml - 2]) * 
            u[:, free_surface_start:nz_pml - 1, 1:nx_pml - 2] - 
            alpha2[free_surface_start:nz_pml - 1, 1:nx_pml - 2] * (
                c1_staggered * (p[:, free_surface_start:nz_pml - 1, 2:nx_pml - 1] -
                                p[:, free_surface_start:nz_pml - 1, 1:nx_pml - 2]) +
                c2_staggered * (p[:, free_surface_start:nz_pml - 1, 3:nx_pml] -
                                p[:, free_surface_start:nz_pml - 1, 0:nx_pml - 3])
            )
        )

        # Update vertical particle velocity: w
        w[:, free_surface_start:nz_pml - 2, 1:nx_pml - 1] = (
            (1.0 - kappa3[free_surface_start:nz_pml - 2, 1:nx_pml - 1]) *
            w[:, free_surface_start:nz_pml - 2, 1:nx_pml - 1] - 
            alpha2[free_surface_start:nz_pml - 2, 1:nx_pml - 1] * (
                c1_staggered * (p[:, free_surface_start + 1:nz_pml - 1, 1:nx_pml - 1] -
                                p[:, free_surface_start:nz_pml - 2, 1:nx_pml - 1]) +
                c2_staggered * (p[:, free_surface_start + 2:nz_pml, 1:nx_pml - 1] -
                                p[:, free_surface_start - 1:nz_pml - 3, 1:nx_pml - 1])
            )
        )

        # Free surface for vertical velocity
        if free_surface:
            w[:, free_surface_start - 1, :] = w[:, free_surface_start, :]

        # Output pressure seismogram
        rcv_p[:, it, :] = p[:, rcv_z, rcv_x]
        rcv_u[:, it, :] = u[:, rcv_z, rcv_x]
        rcv_w[:, it, :] = w[:, rcv_z, rcv_x]

        # Accumulate forward wavefields
        forward_wavefield_p = forward_wavefield_p + torch.sum(p * p, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()
        forward_wavefield_u = forward_wavefield_u + torch.sum(u * u, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()
        forward_wavefield_w = forward_wavefield_u + torch.sum(w * w, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()

    return p, u, w, rcv_p, rcv_u, rcv_w, forward_wavefield_p, forward_wavefield_u, forward_wavefield_w


def forward_kernel(nx: int, nz: int, dx: float, dz: float, nt: int, dt: float,
                   nabc: int, free_surface: bool,                               # Model settings
                   src_x: torch.Tensor, src_z: torch.Tensor, src_n: int, src_v: torch.Tensor,     # Source
                   rcv_x: torch.Tensor, rcv_z: torch.Tensor, rcv_n: int,                  # Receiver
                   damp: torch.Tensor,                                              # PML
                   v: torch.Tensor, rho: torch.Tensor,                                                    # Velocity model
                   checkpoint_segments: int = 1,                                           # Finite Difference
                   device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32
                   ) -> Dict[str, torch.Tensor]:  # Changed return type to Dict for clarity
    """ Forward simulation of Acoustic Waveform Equation

    Parameters:
    --------------
        nx (int)                        : Number of grid points along the X-axis
        nz (int)                        : Number of grid points along the Z-axis
        dx (float)                      : Grid spacing along the X-axis
        dz (float)                      : Grid spacing along the Z-axis
        nt (int)                        : Number of time points for recording waveforms 
        dt (float)                      : Time spacing (unit:s)
        nabc (int)                      : Number of absorbing boundary condition
        free_surface (bool)             : Indicates if there's a free surface
        src_x (Tensor)                  : Source locations along the X-axis
        src_z (Tensor)                  : Source locations along the Z-axis
        src_n (int)                     : Number of sources
        src_v (Tensor)                  : Wavelets for each source
        rcv_x (Tensor)                  : Receiver locations along the X-axis
        rcv_z (Tensor)                  : Receiver locations along the Z-axis
        rcv_n (int)                     : Number of receivers
        damp (Tensor)                   : Damping tensor for the absorbing boundary
        v (Tensor)                      : P-wave velocity (km/s)
        rho (Tensor)                    : Density (kg/m^3)
        checkpoint_segments (int)       : Segments of the checkpoints for saving memory
        device (str)                    : Device type, default is "cpu"
        dtype (torch.dtype)             : Data type for tensors, default is torch.float32
    
    Returns:
    ---------------
        record_waveform (dict)          : Dictionary containing recorded waveforms and forward wavefields
            - rcv_p (Tensor)            : Recorded pressure at the receivers
            - rcv_u (Tensor)            : Recorded vertical velocity at the receivers
            - rcv_w (Tensor)            : Recorded horizontal velocity at the receivers
            - forward_wavefield_p (Tensor): Forward wavefield of pressure
            - forward_wavefield_u (Tensor): Forward wavefield of vertical velocity
            - forward_wavefield_w (Tensor): Forward wavefield of horizontal velocity
    """
    ###################################################################################
    c = pad_torchSingle(v, nabc, nz, nx, src_n, device=device)
    den = pad_torchSingle(rho, nabc, nz, nx, src_n, device=device)
    
    free_surface_start = nabc if free_surface else 1
    
    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc
    
    src_x = src_x + nabc
    src_z = src_z + nabc
    
    rcv_x = rcv_x + nabc
    rcv_z = rcv_z + nabc
    
    # Initialize pressure, velocity fields
    p = torch.zeros((src_n, nz_pml, nx_pml), dtype=dtype, device=device)
    u = torch.zeros((src_n, nz_pml, nx_pml - 1), dtype=dtype, device=device)
    w = torch.zeros((src_n, nz_pml - 1, nx_pml), dtype=dtype, device=device)

    # Initialize recorded waveforms
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_u = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_w = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    forward_wavefield_p = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_u = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_w = torch.zeros((nz, nx), dtype=dtype, device=device)

    # Coefficients for the staggered grid
    c1_staggered = 9.0 / 8.0
    c2_staggered = -1.0 / 24.0
    
    # Parameters for waveform simulation
    alpha1 = den * c * c * dt / dz
    kappa1 = damp * dt
    
    alpha2 = dt / (den * dz)
    kappa2 = torch.zeros_like(damp, device=device)
    kappa2[:, 1:nx_pml - 2] = 0.5 * (damp[:, 1:nx_pml - 2] + damp[:, 2:nx_pml - 1]) * dt
    
    kappa3 = torch.zeros_like(damp, device=device)
    kappa3[free_surface_start:nz_pml - 2, :] = 0.5 * (damp[free_surface_start:nz_pml - 2, :] + damp[free_surface_start + 1:nz_pml - 1, :]) * dt
    
    k = 0
    for i, chunk in enumerate(torch.chunk(src_v, checkpoint_segments, dim=-1)):
        # Step forward
        p, u, w, rcv_p_temp, rcv_u_temp, rcv_w_temp, forward_wavefield_p_temp, forward_wavefield_u_temp, forward_wavefield_w_temp = \
            checkpoint(step_forward,
                       nx, nz, dx, dz, dt,
                       nabc, free_surface,
                       src_x, src_z, src_n, chunk,
                       rcv_x, rcv_z, rcv_n,
                       kappa1, alpha1, kappa2, alpha2, kappa3, c1_staggered, c2_staggered,
                       p, u, w,
                       device, dtype)

        # Save the waveform recorded on the receiver
        rcv_p[:, k:k + chunk.shape[-1]] = rcv_p_temp
        rcv_u[:, k:k + chunk.shape[-1]] = rcv_u_temp
        rcv_w[:, k:k + chunk.shape[-1]] = rcv_w_temp

        # Accumulate the forward wavefield
        forward_wavefield_p = forward_wavefield_p + forward_wavefield_p_temp.detach()
        forward_wavefield_u = forward_wavefield_p + forward_wavefield_u_temp.detach()
        forward_wavefield_w = forward_wavefield_p + forward_wavefield_w_temp.detach()
            
        k = k + chunk.shape[-1]
    
    record_waveform = {
        "p": rcv_p,
        "u": rcv_u,
        "w": rcv_w,
        "forward_wavefield_p": forward_wavefield_p,
        "forward_wavefield_u": forward_wavefield_u,
        "forward_wavefield_w": forward_wavefield_w,
    }
    
    return record_waveform
