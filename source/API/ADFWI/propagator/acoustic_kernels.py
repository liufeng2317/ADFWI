
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Tuple,Dict

@torch.jit.script
def pad_torchSingle(v: torch.Tensor, pml: int, nz: int, nx: int, ns: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Pads the input tensor `v` with a PML (Perfectly Matched Layer) boundary.

    This function expands the input tensor `v` by adding a PML layer on each side 
    (top, bottom, left, and right) and returns the padded tensor.

    Parameters
    ----------
    v : torch.Tensor
        The input tensor to be padded, typically representing a grid of values in a simulation.
        
    pml : int
        The number of grid points to be added as the PML boundary layer on each side of the tensor.
        
    nz : int
        The number of grid points in the Z-direction of the original tensor.
        
    nx : int
        The number of grid points in the X-direction of the original tensor.
        
    ns : int
        Not used in the function, could be intended for a future feature or an error.
        
    device : torch.device, optional
        The device on which the tensor is allocated (default is CPU). It can be set to either "cpu" or "cuda".

    Returns
    -------
    torch.Tensor
        A tensor with dimensions `(nz + 2*pml, nx + 2*pml)` where the original tensor `v` is copied 
        into the center and the boundaries are filled according to the PML conditions.
    """
    
    # Calculate the size of the padded tensor
    nz_pml = nz + 2 * pml
    nx_pml = nx + 2 * pml
    
    # Initialize the padded tensor with zeros
    cc = torch.zeros((nz_pml, nx_pml), device=device)
    
    # Copy the original tensor to the appropriate position in the center
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
    Forward simulation step for the 2nd-order Acoustic Waveform Equation using finite differences.

    This function performs one time step of the forward simulation, updating the pressure and velocities 
    (both vertical and horizontal components) based on the input wavefields and model parameters.

    Parameters
    ----------
    free_surface : bool
        Whether the model includes a free-surface boundary condition.

    nx : int
        Number of grid points in the X-direction (horizontal axis).

    nz : int
        Number of grid points in the Z-direction (vertical axis).

    dx : float
        Grid spacing in the X-direction (in meters).

    dz : float
        Grid spacing in the Z-direction (in meters).

    dt : float
        Time step for the simulation (in seconds).

    src_x : torch.Tensor
        A tensor of source locations in the X-direction (of shape [src_n]).

    src_z : torch.Tensor
        A tensor of source locations in the Z-direction (of shape [src_n]).

    src_n : int
        The number of source points.

    src_v : torch.Tensor
        A tensor of wavelets for each source, specifying the source function for each time step.

    rcv_x : torch.Tensor
        A tensor of receiver locations in the X-direction (of shape [rcv_n]).

    rcv_z : torch.Tensor
        A tensor of receiver locations in the Z-direction (of shape [rcv_n]).

    rcv_n : int
        The number of receiver points.

    kappa1 : torch.Tensor
        Temporary variable used in the forward simulation, representing a physical property in the model.

    alpha1 : torch.Tensor
        Temporary variable used in the forward simulation, representing a physical property in the model.

    kappa2 : torch.Tensor
        Temporary variable used in the forward simulation, representing a physical property in the model.

    alpha2 : torch.Tensor
        Temporary variable used in the forward simulation, representing a physical property in the model.

    kappa3 : torch.Tensor
        Temporary variable used in the forward simulation, representing a physical property in the model.

    c1_staggered : float
        Coefficient used for 2nd-order finite difference in the staggered grid scheme.

    c2_staggered : float
        Coefficient used for 2nd-order finite difference in the staggered grid scheme.

    p : torch.Tensor
        Tensor representing the pressure wavefield (of shape [nz, nx]).

    u : torch.Tensor
        Tensor representing the vertical velocity component (vx) (of shape [nz, nx]).

    w : torch.Tensor
        Tensor representing the horizontal velocity component (vz) (of shape [nz, nx]).

    device : torch.device, optional
        The device on which tensors are allocated. Defaults to the CPU.

    dtype : torch.dtype, optional
        The data type for the tensors (default is `torch.float32`).

    Returns
    -------
    p : torch.Tensor
        Updated pressure wavefield after the time step (of shape [nz, nx]).

    u : torch.Tensor
        Updated vertical velocity wavefield after the time step (of shape [nz, nx]).

    w : torch.Tensor
        Updated horizontal velocity wavefield after the time step (of shape [nz, nx]).

    rcv_p : torch.Tensor
        Recorded pressure values at the receiver locations for this time step (of shape [rcv_n]).

    rcv_u : torch.Tensor
        Recorded vertical velocity values at the receiver locations for this time step (of shape [rcv_n]).

    rcv_w : torch.Tensor
        Recorded horizontal velocity values at the receiver locations for this time step (of shape [rcv_n]).

    forward_wavefield_p : torch.Tensor
        The forward wavefield of pressure at the current time step, useful for visualization or further processing.

    forward_wavefield_u : torch.Tensor
        The forward wavefield of vertical velocity at the current time step.

    forward_wavefield_w : torch.Tensor
        The forward wavefield of horizontal velocity at the current time step.

    Notes
    -----
    - The function assumes the use of staggered grids for pressure and velocity components.
    - The input wavefield tensors (p, u, w) are updated at each time step according to the 2nd-order acoustic wave equation.
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

    wavefields = []
    
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
        # single source
        if src_z.dim() == 1:
            src_update = dt * (src_v[it] if len(src_v.shape) == 1 else src_v[:, it])
            p[torch.arange(src_n), src_z, src_x] = p[torch.arange(src_n), src_z, src_x] + src_update
        else:
        # encoded source
            for i in range(src_n):
                src_update = dt * (src_v[i,it] if len(src_v.shape) == 2 else src_v[i,:, it])
                p[i,src_z[i],src_x[i]] = p[i,src_z[i],src_x[i]] + src_update
        
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
        forward_wavefield_w = forward_wavefield_w + torch.sum(w * w, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()
        
    # if you want to save the wavefield, you need to comments the @torch.jit.script
    #     if it % 10 == 0:
    #         wavefields.append(p[:,nabc:nabc + nz, nabc:nabc + nx].cpu().detach().numpy())
    # wavefields = np.array(wavefields)
    # np.savez(f"/ailab/user/liufeng1/project/04_Inversion/ADFWI-github/examples/acoustic/01-model-test/01-Marmousi2/data/wavefield/wavefields.npz",data=wavefields)
    
    return p, u, w, rcv_p, rcv_u, rcv_w, forward_wavefield_p, forward_wavefield_u, forward_wavefield_w


def forward_kernel(nx: int, nz: int, dx: float, dz: float, nt: int, dt: float,
                   nabc: int, free_surface: bool,                                                   # Model settings
                   src_x: torch.Tensor, src_z: torch.Tensor, src_n: int, src_v: torch.Tensor,       # Source
                   rcv_x: torch.Tensor, rcv_z: torch.Tensor, rcv_n: int,                            # Receiver
                   damp: torch.Tensor,                                                              # PML
                   v: torch.Tensor, rho: torch.Tensor,                                              # Velocity model
                   checkpoint_segments: int = 1,                                                    # Finite Difference
                   device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32
                   ) -> Dict[str, torch.Tensor]:  # Changed return type to Dict for clarity
    """
    Forward simulation of Acoustic Waveform Equation.

    Parameters
    ----------
    nx : int
        Number of grid points along the X-axis.
    nz : int
        Number of grid points along the Z-axis.
    dx : float
        Grid spacing along the X-axis.
    dz : float
        Grid spacing along the Z-axis.
    nt : int
        Number of time points for recording waveforms.
    dt : float
        Time spacing (unit: s).
    nabc : int
        Number of absorbing boundary conditions.
    free_surface : bool
        Indicates if there's a free surface.
    src_x : torch.Tensor
        Source locations along the X-axis.
    src_z : torch.Tensor
        Source locations along the Z-axis.
    src_n : int
        Number of sources.
    src_v : torch.Tensor
        Wavelets for each source.
    rcv_x : torch.Tensor
        Receiver locations along the X-axis.
    rcv_z : torch.Tensor
        Receiver locations along the Z-axis.
    rcv_n : int
        Number of receivers.
    damp : torch.Tensor
        Damping tensor for the absorbing boundary.
    v : torch.Tensor
        P-wave velocity (km/s).
    rho : torch.Tensor
        Density (kg/m^3).
    checkpoint_segments : int, optional
        Segments of the checkpoints for saving memory (default is 1).
    device : torch.device, optional
        Device type, default is "cpu".
    dtype : torch.dtype, optional
        Data type for tensors, default is torch.float32.

    Returns
    -------
    record_waveform : dict
        Dictionary containing recorded waveforms and forward wavefields: Recorded vertical velocity at the receivers, Recorded horizontal velocity at the receivers, Forward wavefield of pressure, Forward wavefield of vertical velocity, Forward wavefield of horizontal velocity.
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
                       device, dtype, 
                       use_reentrant=True
                       )

        # Save the waveform recorded on the receiver
        rcv_p[:, k:k + chunk.shape[-1]] = rcv_p_temp
        rcv_u[:, k:k + chunk.shape[-1]] = rcv_u_temp
        rcv_w[:, k:k + chunk.shape[-1]] = rcv_w_temp

        # Accumulate the forward wavefield
        forward_wavefield_p = forward_wavefield_p + forward_wavefield_p_temp.detach()
        forward_wavefield_u = forward_wavefield_u + forward_wavefield_u_temp.detach()
        forward_wavefield_w = forward_wavefield_w + forward_wavefield_w_temp.detach()
            
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
