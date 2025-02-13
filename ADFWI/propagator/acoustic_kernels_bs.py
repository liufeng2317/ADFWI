###########################################################################################
#  !!! Important
# Under development and testing: reducing memory footprint using boundary-saving methods
# [1] P. Yang, J. Gao, and B. Wang, RTM using effective boundary saving: A staggered grid GPU implementation, Comput. Geosci., vol. 68, pp. 64–72, Jul. 2014, doi: 10.1016/j.cageo.2014.04.004.
# [2] Wang, S., Jiang, Y., Song, P., Tan, J., Liu, Z. & He, B., 2023. Memory optimization in RNN-based full waveform inversion using boundary saving wavefield reconstruction. IEEE Trans. Geosci. Remote Sens., 61, 1–12. doi:10.1109/TGRS.2023.3317529
###########################################################################################
import torch
torch.autograd.set_detect_anomaly(True)
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Tuple,Dict
import itertools


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

# 2Nth order staggered grid finite difference : correct backward propagation needs 2N -1 points on one side
# the default is N = 3 (2N-1)
def save_boundaries(tensor: torch.Tensor, nabc: int, N: int=3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    top     =  tensor[:, nabc:nabc + N, :].clone()
    bottom  =  tensor[:, -(nabc + N):-nabc, :].clone()
    left    = tensor[..., nabc:nabc + N].clone() 
    right   = tensor[..., -(nabc + N):-nabc].clone()
    return top, bottom, left, right

def restore_boundaries(tensor: torch.Tensor, memory: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], nabc: int, N: int = 3) -> torch.Tensor:
    top, bottom, left, right        = memory
    # Restore boundaries from memory
    tensor[:, nabc:nabc + N, :]     = top
    tensor[:, -(nabc + N):-nabc, :] = bottom
    tensor[..., nabc:nabc + N]      = left
    tensor[..., -(nabc + N):-nabc]  = right
    return tensor

def _time_step(
        src_n: int, src_x: torch.Tensor, src_z: torch.Tensor,src_v: torch.Tensor, # source information
        nx_pml: int, nz_pml: int, nabc: int,                                                   
        free_surface: bool, free_surface_start: int,                    
        kappa1: torch.Tensor, alpha1: torch.Tensor, 
        kappa2: torch.Tensor, alpha2: torch.Tensor, 
        kappa3: torch.Tensor,  
        p: torch.Tensor, u: torch.Tensor, w: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # Coefficients for the staggered grid
    c1_staggered = 9.0 / 8.0
    c2_staggered = -1.0 / 24.0
    
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
    p[torch.arange(src_n), src_z, src_x] += src_v

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
        
    return p, u, w

def _time_step_backward(
        src_n: int, src_x: torch.Tensor, src_z: torch.Tensor,src_v: torch.Tensor, # source information
        nx_pml: int, nz_pml: int, nabc: int,                                                    
        free_surface: bool, free_surface_start: int,                    
        kappa1: torch.Tensor, alpha1: torch.Tensor, 
        kappa2: torch.Tensor, alpha2: torch.Tensor, 
        kappa3: torch.Tensor,  
        p_last: torch.Tensor, u_last: torch.Tensor, w_last: torch.Tensor,
        p_boundary_memory: torch.Tensor,u_boundary_memory: torch.Tensor,w_boundary_memory: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    p = p_last.clone()
    u = u_last.clone()
    w = w_last.clone()
    
    # Coefficients for the staggered grid
    c1_staggered = 9.0 / 8.0
    c2_staggered = -1.0 / 24.0
    
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
    p[torch.arange(src_n), src_z, src_x] += src_v

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

    with torch.no_grad():
        p = restore_boundaries(p, p_boundary_memory,nabc)
        u = restore_boundaries(u, u_boundary_memory,nabc)
        w = restore_boundaries(w, w_boundary_memory,nabc)

    return p, u, w

def packup_boundaries(flattened_list, tuple_size):
    return [tuple(flattened_list[i:i+tuple_size]) for i in range(0, len(flattened_list), tuple_size)]

class Checkpoint_TimeStep(torch.autograd.Function):
    wavefields = []
    counts=0
    @staticmethod
    def forward(ctx, 
                run_function,   # time_step forward
                back_function,  # time_step backward
                save_condition, # save the wavefield or not
                src_n: int, src_x: torch.Tensor, src_z: torch.Tensor,src_v: torch.Tensor, # source information
                nx_pml: int, nz_pml: int, nabc:int,                                                  
                free_surface: bool, free_surface_start: int,                    
                kappa1: torch.Tensor, alpha1: torch.Tensor, 
                kappa2: torch.Tensor, alpha2: torch.Tensor, 
                kappa3: torch.Tensor,  
                p: torch.Tensor, u: torch.Tensor, w: torch.Tensor
    ):  
        # require gradients or not
        requires_grad_list      = [False]*17
        for i,val in enumerate([kappa1,alpha1,kappa2,alpha2,kappa3,p,u,w]):
            if val.requires_grad:
                requires_grad_list[i+9] = True
        ctx.requires_grad_list  = requires_grad_list
        
        # forward simulation
        ctx.run_function        = run_function
        ctx.back_function       = back_function
        
        # save the forward parameters
        ctx.inputs_param = (src_n, src_x, src_z, src_v,
                            nx_pml, nz_pml, nabc,
                            free_surface, free_surface_start, 
                            kappa1, alpha1, kappa2, alpha2, kappa3
                            )
        
        # save the last image    
        ctx.save_condition      = save_condition
        
        with torch.no_grad():
            outputs = run_function(
                src_n, src_x, src_z, src_v,
                nx_pml, nz_pml, nabc,
                free_surface, free_surface_start, 
                kappa1, alpha1, kappa2, alpha2, kappa3, 
                p, u, w
            )
        
        # save boundary CheckpointFunction._bound.append(boundarys)
        boundarys = [save_boundaries(output,nabc) for output in outputs]
        ctx.save_for_backward(*itertools.chain(*boundarys))
        
        # Save the wavefields of the last time step
        ctx.is_last_time_step   = save_condition
        ctx.lastframe           = outputs if ctx.is_last_time_step else None
        return outputs
    
    @staticmethod
    def backward(ctx, *args):
        # the last wavefield
        if ctx.is_last_time_step:
            Checkpoint_TimeStep.wavefields = list(ctx.lastframe) # the last wavefields
            # Checkpoint_TimeStep.wavefields.reverse()             # reverse the wavefields
            wavefields = Checkpoint_TimeStep.wavefields
            return (None,None,None) + tuple(None for _ in range(len(ctx.requires_grad_list)))
        else:
            wavefields = Checkpoint_TimeStep.wavefields # get the wavefields

        Checkpoint_TimeStep.counts+=1
        
        #############################################################################
        # Inputs for backwards
        inputs = ctx.inputs_param + tuple(wavefields)
        # Process inputs to manage gradient requirements
        inputs_new = [
            inp.detach().requires_grad_(ctx.requires_grad_list[i]) if torch.is_tensor(inp) else inp
            for i, inp in enumerate(inputs)
        ]
        num_boundaries  = 4 
        boundaries      = packup_boundaries(ctx.saved_tensors, num_boundaries)
        # Combine new inputs with boundaries
        inputs = tuple(inputs_new) + tuple(boundaries)

        # reconstruct the wavefield
        with torch.enable_grad():
            outputs = ctx.back_function(*inputs)

        # calculate the gradients
        outputs_with_grad = []
        args_with_grad    = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary")

        torch.autograd.backward(outputs_with_grad, args_with_grad)
        
        # assign boundary values
        outputs = list(outputs)
        
        # Update wavefields
        if not (Checkpoint_TimeStep.counts == 1) or not Checkpoint_TimeStep.counts == 0:
            Checkpoint_TimeStep.wavefields.clear()
            Checkpoint_TimeStep.wavefields.extend(list(outputs))
        
        grads = (None,None,None) + tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in inputs[:len(ctx.requires_grad_list)])
        
        return grads


def step_forward(nx: int, nz: int, dx: float, dz: float, dt: float,
                 nabc: int, free_surface: bool,                                             # Model settings
                 src_x: torch.Tensor, src_z: torch.Tensor, src_n: int, src_v: torch.Tensor, # Source
                 rcv_x: torch.Tensor, rcv_z: torch.Tensor, rcv_n: int,                      # Receiver
                 damp: torch.Tensor, v: torch.Tensor, rho: torch.Tensor,                    # Pass in velocity and density to compute alpha, kappa
                 p: torch.Tensor, u: torch.Tensor, w: torch.Tensor,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    nt = src_v.shape[-1]
    free_surface_start = nabc if free_surface else 1
    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc

    # Initialize recorded values
    rcv_p               = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_u               = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_w               = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    # Initialize forward wavefield
    forward_wavefield_p = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_u = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_w = torch.zeros((nz, nx), dtype=dtype, device=device)
    
    # Dynamically compute alpha and kappa at each time step to avoid storing large tensors
    alpha1 = rho * v * v * dt / dz
    kappa1 = damp * dt
    
    alpha2 = dt / (rho * dz)
    kappa2 = torch.zeros_like(damp, device=device)
    kappa2[:, 1:nx_pml - 2] = 0.5 * (damp[:, 1:nx_pml - 2] + damp[:, 2:nx_pml - 1]) * dt
    
    kappa3 = torch.zeros_like(damp, device=device)
    kappa3[free_surface_start:nz_pml - 2, :] = 0.5 * (damp[free_surface_start:nz_pml - 2, :] + damp[free_surface_start + 1:nz_pml - 1, :]) * dt

    time_step_fun = Checkpoint_TimeStep()
    for it in range(nt):
        # sources item
        src_v_it = (src_v[it]*dt if len(src_v.shape) == 1 else src_v[:, it])
        
        # Update fields and record values
        # p, u, w = _time_step(
        #     src_n, src_x, src_z, src_v_it,
        #     nx_pml, nz_pml, 
        #     free_surface, free_surface_start, 
        #     kappa1, alpha1, kappa2, alpha2, kappa3, 
        #     p, u, w
        # )
        # onlye the alpha1 requires gradients
        p,u,w = time_step_fun.apply(_time_step,_time_step_backward,it == nt-1,
                                    src_n, src_x, src_z, src_v_it,
                                    nx_pml, nz_pml, nabc,
                                    free_surface, free_surface_start, 
                                    kappa1, alpha1, kappa2, alpha2, kappa3, 
                                    p, u, w
                                )
        
        # record pressure seismogram
        rcv_p[:, it, :] = p[:, rcv_z, rcv_x]
        rcv_u[:, it, :] = u[:, rcv_z, rcv_x]
        rcv_w[:, it, :] = w[:, rcv_z, rcv_x]

        # Accumulate forward wavefields
        forward_wavefield_p += torch.sum(p * p, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()
        forward_wavefield_u += torch.sum(u * u, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()
        forward_wavefield_w += torch.sum(w * w, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()

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
    ###################################################################################
    # Padding velocity and density fields
    c   = pad_torchSingle(v, nabc, nz, nx, src_n, device=device)
    den = pad_torchSingle(rho, nabc, nz, nx, src_n, device=device)
    
    free_surface_start = nabc if free_surface else 1
    
    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc
    
    # Adjust source and receiver positions for PML
    src_x = src_x + nabc
    src_z = src_z + nabc
    rcv_x = rcv_x + nabc
    rcv_z = rcv_z + nabc
    
    # Initialize pressure, velocity fields
    p = torch.zeros((src_n, nz_pml, nx_pml)    , dtype=dtype, device=device)
    u = torch.zeros((src_n, nz_pml, nx_pml - 1), dtype=dtype, device=device)
    w = torch.zeros((src_n, nz_pml - 1, nx_pml), dtype=dtype, device=device)

    # Initialize recorded waveforms
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_u = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_w = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    forward_wavefield_p = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_u = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_w = torch.zeros((nz, nx), dtype=dtype, device=device)

    k = 0
    for i, chunk in enumerate(torch.chunk(src_v, checkpoint_segments, dim=-1)):
        # Step forward: Calculate alpha, kappa inside step_forward
        p, u, w, rcv_p_temp, rcv_u_temp, rcv_w_temp, forward_wavefield_p_temp, forward_wavefield_u_temp, forward_wavefield_w_temp = \
            checkpoint(step_forward,
                       nx, nz, dx, dz, dt,
                       nabc, free_surface,
                       src_x, src_z, src_n, chunk,
                       rcv_x, rcv_z, rcv_n,
                       damp, c, den,  # Pass velocity, density, and damping (PML)
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



# def step_forward(nx: int, nz: int, dx: float, dz: float, dt: float,
#                  nabc: int, free_surface: bool,                                             # Model settings
#                  src_x: torch.Tensor, src_z: torch.Tensor, src_n: int, src_v: torch.Tensor, # Source
#                  rcv_x: torch.Tensor, rcv_z: torch.Tensor, rcv_n: int,                      # Receiver
#                  damp: torch.Tensor, v: torch.Tensor, rho: torch.Tensor,                    # Pass in velocity and density to compute alpha, kappa
#                  p: torch.Tensor, u: torch.Tensor, w: torch.Tensor,
#                  device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
#     nt = src_v.shape[-1]
#     free_surface_start = nabc if free_surface else 1
#     nx_pml = nx + 2 * nabc
#     nz_pml = nz + 2 * nabc

#     # Initialize recorded values
#     rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
#     rcv_u = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
#     rcv_w = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

#     # Initialize forward wavefield
#     forward_wavefield_p = torch.zeros((nz, nx), dtype=dtype, device=device)
#     forward_wavefield_u = torch.zeros((nz, nx), dtype=dtype, device=device)
#     forward_wavefield_w = torch.zeros((nz, nx), dtype=dtype, device=device)
    
#     # Dynamically compute alpha and kappa at each time step to avoid storing large tensors
#     alpha1 = rho * v * v * dt / dz
#     kappa1 = damp * dt
    
#     alpha2 = dt / (rho * dz)
#     kappa2 = torch.zeros_like(damp, device=device)
#     kappa2[:, 1:nx_pml - 2] = 0.5 * (damp[:, 1:nx_pml - 2] + damp[:, 2:nx_pml - 1]) * dt
    
#     kappa3 = torch.zeros_like(damp, device=device)
#     kappa3[free_surface_start:nz_pml - 2, :] = 0.5 * (damp[free_surface_start:nz_pml - 2, :] + damp[free_surface_start + 1:nz_pml - 1, :]) * dt

#     for it in range(nt):
#         # sources item
#         src_v_it = (src_v[it]*dt if len(src_v.shape) == 1 else src_v[:, it])
        
#         # Update fields and record values
#         p, u, w = _time_step(
#             src_n, src_x, src_z, src_v_it,
#             nx_pml, nz_pml, 
#             free_surface, free_surface_start, 
#             kappa1, alpha1, kappa2, alpha2, kappa3, 
#             p, u, w
#         )
        
#         # record pressure seismogram
#         rcv_p[:, it, :] = p[:, rcv_z, rcv_x]
#         rcv_u[:, it, :] = u[:, rcv_z, rcv_x]
#         rcv_w[:, it, :] = w[:, rcv_z, rcv_x]

#         # Accumulate forward wavefields
#         forward_wavefield_p += torch.sum(p * p, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()
#         forward_wavefield_u += torch.sum(u * u, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()
#         forward_wavefield_w += torch.sum(w * w, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()

#     return p, u, w, rcv_p, rcv_u, rcv_w, forward_wavefield_p, forward_wavefield_u, forward_wavefield_w




# def step_forward(nx: int, nz: int, dx: float, dz: float, dt: float,
#                  nabc: int, free_surface: bool,                               # Model settings
#                  src_x: torch.Tensor, src_z: torch.Tensor, src_n: int, src_v: torch.Tensor,     # Source
#                  rcv_x: torch.Tensor, rcv_z: torch.Tensor, rcv_n: int,                  # Receiver
#                  kappa1: torch.Tensor, alpha1: torch.Tensor, kappa2: torch.Tensor, alpha2: torch.Tensor,
#                  kappa3: torch.Tensor, c1_staggered: float, c2_staggered: float,
#                  p: torch.Tensor, u: torch.Tensor, w: torch.Tensor,
#                  device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

#     p = p.clone()
#     u = u.clone()
#     w = w.clone()
    
#     nt = src_v.shape[-1]
#     free_surface_start = nabc if free_surface else 1
#     nx_pml = nx + 2 * nabc
#     nz_pml = nz + 2 * nabc

#     # Initialize recorded values
#     rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
#     rcv_u = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
#     rcv_w = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

#     # Initialize forward wavefield
#     forward_wavefield_p = torch.zeros((nz, nx), dtype=dtype, device=device)
#     forward_wavefield_u = torch.zeros((nz, nx), dtype=dtype, device=device)
#     forward_wavefield_w = torch.zeros((nz, nx), dtype=dtype, device=device)
    
#     for it in range(nt):
#         # Update fields and record values
#         p, u, w = _time_step(
#             it, nx_pml, nz_pml, free_surface, free_surface_start, kappa1, alpha1, 
#             kappa2, alpha2, kappa3, c1_staggered, c2_staggered, dt, src_v, 
#             src_x, src_z, p, u, w, src_n, nabc, nx, nz
#         )
        
#         # Output pressure seismogram
#         rcv_p[:, it, :] = p[:, rcv_z, rcv_x]
#         rcv_u[:, it, :] = u[:, rcv_z, rcv_x]
#         rcv_w[:, it, :] = w[:, rcv_z, rcv_x]

#         # Accumulate forward wavefields
#         forward_wavefield_p += torch.sum(p * p, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()
#         forward_wavefield_u += torch.sum(u * u, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()
#         forward_wavefield_w += torch.sum(w * w, dim=0)[nabc:nabc + nz, nabc:nabc + nx].detach()

#     return p, u, w, rcv_p, rcv_u, rcv_w, forward_wavefield_p, forward_wavefield_u, forward_wavefield_w


# def forward_kernel(nx: int, nz: int, dx: float, dz: float, nt: int, dt: float,
#                    nabc: int, free_surface: bool,                               # Model settings
#                    src_x: torch.Tensor, src_z: torch.Tensor, src_n: int, src_v: torch.Tensor,     # Source
#                    rcv_x: torch.Tensor, rcv_z: torch.Tensor, rcv_n: int,                  # Receiver
#                    damp: torch.Tensor,                                              # PML
#                    v: torch.Tensor, rho: torch.Tensor,                                                    # Velocity model
#                    checkpoint_segments: int = 1,                                           # Finite Difference
#                    device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32
#                    ) -> Dict[str, torch.Tensor]:  # Changed return type to Dict for clarity
#     ###################################################################################
#     c   = pad_torchSingle(v, nabc, nz, nx, src_n, device=device)
#     den = pad_torchSingle(rho, nabc, nz, nx, src_n, device=device)
    
#     free_surface_start = nabc if free_surface else 1
    
#     nx_pml = nx + 2 * nabc
#     nz_pml = nz + 2 * nabc
    
#     src_x = src_x + nabc
#     src_z = src_z + nabc
    
#     rcv_x = rcv_x + nabc
#     rcv_z = rcv_z + nabc
    
#     # Initialize pressure, velocity fields
#     p = torch.zeros((src_n, nz_pml, nx_pml), dtype=dtype, device=device)
#     u = torch.zeros((src_n, nz_pml, nx_pml - 1), dtype=dtype, device=device)
#     w = torch.zeros((src_n, nz_pml - 1, nx_pml), dtype=dtype, device=device)

#     # Initialize recorded waveforms
#     rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
#     rcv_u = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
#     rcv_w = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
#     forward_wavefield_p = torch.zeros((nz, nx), dtype=dtype, device=device)
#     forward_wavefield_u = torch.zeros((nz, nx), dtype=dtype, device=device)
#     forward_wavefield_w = torch.zeros((nz, nx), dtype=dtype, device=device)

#     # Coefficients for the staggered grid
#     c1_staggered = 9.0 / 8.0
#     c2_staggered = -1.0 / 24.0
    
#     # Parameters for waveform simulation
#     alpha1 = den * c * c * dt / dz
#     kappa1 = damp * dt
    
#     alpha2 = dt / (den * dz)
#     kappa2 = torch.zeros_like(damp, device=device)
#     kappa2[:, 1:nx_pml - 2] = 0.5 * (damp[:, 1:nx_pml - 2] + damp[:, 2:nx_pml - 1]) * dt
    
#     kappa3 = torch.zeros_like(damp, device=device)
#     kappa3[free_surface_start:nz_pml - 2, :] = 0.5 * (damp[free_surface_start:nz_pml - 2, :] + damp[free_surface_start + 1:nz_pml - 1, :]) * dt
    
#     k = 0
#     for i, chunk in enumerate(torch.chunk(src_v, checkpoint_segments, dim=-1)):
#         # Step forward
#         p, u, w, rcv_p_temp, rcv_u_temp, rcv_w_temp, forward_wavefield_p_temp, forward_wavefield_u_temp, forward_wavefield_w_temp = \
#             checkpoint(step_forward,
#                        nx, nz, dx, dz, dt,
#                        nabc, free_surface,
#                        src_x, src_z, src_n, chunk,
#                        rcv_x, rcv_z, rcv_n,
#                        kappa1, alpha1, kappa2, alpha2, kappa3, c1_staggered, c2_staggered,
#                        p, u, w,
#                        device, dtype)

#         # Save the waveform recorded on the receiver
#         rcv_p[:, k:k + chunk.shape[-1]] = rcv_p_temp
#         rcv_u[:, k:k + chunk.shape[-1]] = rcv_u_temp
#         rcv_w[:, k:k + chunk.shape[-1]] = rcv_w_temp

#         # Accumulate the forward wavefield
#         forward_wavefield_p = forward_wavefield_p + forward_wavefield_p_temp.detach()
#         forward_wavefield_u = forward_wavefield_p + forward_wavefield_u_temp.detach()
#         forward_wavefield_w = forward_wavefield_p + forward_wavefield_w_temp.detach()
            
#         k = k + chunk.shape[-1]
    
#     record_waveform = {
#         "p": rcv_p,
#         "u": rcv_u,
#         "w": rcv_w,
#         "forward_wavefield_p": forward_wavefield_p,
#         "forward_wavefield_u": forward_wavefield_u,
#         "forward_wavefield_w": forward_wavefield_w,
#     }
    
#     return record_waveform