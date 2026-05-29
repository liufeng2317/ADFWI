"""Boundary damping profile builders for propagator wrappers.

This module returns NumPy arrays used by acoustic and elastic propagator
wrappers. It does not own wavefield time stepping, source injection, receiver
sampling, or FWI loss behavior.
"""

import numpy as np
from math import log


def bc_pml(nx,nz,dx,dz,pml,vmax,free_surface=True):
    """Return the scalar PML damping profile used by acoustic propagation."""
    if free_surface:
        nx_pml = nx + 2*pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2*pml
        nz_pml = nz + 2*pml
    damp_global = np.zeros((nx_pml,nz_pml))
    damp        = np.zeros(pml)
    a = (pml-1)*dx
    # Adjust the damping effect.
    R = 1e-6
    # R = 1e-3
    kappa = -3.0*vmax*log(R)/(2.0*a)
    for ix in range(0,pml):
        xa = ix*dx/a
        damp[ix] = kappa*xa*xa
        
    for ix in range(0,pml):
        for iz in range(0,nz_pml):
            damp_global[pml-ix-1,iz]            = damp[ix]
            damp_global[nx_pml+ix-pml,iz]       = damp[ix]

    for iz in range(0,pml):
        for ix in range((pml-(iz-1))-1,nx_pml-(pml-(iz))):
            if not free_surface:
                damp_global[ix,pml-iz-1]        = damp[iz]
            damp_global[ix,nz_pml+iz-pml]       = damp[iz]
            
    return damp_global.T

def bc_sincos(nx,nz,dx,dz,pml,free_surface=False):
    """Return a SinCos multiplicative damping profile."""
    if free_surface:
        nx_pml = nx + 2*pml
        nz_pml = nz +   pml
    else:
        nx_pml = nx + 2*pml
        nz_pml = nz + 2*pml
    damp = np.ones((nz_pml, nx_pml))

    for i in range(pml):
        if not free_surface:
            damp[i, :]      *=  np.sin(np.pi/2 * i/pml)**2
        damp[-i-1, :]   *=  np.sin(np.pi/2 * i/pml)**2
        damp[:, i]      *=  np.sin(np.pi/2 * i/pml)**2
        damp[:, -i-1]   *=  np.sin(np.pi/2 * i/pml)**2

    return damp

def bc_gerjan(nx,nz,dx,dz,pml,alpha=0.0053,free_surface=True):
    """Return a Gerjan-style damping profile.

    The formula follows the current ADFWI implementation of the Gerjan et al.
    attenuation profile and is intentionally unchanged in this readability
    pass.
    """
    wt = np.exp(-(alpha*(pml-np.arange(1,pml+1)))**2)
    if free_surface:
        nx_pml = nx + 2*pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2*pml
        nz_pml = nz + 2*pml
    damp = np.ones((nz_pml,nx_pml))
    if free_surface:
        for k in range(1,len(wt)+1):
            damp[:nz_pml-k+1,k-1]               = wt[k-1]        # left
            damp[nz_pml-k   ,k-1:nx_pml-k+1]    = wt[k-1]        # bottom
            damp[:nz_pml-k+1,nx_pml-k]          = wt[k-1]        # right
    else:
        for k in range(1,len(wt)+1):
            damp[k-1:nz_pml-k+1 ,k-1]            = wt[k-1]        # left
            damp[k:nz_pml-k+1   ,nx_pml-k]       = wt[k-1]        # right
            damp[nz_pml-k       ,k-1:nx_pml-k+1] = wt[k-1]        # bottom
            damp[k-1            ,k-1:nx_pml-k+1] = wt[k-1]        # top
    return damp

def bc_pml_xz(nx,nz,dx,dz,pml,vmax,free_surface=True):
    """Return separate x/z PML damping profiles used by elastic propagation."""
    if free_surface:
        nx_pml = nx + 2*pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2*pml
        nz_pml = nz + 2*pml
    R = 1e-6
    ppml = -np.log(R)*3*vmax/(2*pml**3)
    BCx = np.zeros((nz_pml,nx_pml))
    BCz = np.zeros((nz_pml,nx_pml))
    if free_surface:
        for k in range(1,pml+1):
            BCx[:,k-1]           = (pml-k+1)**2*ppml/dx
            BCx[:,nx_pml-k]      = (pml-k+1)**2*ppml/dx
            BCz[nz_pml-k,:]      = (pml-k+1)**2*ppml/dz
    else:
        for k in range(1,pml+1):
            BCx[:,k-1]           = (pml-k+1)**2*ppml/dx
            BCx[:,nx_pml-k]      = (pml-k+1)**2*ppml/dx
            BCz[k-1,:]           = (pml-k+1)**2*ppml/dz
            BCz[nz_pml-k,:]      = (pml-k+1)**2*ppml/dz
    return BCx,BCz
