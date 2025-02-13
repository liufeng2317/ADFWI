import numpy as np
from math import log
import torch

def bc_pml(nx, nz, dx, dz, pml, vmax, free_surface=True):
    """
    Calculate the damping in both x and z directions for the Perfectly Matched Layer (PML).

    Parameters
    ----------
    nx : int
        Number of grid points along the x-axis.
    nz : int
        Number of grid points along the z-axis.
    dx : float
        Grid spacing along the x-axis.
    dz : float
        Grid spacing along the z-axis.
    pml : int
        The width of the PML region (number of grid points).
    vmax : float
        Maximum velocity in the model.
    free_surface : bool, optional
        If True, assume a free surface at the top of the model. Default is True.

    Returns
    -------
    damp_global : numpy.ndarray
        The damping profile for both x and z directions, applied across the grid.
    """
    if free_surface:
        nx_pml = nx + 2 * pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2 * pml
        nz_pml = nz + 2 * pml

    damp_global = np.zeros((nx_pml, nz_pml))
    damp = np.zeros(pml)
    a = (pml - 1) * dx

    # Adjust the damping effect.
    R = 1e-6
    kappa = -3.0 * vmax * np.log(R) / (2.0 * a)

    for ix in range(0, pml):
        xa = ix * dx / a
        damp[ix] = kappa * xa * xa

    for ix in range(0, pml):
        for iz in range(0, nz_pml):
            damp_global[pml - ix - 1, iz] = damp[ix]
            damp_global[nx_pml + ix - pml, iz] = damp[ix]

    for iz in range(0, pml):
        for ix in range((pml - (iz - 1)) - 1, nx_pml - (pml - iz)):
            if not free_surface:
                damp_global[ix, pml - iz - 1] = damp[iz]
            damp_global[ix, nz_pml + iz - pml] = damp[iz]

    return damp_global.T


# SinCos damp
def bc_sincos(nx, nz, dx, dz, pml, free_surface=False):
    """
    Set up a damping profile using a sine-cosine function for the Perfectly Matched Layer (PML).

    Parameters
    ----------
    nx : int
        Number of grid points along the x-axis.
    nz : int
        Number of grid points along the z-axis.
    dx : float
        Grid spacing along the x-axis.
    dz : float
        Grid spacing along the z-axis.
    pml : int
        The width of the PML region (number of grid points).
    free_surface : bool, optional
        If True, assume a free surface at the top of the model. Default is False.

    Returns
    -------
    damp : numpy.ndarray
        The damping profile applied across the grid in both x and z directions.
    """
    if free_surface:
        nx_pml = nx + 2 * pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2 * pml
        nz_pml = nz + 2 * pml

    damp = np.ones((nz_pml, nx_pml))

    for i in range(pml):
        if not free_surface:
            damp[i, :] *= np.sin(np.pi / 2 * i / pml) ** 2
        damp[-i - 1, :] *= np.sin(np.pi / 2 * i / pml) ** 2
        damp[:, i] *= np.sin(np.pi / 2 * i / pml) ** 2
        damp[:, -i - 1] *= np.sin(np.pi / 2 * i / pml) ** 2

    return damp

# FDWave3D damp
def bc_gerjan(nx, nz, dx, dz, pml, alpha=0.0053, free_surface=True):
    """
    PML implementation based on Gerjan et al., 1985, where the damping profile is defined as G = exp(a * [I - i]^2), with 'a' as the attenuation factor.

    Parameters
    ----------
    nx : int
        Number of grid points along the x-axis.
    nz : int
        Number of grid points along the z-axis.
    dx : float
        Grid spacing along the x-axis.
    dz : float
        Grid spacing along the z-axis.
    pml : int
        The width of the PML region (number of grid points).
    alpha : float, optional
        The attenuation factor, default is 0.0053.
    free_surface : bool, optional
        If True, assume a free surface at the top of the model. Default is True.

    Returns
    -------
    damp : numpy.ndarray
        The damping profile applied across the grid in both x and z directions.
    """
    wt = np.exp(-(alpha * (pml - np.arange(1, pml + 1))) ** 2)
    
    if free_surface:
        nx_pml = nx + 2 * pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2 * pml
        nz_pml = nz + 2 * pml

    damp = np.ones((nz_pml, nx_pml))

    if free_surface:
        for k in range(1, len(wt) + 1):
            damp[:nz_pml - k + 1, k - 1] = wt[k - 1]        # left
            damp[nz_pml - k, k - 1:nx_pml - k + 1] = wt[k - 1]  # bottom
            damp[:nz_pml - k + 1, nx_pml - k] = wt[k - 1]    # right
    else:
        for k in range(1, len(wt) + 1):
            damp[k - 1:nz_pml - k + 1, k - 1] = wt[k - 1]    # left
            damp[k:nz_pml - k + 1, nx_pml - k] = wt[k - 1]    # right
            damp[nz_pml - k, k - 1:nx_pml - k + 1] = wt[k - 1]  # bottom
            damp[k - 1, k - 1:nx_pml - k + 1] = wt[k - 1]      # top

    return damp


# FDWave3D ABCdamp
def bc_pml_xz(nx, nz, dx, dz, pml, vmax, free_surface=True):
    """
    Calculate the damping coefficients for both x and z directions using the PML (Perfectly Matched Layer) method. The damping is applied to the boundaries in both directions (x and z).

    Parameters
    ----------
    nx : int
        Number of grid points along the x-axis.
    nz : int
        Number of grid points along the z-axis.
    dx : float
        Grid spacing along the x-axis.
    dz : float
        Grid spacing along the z-axis.
    pml : int
        The width of the PML region (number of grid points).
    vmax : float
        Maximum wave velocity in the model.
    free_surface : bool, optional
        If True, assume a free surface at the top of the model. Default is True.

    Returns
    -------
    BCx : numpy.ndarray
        The damping profile applied to the x boundaries.
    BCz : numpy.ndarray
        The damping profile applied to the z boundaries.
    """
    if free_surface:
        nx_pml = nx + 2 * pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2 * pml
        nz_pml = nz + 2 * pml

    R = 1e-6  # Reflection coefficient
    ppml = -np.log(R) * 3 * vmax / (2 * pml ** 3)  # Damping coefficient

    BCx = np.zeros((nz_pml, nx_pml))  # Damping profile in x-direction
    BCz = np.zeros((nz_pml, nx_pml))  # Damping profile in z-direction

    if free_surface:
        for k in range(1, pml + 1):
            BCx[:, k - 1] = (pml - k + 1) ** 2 * ppml / dx  # Left boundary in x-direction
            BCx[:, nx_pml - k] = (pml - k + 1) ** 2 * ppml / dx  # Right boundary in x-direction
            BCz[nz_pml - k, :] = (pml - k + 1) ** 2 * ppml / dz  # Bottom boundary in z-direction
    else:
        for k in range(1, pml + 1):
            BCx[:, k - 1] = (pml - k + 1) ** 2 * ppml / dx  # Left boundary in x-direction
            BCx[:, nx_pml - k] = (pml - k + 1) ** 2 * ppml / dx  # Right boundary in x-direction
            BCz[k - 1, :] = (pml - k + 1) ** 2 * ppml / dz  # Top boundary in z-direction
            BCz[nz_pml - k, :] = (pml - k + 1) ** 2 * ppml / dz  # Bottom boundary in z-direction

    return BCx, BCz
