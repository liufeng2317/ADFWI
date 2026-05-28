"""Formula-sensitive physical transforms used by elastic model containers.

Unless noted otherwise, inputs are backend tensors with shape ``(nz, nx)``.
Velocity is in m/s, density is in kg/m^3, stiffness/moduli are in Pa,
buoyancy is ``1 / density``, and Thomsen parameters are dimensionless.
"""

import torch
from ADFWI.utils import numpy2tensor

def thomsen_init(vp,vs,rho,eps,delta,gamma,device,dtype=torch.float32):
    """Convert Thomsen-style model arrays to backend tensors."""
    vp      = numpy2tensor(vp,dtype).to(device)
    vs      = numpy2tensor(vs,dtype).to(device)
    rho     = numpy2tensor(rho,dtype).to(device)
    eps     = numpy2tensor(eps,dtype).to(device)
    delta   = numpy2tensor(delta,dtype).to(device)
    gamma   = numpy2tensor(gamma,dtype).to(device)
    return vp,vs,rho,eps,delta,gamma

def elastic_moduli_init(nz,nx,device,dtype=torch.float32):
    """Return zero tensors for the 21 upper-triangle stiffness components.

    The returned list order is::

        C11 C12 C13 C14 C15 C16
            C22 C23 C24 C25 C26
                C33 C34 C35 C36
                    C44 C45 C46
                        C55 C56
                            C66
    """
    # The 21 independent elastic parameters for full anisotropic models
    C11,C12,C13,C14,C15,C16 = torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device)
    C22,C23,C24,C25,C26     = torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device)
    C33,C34,C35,C36         = torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device)
    C44,C45,C46             = torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device)
    C55,C56                 = torch.zeros((nz,nx),dtype=dtype).to(device),torch.zeros((nz,nx),dtype=dtype).to(device)
    C66                     = torch.zeros((nz,nx),dtype=dtype).to(device)
    CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66]
    return CC

def vs_vp_to_Lame(vp,vs,rho):
    """Convert velocity and density to Lame-related elastic quantities.

    Parameters
    ----------
        vp : P-wave velocity.
        vs : S-wave velocity.
        rho: Density.

    Returns
    -------
        mu  : Shear modulus, ``rho * vs**2``.
        lamu: P-wave modulus, ``rho * vp**2 = lambda + 2 * mu``.
        lam : First Lame parameter, ``lamu - 2 * mu``.
        b   : Buoyancy, ``1 / rho``.
    """
    mu   = vs**2*rho
    lamu = vp**2*rho
    lam  = lamu-2*mu
    b    = 1/rho
    return mu,lamu,lam,b


def thomsen_to_elastic_moduli(alpha_thomsen,beta_thomsen,rho,eps_thomsen,delta_thomsen,gamma_thomsen):
    """Convert Thomsen parameters to the five TI stiffness components.

    For transverse isotropy, the independent components used here are
    ``C11``, ``C13``, ``C33``, ``C44``, and ``C66``. The full VTI matrix can be
    completed with ``C12 = C11 - 2 * C66``, ``C22 = C11``, ``C23 = C13``,
    and ``C55 = C44``.

    Parameters
    ----------
        alpha_thomsen: P-wave velocity along the symmetry axis.
        beta_thomsen : S-wave velocity along the symmetry axis.
        rho          : Density.
        eps_thomsen  : Thomsen epsilon.
        delta_thomsen: Thomsen delta.
        gamma_thomsen: Thomsen gamma.

    Returns
    -------
        C11, C13, C33, C44, C66 : Stiffness components in Pa.
    """
    C33 = alpha_thomsen**2*rho
    C44 =  beta_thomsen**2*rho
    C11 = C33*(1+2*eps_thomsen)
    C66 = C44*(1+2*gamma_thomsen)
    C13 = torch.sqrt(2*C33*(C33-C44)*delta_thomsen + (C33-C44)**2) - C44
    return C11,C13,C33,C44,C66

def elastic_moduli_to_thomsen(C11,C13,C33,C44,C66,rho):
    """Convert five TI stiffness components back to Thomsen parameters.

    Parameters
    ----------
        C11, C13, C33, C44, C66 : Stiffness components in Pa.
        rho : Density.
    
    Returns
    -------
        alpha_thomsen: P-wave velocity along the symmetry axis.
        beta_thomsen : S-wave velocity along the symmetry axis.
        eps_thomsen  : Thomsen epsilon.
        delta_thomsen: Thomsen delta.
        gamma_thomsen: Thomsen gamma.
    """
    alpha_thomsen   = torch.sqrt(C33/rho)
    beta_thomsen    = torch.sqrt(C44/rho)
    eps_thomsen     = (C11 - C33)/(2*C33)
    gamma_thomsen   = (C66 - C44)/(2*C44)
    delta_thomsen   = ((C13+C44)**2 - (C33-C44)**2)/(2*(C33)*(C33 - C44))
    return alpha_thomsen,beta_thomsen,eps_thomsen,delta_thomsen,gamma_thomsen

def elastic_moduli_for_isotropic(CC):
    """Complete the stiffness list for an isotropic elastic model.

    The input list follows the order returned by ``elastic_moduli_init`` and
    already contains the primary components from the current velocity/density
    tensors. This helper enforces isotropic symmetry by setting ``C22=C11``,
    ``C55=C44``, ``C23=C13``, and ``C12=C11 - 2*C66``.
    """
    [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66] = CC
    C22 = C11.clone()
    C55 = C44.clone()
    C23 = C13.clone()
    C12 = (C11 - 2*C66).clone()
    CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66]
    return CC

def elastic_moduli_for_TI(CC,anisotropic_type="VTI"):
    """Complete the stiffness list for VTI or HTI transverse isotropy.

    For VTI, the five independent components are ``C11``, ``C13``, ``C33``,
    ``C44``, and ``C66``. For HTI, this helper rotates the VTI assignment into
    the horizontal transverse-isotropy convention used by the elastic
    propagator.
    """
    [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66] = CC
    C22 = C11.clone()
    C55 = C44.clone()
    C23 = C13.clone()
    C12 = (C11 - 2*C66).clone()
    # HTI Rotated by VTI anticlockwise (Y) pi/2
    if anisotropic_type.lower() in ["vti"]:
        CC = [C11,C12,C13,C14,C15,C16,
                  C22,C23,C24,C25,C26,
                      C33,C34,C35,C36,
                          C44,C45,C46,
                              C55,C56,
                                  C66]
        # [C11,C12,C13,C15,C22,C23,C33,C35,C44,C55,C66]
    elif anisotropic_type.lower() in ["hti"]:
        CC = [C33,C13,C13,C14,C15,C16,
                  C11,C12,C24,C25,C26,
                      C11,C34,C35,C36,
                          C66,C45,C46,
                              C55,C56,
                                  C55]
        # [C33,C13,C13,C15,C11,C12,C11,C35,C66,C55,C55]
    return CC


def parameter_staggered_grid(mu,b,C44,C55,C66,nx,nz):
    """Prepare buoyancy and shear/stiffness fields on staggered grids.

    Parameters
    ----------
        mu  : Shear modulus on the regular ``(nz, nx)`` grid.
        b   : Buoyancy on the regular ``(nz, nx)`` grid.
        C44 : Stiffness component on the regular ``(nz, nx)`` grid.
        C55 : Stiffness component on the regular ``(nz, nx)`` grid.
        C66 : Stiffness component on the regular ``(nz, nx)`` grid.
        nx  : Number of x grid points.
        nz  : Number of z grid points.

    Returns
    -------
        bx  : Buoyancy averaged in x, shape ``(nz, nx - 1)``.
        bz  : Buoyancy averaged in z, shape ``(nz - 1, nx)``.
        muxz: Shear modulus on the x-z staggered grid, shape ``(nz - 2, nx - 2)``.
        C44 : Staggered ``C44``, shape ``(nz - 2, nx - 2)``.
        C55 : Staggered ``C55``, shape ``(nz - 2, nx - 2)``.
        C66 : Staggered ``C66``, shape ``(nz - 2, nx - 2)``.
    """
    bx   = 0.5*(b[:,0:nx-1]+b[:,1:nx])
    bz   = 0.5*(b[0:nz-1,:]+b[1:nz,:])

    muxz = 0.2*(mu[1:nz-1,1:nx-1]  + mu[2:nz,1:nx-1] + mu[1:nz-1,2:nx]      +\
                mu[2:nz  ,1:nx-1]  + mu[2:nz,2:nx])

    C44  =  0.2*(C44[1:nz-1,1:nx-1] + C44[2:nz,1:nx-1] + C44[1:nz-1,2:nx]    +\
                C44[2:nz  ,1:nx-1] + C44[2:nz,2:nx])

    C55  =  0.2*(C55[1:nz-1,1:nx-1] + C55[2:nz,1:nx-1] + C55[1:nz-1,2:nx]    +\
                C55[2:nz  ,1:nx-1] + C55[2:nz,2:nx])

    C66  =  0.2*(C66[1:nz-1,1:nx-1] + C66[2:nz,1:nx-1] + C66[1:nz-1,2:nx]    +\
                C66[2:nz  ,1:nx-1] + C66[2:nz,2:nx])
    return bx,bz,muxz,C44,C55,C66
