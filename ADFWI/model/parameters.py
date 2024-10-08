import torch
from ADFWI.utils import numpy2tensor

def thomsen_init(vp,vs,rho,eps,delta,gamma,device,dtype=torch.float32):
    """

    """
    vp      = numpy2tensor(vp,dtype).to(device)
    vs      = numpy2tensor(vs,dtype).to(device)
    rho     = numpy2tensor(rho,dtype).to(device)
    eps     = numpy2tensor(eps,dtype).to(device)
    delta   = numpy2tensor(delta,dtype).to(device)
    gamma   = numpy2tensor(gamma,dtype).to(device)
    return vp,vs,rho,eps,delta,gamma

def elastic_moduli_init(nz,nx,device,dtype=torch.float32):
    """ full anisotropic medium with 21 independent parameters
    Description:
    ------------
        *************************************************
        *   C11     C12     C13     C14     C15     C16 *
        *           C22     C23     C24     C25     C26 * 
        *                   C33     C34     C35     C36 * 
        *                           C44     C45     C46 * 
        *                                   C55     C66 * 
        *                                           C66 *
        *************************************************
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
    """Transform Thomsen parameter to Lame constant
    Parameters
    ----------
        vp : P-wave velocity along the respective axis
        vs : S-wave velocity along the respective axis
        rho: density
        nx : grids number along X-axis
        nz : grids number along Z-axis

    Returns
    -------
        mu  : Lame constant:shear modulus $\mu$
        lamu: Lame constant $\lambda$
        lamu: lambda - 2*mu
        b   : buoyancy = 1/density
    """
    mu   = vs**2*rho
    lamu = vp**2*rho
    lam  = lamu-2*mu
    b    = 1/rho
    return mu,lamu,lam,b


def thomsen_to_elastic_moduli(alpha_thomsen,beta_thomsen,rho,eps_thomsen,delta_thomsen,gamma_thomsen):
    """ Transform Thomsen parameter to Elastic Moduli
    Description:
    ----------
    For Transverse isotropy case, only 5 independent components: C11,C13,C33,C44,C66 ,and C12=C11-2*C66
        *********************************************************
        *   C11         C11-2C66    C13     0       0       0   *
        *   C11-2C66    C11         C13     0       0       0   *
        *   C13         C13         C33     0       0       0   *
        *   0           0           0       C44     0       0   *
        *   0           0           0       0       C44     0   *
        *   0           0           0       0       0       C66 *
        *********************************************************

    Parameters
    ----------
        alpha_thomsen: vp
        beta_thomsen : vs
        rho          : density
        eps_thomsen  : anisotropy parameter
        delta_thomsen: anisotropy parameter
        gamma_thomsen: anisotropy parameter

    Returns
    -------
        C11 : Elastic Moduli
        C13 : Elastic Moduli
        C33 : Elastic Moduli
        C44 : Elastic Moduli
        C66 : Elastic Moduli
    """
    C33 = alpha_thomsen**2*rho
    C44 =  beta_thomsen**2*rho
    C11 = C33*(1+2*eps_thomsen)
    C66 = C44*(1+2*gamma_thomsen)
    C13 = torch.sqrt(2*C33*(C33-C44)*delta_thomsen + (C33-C44)**2) - C44
    return C11,C13,C33,C44,C66

def elastic_moduli_to_thomsen(C11,C13,C33,C44,C66,rho):
    """ Transform Elastic Moduli to Thomsen parameter
    Parameters
    ----------
        C11 : Elastic Moduli
        C13 : Elastic Moduli
        C33 : Elastic Moduli
        C44 : Elastic Moduli
        C66 : Elastic Moduli
        rho : density
    
    Returns
    -------
        alpha_thomsen: vp
        beta_thomsen : vs
        eps_thomsen  : anisotropy parameter
        delta_thomsen: anisotropy parameter
        gamma_thomsen: anisotropy parameter
    """
    alpha_thomsen   = torch.sqrt(C33/rho)
    beta_thomsen    = torch.sqrt(C44/rho)
    eps_thomsen     = (C11 - C33)/(2*C33)
    gamma_thomsen   = (C66 - C44)/(2*C44)
    delta_thomsen   = ((C13+C44)**2 - (C33-C44)**2)/(2*(C33)*(C33 - C44))
    return alpha_thomsen,beta_thomsen,eps_thomsen,delta_thomsen,gamma_thomsen

def elastic_moduli_for_isotropic(CC):
    """ For  isotropy case, only 2 independent components
    Description
    ------------
        *****************************************************************
        *   lambda+2mu  lambda      lambda      0       0       0       *
        *   C11-2C66    lambda+2mu  lambda      0       0       0       *
        *   lambda      lambda      lambda+2mu  0       0       0       *
        *   0           0           0           mu      0       0       *
        *   0           0           0           0       mu      0       *
        *   0           0           0           0       0       mu      *
        *****************************************************************
    """
    [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66] = CC
    C22 = C11.clone()
    C55 = C44.clone()
    C23 = C13.clone()
    C12 = (C11 - 2*C66).clone()
    CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66]
    return CC

def elastic_moduli_for_TI(CC,anisotropic_type="VTI"):
    """ For Transverse isotropy case, only 5 independent components: C11,C13,C33,C44,C66 ,and C12=C11-2*C66
    Description
    -------------
        *********************************************************
        *   C11         C11-2C66    C13     0       0       0   *
        *   C11-2C66    C11         C13     0       0       0   *
        *   C13         C13         C33     0       0       0   *
        *   0           0           0       C44     0       0   *
        *   0           0           0       0       C44     0   *
        *   0           0           0       0       0       C66 *
        *********************************************************
    """
    [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66] = CC
    C22 = C11.clone()
    C55 = C44.clone()
    C23 = C13.clone()
    C12 = (C11 - 2*C66).clone()
    # HTI Rotated by VTI anticlockwise (Y) pi/2
    if anisotropic_type.lower() in ["vti"]:
        CC = [C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66]
        # [C11,C12,C13,C15,C22,C23,C33,C35,C44,C55,C66]
    elif anisotropic_type.lower() in ["hti"]:
        CC = [C33,C13,C13,C14,C15,C16,C11,C12,C24,C25,C26,C11,C34,C35,C36,C66,C45,C46,C55,C56,C55]
        # [C33,C13,C13,C15,C11,C12,C11,C35,C66,C55,C55]
    return CC


def parameter_staggered_grid(mu,b,C44,C55,C66,nx,nz):
    """ Staggered mesh settings for model parameters
    Parameters
    ----------
        mu  : Lame constant:shear modulus $\mu$
        b   : buoyancy = 1/density 
    Returns
    -------
        bx  : staggered grid for b  in X-axis
        bz  : staggered grid for b  in Z-axis
        muxz: staggered grid for mu
        C44 : staggered grid for C44
        C55 : staggered grid for C55
        C66 : staggered grid for C66
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
