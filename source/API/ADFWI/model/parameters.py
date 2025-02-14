import torch
from ADFWI.utils import numpy2tensor

def thomsen_init(vp, vs, rho, eps, delta, gamma, device, dtype=torch.float32):
    """Initialize the Thomsen parameters.

    Parameters
    ----------
    vp : array_like
        P-wave velocity model.
    vs : array_like
        S-wave velocity model.
    rho : array_like
        Density model.
    eps : array_like
        Anisotropic parameter epsilon.
    delta : array_like
        Anisotropic parameter delta.
    gamma : array_like
        Anisotropic parameter gamma.
    device : str
        The device to place the tensors on ('cpu' or 'cuda').
    dtype : torch.dtype, optional
        Data type of the tensors, default is torch.float32.

    Returns
    -------
    vp : torch.Tensor
        P-wave velocity as a tensor.
    vs : torch.Tensor
        S-wave velocity as a tensor.
    rho : torch.Tensor
        Density as a tensor.
    eps : torch.Tensor
        Anisotropic epsilon parameter as a tensor.
    delta : torch.Tensor
        Anisotropic delta parameter as a tensor.
    gamma : torch.Tensor
        Anisotropic gamma parameter as a tensor.
    """
    vp      = numpy2tensor(vp, dtype).to(device)
    vs      = numpy2tensor(vs, dtype).to(device)
    rho     = numpy2tensor(rho, dtype).to(device)
    eps     = numpy2tensor(eps, dtype).to(device)
    delta   = numpy2tensor(delta, dtype).to(device)
    gamma   = numpy2tensor(gamma, dtype).to(device)
    return vp, vs, rho, eps, delta, gamma


def elastic_moduli_init(nz, nx, device, dtype=torch.float32):
    """Initialize the 21 independent elastic parameters for a full anisotropic medium. The function initializes the following 21 independent elastic parameters for a full anisotropic medium.

    Parameters
    ----------
    nz : int
        The number of grid points in the z-direction.
    nx : int
        The number of grid points in the x-direction.
    device : str
        The device to place the tensors on ('cpu' or 'cuda').
    dtype : torch.dtype, optional
        The data type of the tensors. Default is torch.float32.

    Returns
    -------
    list of torch.Tensor
        A list containing the 21 independent elastic parameters, each as a tensor. The parameters are in the following order: [C11, C12, C13, C14, C15, C16, C22, C23, C24, C25, C26, C33, C34, C35, C36, C44, C45, C46, C55, C56, C66].
    """
    # The 21 independent elastic parameters for full anisotropic models
    C11,C12,C13,C14,C15,C16 = torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device)
    C22,C23,C24,C25,C26     = torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device)
    C33,C34,C35,C36         = torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device)
    C44,C45,C46             = torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device)
    C55,C56                 = torch.zeros((nz,nx),dtype=dtype).to(device), torch.zeros((nz,nx),dtype=dtype).to(device)
    C66                     = torch.zeros((nz,nx),dtype=dtype).to(device)
    
    CC = [C11, C12, C13, C14, C15, C16, C22, C23, C24, C25, C26, 
          C33, C34, C35, C36, C44, C45, C46, C55, C56, C66]
    
    return CC


def vs_vp_to_Lame(vp, vs, rho):
    """
    Transform P-wave and S-wave velocities along with density to the Lame constants.

    Parameters
    ----------
    vp : torch.Tensor
        P-wave velocity along the respective axis.
    vs : torch.Tensor
        S-wave velocity along the respective axis.
    rho : torch.Tensor
        Density.

    Returns
    -------
    mu : torch.Tensor
        Lame constant (shear modulus).
    lamu : torch.Tensor
        Lame constant (lambda).
    lam : torch.Tensor
        Lambda constant (lambda - 2 * mu).
    b : torch.Tensor
        Buoyancy, which is the inverse of density (1 / rho).
    """
    mu   = vs**2 * rho
    lamu = vp**2 * rho
    lam  = lamu - 2 * mu
    b    = 1 / rho
    return mu, lamu, lam, b


def thomsen_to_elastic_moduli(alpha_thomsen, beta_thomsen, rho, eps_thomsen, delta_thomsen, gamma_thomsen):
    """
    Transform Thomsen parameters to Elastic Moduli for Transverse Isotropy. For Transverse isotropy case, only 5 independent components are used: C11, C13, C33, C44, C66. Additionally, C12 is calculated as C11 - 2*C66.

    Parameters
    ----------
    alpha_thomsen : torch.Tensor
        P-wave velocity (vp).
    beta_thomsen : torch.Tensor
        S-wave velocity (vs).
    rho : torch.Tensor
        Density.
    eps_thomsen : torch.Tensor
        Anisotropy parameter (eps).
    delta_thomsen : torch.Tensor
        Anisotropy parameter (delta).
    gamma_thomsen : torch.Tensor
        Anisotropy parameter (gamma).

    Returns
    -------
    C11 : torch.Tensor
        Elastic Moduli, C11.
    C13 : torch.Tensor
        Elastic Moduli, C13.
    C33 : torch.Tensor
        Elastic Moduli, C33.
    C44 : torch.Tensor
        Elastic Moduli, C44.
    C66 : torch.Tensor
        Elastic Moduli, C66.
    """
    C33 = alpha_thomsen**2 * rho
    C44 = beta_thomsen**2 * rho
    C11 = C33 * (1 + 2 * eps_thomsen)
    C66 = C44 * (1 + 2 * gamma_thomsen)
    C13 = torch.sqrt(2 * C33 * (C33 - C44) * delta_thomsen + (C33 - C44)**2) - C44
    return C11, C13, C33, C44, C66

def elastic_moduli_to_thomsen(C11, C13, C33, C44, C66, rho):
    """
    Transform Elastic Moduli to Thomsen parameters.

    Parameters
    ----------
    C11 : torch.Tensor
        Elastic Moduli, C11.
    C13 : torch.Tensor
        Elastic Moduli, C13.
    C33 : torch.Tensor
        Elastic Moduli, C33.
    C44 : torch.Tensor
        Elastic Moduli, C44.
    C66 : torch.Tensor
        Elastic Moduli, C66.
    rho : torch.Tensor
        Density.

    Returns
    -------
    alpha_thomsen : torch.Tensor
        P-wave velocity (vp).
    beta_thomsen : torch.Tensor
        S-wave velocity (vs).
    eps_thomsen : torch.Tensor
        Anisotropy parameter (eps).
    delta_thomsen : torch.Tensor
        Anisotropy parameter (delta).
    gamma_thomsen : torch.Tensor
        Anisotropy parameter (gamma).
    """
    alpha_thomsen = torch.sqrt(C33 / rho)
    beta_thomsen = torch.sqrt(C44 / rho)
    eps_thomsen = (C11 - C33) / (2 * C33)
    gamma_thomsen = (C66 - C44) / (2 * C44)
    delta_thomsen = ((C13 + C44)**2 - (C33 - C44)**2) / (2 * C33 * (C33 - C44))
    return alpha_thomsen, beta_thomsen, eps_thomsen, delta_thomsen, gamma_thomsen

def elastic_moduli_for_isotropic(CC):
    """
    For isotropy case, only 2 independent components. The matrix of elastic moduli is simplified as:

    Parameters
    ----------
    CC : list of torch.Tensor
        List containing the 21 elastic moduli parameters: [C11, C12, C13, ..., C66].

    Returns
    -------
    CC : list of torch.Tensor
        Modified list of elastic moduli with the isotropic assumption applied.
    """
    [C11, C12, C13, C14, C15, C16, C22, C23, C24, C25, C26, C33, C34, C35, C36, C44, C45, C46, C55, C56, C66] = CC
    C22 = C11.clone()
    C55 = C44.clone()
    C23 = C13.clone()
    C12 = (C11 - 2 * C66).clone()
    CC = [C11, C12, C13, C14, C15, C16, C22, C23, C24, C25, C26, C33, C34, C35, C36, C44, C45, C46, C55, C56, C66]
    return CC

def elastic_moduli_for_TI(CC, anisotropic_type="VTI"):
    """
    For Transverse isotropy case, only 5 independent components: C11, C13, C33, C44, C66, and C12 = C11 - 2*C66.

    Parameters
    ----------
    CC : list of torch.Tensor
        List containing the 21 elastic moduli parameters: [C11, C12, C13, ..., C66].

    anisotropic_type : str, optional
        Type of anisotropy. Can be 'VTI' for vertical transverse isotropy (default) or 'HTI' for horizontal transverse isotropy.

    Returns
    -------
    CC : list of torch.Tensor
        Modified list of elastic moduli with the corresponding anisotropic structure applied, depending on `anisotropic_type`.
    """
    [C11, C12, C13, C14, C15, C16, C22, C23, C24, C25, C26, C33, C34, C35, C36, C44, C45, C46, C55, C56, C66] = CC
    C22 = C11.clone()
    C55 = C44.clone()
    C23 = C13.clone()
    C12 = (C11 - 2 * C66).clone()
    
    if anisotropic_type.lower() in ["vti"]:
        CC = [C11, C12, C13, C14, C15, C16,
              C22, C23, C24, C25, C26,
              C33, C34, C35, C36,
              C44, C45, C46,
              C55, C56,
              C66]
    elif anisotropic_type.lower() in ["hti"]:
        CC = [C33, C13, C13, C14, C15, C16,
              C11, C12, C24, C25, C26,
              C11, C34, C35, C36,
              C66, C45, C46,
              C55, C56,
              C55]
    
    return CC


def parameter_staggered_grid(mu, b, C44, C55, C66, nx, nz):
    """
    Staggered mesh settings for model parameters.

    This function calculates the staggered grid values for different model parameters:
    shear modulus (mu), buoyancy (b), and elastic moduli (C44, C55, C66) based on 
    the grid size (nx, nz) and applies an averaging method for staggered grids.

    Parameters
    ----------
    mu : torch.Tensor
        Shear modulus, also known as Lame constant.
    
    b : torch.Tensor
        Buoyancy, the inverse of density.
    
    C44 : torch.Tensor
        Elastic modulus C44, describing shear waves.
    
    C55 : torch.Tensor
        Elastic modulus C55, another shear modulus term.
    
    C66 : torch.Tensor
        Elastic modulus C66, also related to shear waves.
    
    nx : int
        Number of grid points in the X direction (horizontal axis).
    
    nz : int
        Number of grid points in the Z direction (vertical axis).

    Returns
    -------
    bx : torch.Tensor
        Staggered grid for buoyancy (b) in the X-axis.
    
    bz : torch.Tensor
        Staggered grid for buoyancy (b) in the Z-axis.
    
    muxz : torch.Tensor
        Staggered grid for shear modulus (mu).
    
    C44 : torch.Tensor
        Staggered grid for C44 (shear modulus term).
    
    C55 : torch.Tensor
        Staggered grid for C55 (shear modulus term).
    
    C66 : torch.Tensor
        Staggered grid for C66 (shear modulus term).
    """
    bx   = 0.5 * (b[:, 0:nx-1] + b[:, 1:nx])
    bz   = 0.5 * (b[0:nz-1, :] + b[1:nz, :])

    muxz = 0.2 * (mu[1:nz-1, 1:nx-1] + mu[2:nz, 1:nx-1] + mu[1:nz-1, 2:nx] +
                  mu[2:nz, 1:nx-1] + mu[2:nz, 2:nx])

    C44  = 0.2 * (C44[1:nz-1, 1:nx-1] + C44[2:nz, 1:nx-1] + C44[1:nz-1, 2:nx] +
                  C44[2:nz, 1:nx-1] + C44[2:nz, 2:nx])

    C55  = 0.2 * (C55[1:nz-1, 1:nx-1] + C55[2:nz, 1:nx-1] + C55[1:nz-1, 2:nx] +
                  C55[2:nz, 1:nx-1] + C55[2:nz, 2:nx])

    C66  = 0.2 * (C66[1:nz-1, 1:nx-1] + C66[2:nz, 1:nx-1] + C66[1:nz-1, 2:nx] +
                  C66[2:nz, 1:nx-1] + C66[2:nz, 2:nx])

    return bx, bz, muxz, C44, C55, C66
