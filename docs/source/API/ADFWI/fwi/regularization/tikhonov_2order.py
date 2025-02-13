from .base import Regularization, regular_StepLR
import torch
import numpy as np
from ADFWI.utils import numpy2tensor

class Tikhonov_2order(Regularization):
    """
    2nd-Order Tikhonov Regularization.

    The regularization function is defined as:
    \[
    \| \alpha_x L_2 m_x + \alpha_z L_2 m_z \|_2
    \]
    Where \( L_2 \) is the second-order finite difference operator for the grid, and \( m_x \), \( m_z \) are the model components in the horizontal and vertical directions, respectively.

    References
    ----------
    Du, Z., et al., 2021. A high-order total-variation regularisation method for full-waveform inversion. 
    *Journal of Geophysics and Engineering*, 18, 241–252. doi:10.1093/jge/gxab010

    Parameters
    ----------
    nx : int
        Number of grid points in the x-direction.
    nz : int
        Number of grid points in the z-direction.
    dx : float
        Grid size in the x-direction (m).
    dz : float
        Grid size in the z-direction (m).
    alphax : float, optional
        Regularization factor for the x-direction. Default is 0.
    alphaz : float, optional
        Regularization factor for the z-direction. Default is 0.
    step_size : int, optional
        Step size for updating the regularization parameters. Default is 1000.
    gamma : float, optional
        Decay factor for regularization parameter updates. Default is 1.
    device : str, optional
        Device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
    dtype : torch.dtype, optional
        The data type of the tensors. Default is `torch.float32`.
    """
    
    def __init__(self, nx, nz, dx, dz, alphax=0, alphaz=0, step_size=1000, gamma=1, device='cpu', dtype=torch.float32) -> None:
        super().__init__(nx, nz, dx, dz, alphax, alphaz, step_size, gamma, device, dtype)
        
        # Vertical constraint: Second-order finite difference operator for z-direction
        L0 = np.diag(1 * np.ones(nz - 1), 1) + np.diag(-2 * np.ones(nz)) + np.diag(1 * np.ones(nz - 1), -1)
        L0[0, :] = 0  # Zero out the first row to handle boundary condition
        L0[-1, :] = 0  # Zero out the last row to handle boundary condition
        self.L0 = numpy2tensor(L0, dtype=dtype).to(device)
        
        # Horizontal constraint: Second-order finite difference operator for x-direction
        L1 = np.diag(1 * np.ones(nx - 1), 1) + np.diag(1 * np.ones(nx - 1), -1) + np.diag(-2 * np.ones(nx))
        L1[0, :] = 0  # Zero out the first row to handle boundary condition
        L1[-1, :] = 0  # Zero out the last row to handle boundary condition
        self.L1 = numpy2tensor(L1, dtype=dtype).to(device)

    def forward(self, m):
        """
        Compute the regularized misfit for the given model.

        Parameters
        ----------
        m : torch.Tensor
            The model tensor to apply regularization on.

        Returns
        -------
        torch.Tensor
            The regularized misfit value.
        """
        dz, dx = self.dz / 1000, self.dx / 1000  # Convert to km

        # Apply vertical constraint (L0) and normalize by dz
        m_norm_z = torch.matmul(self.L0, m) / dz

        # Apply horizontal constraint (L1) and normalize by dx
        m_norm_x = torch.matmul(self.L1, m.T).T / dx
        
        # Update regularization parameters (alphax and alphaz)
        alphax = regular_StepLR(self.iter, self.step_size, self.alphax, self.gamma)
        alphaz = regular_StepLR(self.iter, self.step_size, self.alphaz, self.gamma)
        
        # Compute misfit (L2 norm of regularized model components)
        misfit_norm = torch.sqrt(torch.sum(alphax * m_norm_x * m_norm_x + alphaz * m_norm_z * m_norm_z))
        
        # Increment iteration counter
        self.iter += 1
        return misfit_norm