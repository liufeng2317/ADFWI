from .base import Regularization, regular_StepLR
import torch
import numpy as np
from ADFWI.utils import numpy2tensor

class TV_1order(Regularization):
    """
    1st-Order Total Variation Regularization.

    The regularization function is defined as:
    \[
    \alpha_x \| L_1 m_x \|_p + \alpha_z \| L_1 m_z \|_p
    \]
    Where \( L_1 \) is the first-order finite difference operator for the grid, and \( m_x \), \( m_z \) are the model components in the horizontal and vertical directions, respectively.

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
        
        # Vertical constraint: First-order finite difference operator for z-direction
        L0 = np.diag(1 * np.ones(nz), 0) + np.diag(-1 * np.ones(nz - 1), 1)
        L0[-1, :] = 0  # Zero out the last row to handle boundary condition
        self.L0 = numpy2tensor(L0, dtype=torch.float32).to(device)
        
        # Horizontal constraint: First-order finite difference operator for x-direction
        L1 = np.diag(1 * np.ones(nx), 0) + np.diag(-1 * np.ones(nx - 1), 1)
        L1[-1, :] = 0  # Zero out the last row to handle boundary condition
        self.L1 = numpy2tensor(L1, dtype=torch.float32).to(device)

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
        
        # Compute misfit (L1 norm of regularized model components)
        misfit_norm = torch.sum(alphax * torch.abs(m_norm_x) + alphaz * torch.abs(m_norm_z))
        
        # Increment iteration counter
        self.iter += 1
        return misfit_norm