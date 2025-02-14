import scipy.signal as _signal
import scipy
import numpy as np

def gauss2(X, Y, mu, sigma, normalize=True):
    ''' 
    Evaluates the 2D Gaussian function over the points X and Y.

    Parameters
    -----------
    X : ndarray
        A 1D array of x coordinates where the Gaussian function will be evaluated.
    Y : ndarray
        A 1D array of y coordinates where the Gaussian function will be evaluated.
    mu : ndarray
        A 1D array of length 2, representing the mean (mu_x, mu_y) of the Gaussian distribution.
    sigma : ndarray
        A 2x2 array representing the covariance matrix of the Gaussian distribution.
    normalize : bool, optional, default=True
        Whether to normalize the Gaussian function (i.e., to ensure the integral over all space equals 1).

    Returns
    --------
    Z : ndarray
        The evaluated Gaussian function at the points (X, Y).
    '''
    # Determinant of the covariance matrix
    D = sigma[0, 0]*sigma[1, 1] - sigma[0, 1]*sigma[1, 0]
    
    # Inverse of the covariance matrix
    B = np.linalg.inv(sigma)
    
    # Shift X and Y by the mean
    X = X - mu[0]
    Y = Y - mu[1]
    
    # Compute the quadratic form
    Z = B[0, 0]*X**2. + B[0, 1]*X*Y + B[1, 0]*X*Y + B[1, 1]*Y**2.
    
    # Apply the Gaussian exponent
    Z = np.exp(-0.5*Z)

    # Normalize if required
    if normalize:
        Z *= (2.*np.pi*np.sqrt(D))**(-1.)

    return Z


def smooth2d(Z, span=10):
    ''' 
    Smooths values on a 2D rectangular grid using a Gaussian filter.

    Parameters
    ----------
    Z : ndarray
        A 2D array representing the values to be smoothed (input grid).
    span : int, optional
        The span or size of the Gaussian filter, which determines the extent of smoothing. Default is 10.

    Returns
    -------
    Z : ndarray
        A 2D array with the smoothed values.
    '''
    import warnings
    warnings.filterwarnings('ignore')

    Z = np.copy(Z)

    x = np.linspace(-2.*span, 2.*span, 2*span + 1)
    y = np.linspace(-2.*span, 2.*span, 2*span + 1)
    (X, Y) = np.meshgrid(x, y)
    mu = np.array([0., 0.])
    sigma = np.diag([span, span])**2.
    F = gauss2(X, Y, mu, sigma)
    F = F/np.sum(F)
    W = np.ones(Z.shape)
    Z = _signal.convolve2d(Z, F, 'same')
    W = _signal.convolve2d(W, F, 'same')
    Z = Z/W

    return Z


def grad_taper(nz, nx, tapersize=20, thred=0.05, marine_or_land='marine'):
    ''' 
    Creates a tapering function for gradient damping, typically used for applying boundary conditions or damping regions in wave propagation models.
    
    Parameters
    ----------
    nz : int
        The number of grid points in the vertical (z) direction.
    nx : int
        The number of grid points in the horizontal (x) direction.
    tapersize : int, optional
        The size of the tapering region. Default is 20.
    thred : float, optional
        The threshold value for the tapering effect. Default is 0.05.
    marine_or_land : str, optional
        Specifies the type of environment. 'marine' or 'Offshore' for water (marine) 
        or other values for land environments. Default is 'marine'.

    Returns
    -------
    taper : ndarray
        A 2D array representing the tapering function applied to the grid. 
        It has values between 0 and 1, with the boundary regions having lower values.
    '''
    # for masking the water layer, use the zero threds
    if marine_or_land in ['marine', 'Offshore']: 
        taper = np.ones((nz, nx))
        taper[:tapersize,:] = 0.0
            
    # for the land gradient damping, use the small threds
    else:
        H = scipy.signal.hamming(tapersize*2)  # gaussian window
        H = H[tapersize:]
        taper = np.zeros((nz, nx))
        for ix in range(nz):
            taper[ix, :tapersize] = H
        taper = smooth2d(taper, span=tapersize//2)
        taper /= taper.max()
        taper *= (1 - thred)
        taper = - taper + 1
        taper = taper * taper      # taper^2 is better than taper^1
    return taper


class GradProcessor():
    ''' 
    Class for processing gradients, applying various gradient manipulation techniques such as masking, tapering, smoothing, and normalization.
    
    Parameters
    ----------
    grad_mute : int, optional
        Size of the gradient mute zone (default is 0). 
    grad_smooth : int, optional
        Size of the smoothing window (default is 0). 
    grad_mask : ndarray, optional
        A mask to apply to the gradient (default is None). 
    norm_grad : bool, optional
        Whether to normalize the gradient (default is True). 
    forw_illumination : bool, optional
        Whether to apply the forward illumination (default is True).
    marine_or_land : str, optional
        Defines the environment ('marine' or 'land', default is 'land').
    '''
    def __init__(self, grad_mute=0,
                 grad_smooth=0,
                 grad_mask=None,
                 norm_grad=True,
                 forw_illumination=True,
                 marine_or_land="land"):
        '''
        Initializes the GradProcessor object with the specified parameters.
        '''
        self.grad_mute = grad_mute
        self.grad_smooth = grad_smooth   
        self.grad_mask = grad_mask
        self.marine_or_land = marine_or_land
        self.norm_grad = norm_grad
        self.forw_illumination = forw_illumination

    def forward(self, nx, nz, vmax, grad, forw=None):
        ''' 
        Processes the gradient by applying tapering, masking, smoothing, and normalization.

        Parameters
        ----------
        nx : int
            The number of grid points in the horizontal (x) direction.
        nz : int
            The number of grid points in the vertical (z) direction.
        vmax : float
            The maximum value of the gradient, used for normalization.
        grad : ndarray
            The gradient to be processed.
        forw : ndarray, optional
            The forward model, used for applying forward illumination (default is None).

        Returns
        -------
        grad : ndarray
            The processed gradient after applying all operations (mute, mask, smoothing, etc.).
        '''
        # tapper mask
        if self.grad_mute > 0:
            if self.marine_or_land.lower() in ['marine', 'offshore']:
                grad_thred = 0.0
            elif self.marine_or_land.lower() in ['land', 'onshore']:
                grad_thred = 0.001
            else:
                raise ValueError(f'Not supported modeling marine_or_land: {self.marine_or_land}')
            grad *= grad_taper(nz, nx, tapersize=self.grad_mute, thred=grad_thred, marine_or_land=self.marine_or_land)
        
        # grad mask
        if np.any(self.grad_mask == None):
            pass
        else:
            if np.shape(self.grad_mask) != np.shape(grad):
                raise ValueError(f'Wrong size of grad mask: the size of the mask should be identical to the size of vp model')
            else:
                grad *= self.grad_mask
        
        # apply the inverse Hessian
        if self.forw_illumination and forw is not None:
            if min(nz, nx) > 40:      # set 40 grids in default
                span = 40
            else:                     # in case the grid number is less than 40
                span = int(min(nz, nx)/2)
            
            forw = smooth2d(forw, span)
            epsilon = 0.0001
            precond = forw
            precond = precond / np.max(precond+1e-5)
            precond[precond < epsilon] = epsilon
            grad = grad / np.power(precond, 2)
        
        # smooth the gradient
        if self.grad_smooth > 0:
            # exclude water-layer
            if self.marine_or_land in ['marine', 'offshore']: 
                grad[self.grad_mute:, :] = smooth2d(grad[self.grad_mute:, :], span=self.grad_smooth)
            # land gradient smooth
            else:
                grad = smooth2d(grad, span=self.grad_smooth)
        
        # scale the gradient properly
        if self.norm_grad:
            grad = vmax * grad / abs(grad).max()        
        
        return grad
