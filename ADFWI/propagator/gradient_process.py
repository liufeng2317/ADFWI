import scipy.signal as _signal
import scipy
import numpy as np

def gauss2(X, Y, mu, sigma, normalize=True):
    ''' Evaluates Gaussian over points of X,Y
    '''
    D = sigma[0, 0]*sigma[1, 1] - sigma[0, 1]*sigma[1, 0]
    B = np.linalg.inv(sigma)
    X = X - mu[0]
    Y = Y - mu[1]
    Z = B[0, 0]*X**2. + B[0, 1]*X*Y + B[1, 0]*X*Y + B[1, 1]*Y**2.
    Z = np.exp(-0.5*Z)

    if normalize:
        Z *= (2.*np.pi*np.sqrt(D))**(-1.)
    return Z


def smooth2d(Z, span=10):
    ''' Smooths values on 2D rectangular grid
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

def grad_taper(nz, nx, tapersize=20, thred=0.05, marine_or_land='Marine'):
    ''' Gradient taper
    '''
    # for masking the water layer, use the zero threds
    if marine_or_land in ['Marine', 'Offshore']: 
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
    def __init__(self,grad_mute=0,
                 grad_smooth=0,
                 grad_mask=None,
                 norm_grad=True,
                 forw_illumination=True,
                 marine_or_land="land"):
        self.grad_mute      = grad_mute
        self.grad_smooth    = grad_smooth   
        self.grad_mask      = grad_mask
        self.marine_or_land = marine_or_land
        self.norm_grad      = norm_grad
        self.forw_illumination = forw_illumination

    def forward(self,nx,nz,vmax,grad,forw=None):
        # tapper mask
        if self.marine_or_land.lower() in ['marine', 'offshore']:
            grad_thred = 0.0
        elif self.marine_or_land.lower() in ['land', 'onshore']:
            grad_thred = 0.001
        else:
            raise ValueError('not supported modeling marine_or_land: %s'%(self.marine_or_land))
        
        if self.grad_mute > 0:
            grad *= grad_taper(nz, nx, tapersize = self.grad_mute, thred = grad_thred, marine_or_land=self.marine_or_land)
        
        # grad mask
        if np.any(self.grad_mask == None):
            pass
        else:
            if np.shape(self.grad_mask) != np.shape(grad):
                raise('Wrong size of grad mask: the size of the mask should be identical to the size of vp model')
            else:
                grad *= self.grad_mask
        
        # apply the inverse Hessian
        if min(nz, nx) > 40:      # set 40 grids in default
            span = 40
        else:                     # in case the grid number is less than 40
            span = int(min(nz, nx)/2)
        
        if self.forw_illumination and forw is not None:
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
                grad[self.grad_mute:,:] = smooth2d(grad[self.grad_mute:,:], span=self.grad_smooth)
            # land gradient smooth
            else:
                grad = smooth2d(grad, span=self.grad_smooth)
        
        # scale the gradient properly
        if self.norm_grad:
            grad = vmax * grad/ abs(grad).max()        
        return grad