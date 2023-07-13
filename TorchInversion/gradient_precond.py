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

def grad_taper(nx, ny, tapersize=20, thred=0.05, marine_or_land='Marine'):
    ''' Gradient taper
    '''
    # for masking the water layer, use the zero threds
    if marine_or_land in ['Marine', 'Offshore']: 
        taper = np.ones((nx, ny))
        for ix in range(nx):
            taper[ix, :tapersize] = 0.0
            
    # for the land gradient damping, use the small threds
    else:
        H = scipy.signal.hamming(tapersize*2)  # gaussian window
        H = H[tapersize:]
        taper = np.zeros((nx, ny))
        for ix in range(nx):
            taper[ix, :tapersize] = H
        taper = smooth2d(taper, span=tapersize//2)
        taper /= taper.max()
        taper *= (1 - thred)
        taper = - taper + 1
        taper = taper * taper      # taper^2 is better than taper^1

    return taper

def grad_precond(param,grad,forw,grad_mute=0,grad_smooth=0,marine_or_land = 'land'):
    nx = param.nx
    ny = param.ny
    vpmax = param.vmax

    if marine_or_land.lower() in ['marine', 'offshore']:
        grad_thred = 0.0
    elif marine_or_land.lower() in ['land', 'onshore']:
        grad_thred = 0.001
    else:
        raise ValueError('not supported modeling marine_or_land: %s'%(marine_or_land))
    # tapper mask 
    if grad_mute > 0:
        grad *= grad_taper(nx, ny, tapersize = grad_mute, thred = grad_thred, marine_or_land=marine_or_land)

    #apply the inverse Hessian
    if min(nx, ny) > 40:      # set 40 grids in default
        span = 40
    else:                     # in case the grid number is less than 40
        span = int(min(nx, ny)/2)
    
    forw = smooth2d(forw, span)
    epsilon = 0.0001
    forw = forw / np.max(forw)
    precond = forw
    precond = precond / np.max(precond)
    precond[precond < epsilon] = epsilon
    grad = grad / np.power(precond, 1)
    
    # smooth the gradient
    if grad_smooth > 0:
        # exclude water-layer
        if marine_or_land in ['Marine', 'Offshore']: 
            grad[grad_mute:,:] = smooth2d(grad[grad_mute:,:], span=grad_smooth)
        # land gradient smooth
        else:
            grad = smooth2d(grad, span=grad_smooth)

    # gradient with respect to the velocity
    grad = - 2 * grad   #  / np.power(simu.model.vp, 3)

    # grad = np.linspace(0,1,ny).reshape(-1,1)*grad
    
    # scale the gradient properly
    grad *= vpmax / abs(grad).max()
    return grad