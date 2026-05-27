'''
* Author: LiuFeng(SJTU) : liufeng2317@sjtu.edu.cn
* Date: 2024-04-20 09:32:43
* LastEditors: LiuFeng
* LastEditTime: 2024-05-15 19:30:38
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@sjtu.edu.cn, All Rights Reserved.
'''

import scipy.signal as _signal
import scipy
import numpy as np
import torch
import torch.nn.functional as F

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

def _torch_smooth2d(Z, span=10):
    """Torch equivalent of smooth2d for device-native gradient processing."""
    if span <= 0:
        return Z
    original_dtype = Z.dtype
    work_dtype = Z.dtype if Z.is_floating_point() else torch.float32
    work = Z.to(dtype=work_dtype)
    coords = torch.linspace(-2.0 * span, 2.0 * span, 2 * span + 1, device=Z.device, dtype=work_dtype)
    Y, X = torch.meshgrid(coords, coords, indexing="ij")
    sigma2 = float(span * span)
    kernel = torch.exp(-0.5 * (X * X + Y * Y) / sigma2)
    kernel = kernel / kernel.sum()
    kernel = kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1])
    data = work.reshape(1, 1, work.shape[0], work.shape[1])
    weight = torch.ones_like(data)
    padding = kernel.shape[-1] // 2
    smoothed = F.conv2d(data, kernel, padding=padding)
    normalizer = F.conv2d(weight, kernel, padding=padding)
    return (smoothed / normalizer).reshape_as(work).to(dtype=original_dtype)


def _hamming_window(length):
    """Return the NumPy Hamming window used by the legacy taper path."""
    if hasattr(scipy.signal, "windows") and hasattr(scipy.signal.windows, "hamming"):
        return scipy.signal.windows.hamming(length, sym=True)
    if hasattr(scipy.signal, "hamming"):
        return scipy.signal.hamming(length)
    return np.hamming(length)


def grad_taper_torch(nz, nx, tapersize=20, thred=0.05, marine_or_land='marine', *, device=None, dtype=None):
    """Torch version of grad_taper that keeps masks on the active device."""
    device = torch.device("cpu") if device is None else device
    dtype = torch.float32 if dtype is None else dtype
    if marine_or_land in ['marine', 'Offshore']:
        taper = torch.ones((nz, nx), device=device, dtype=dtype)
        taper[:tapersize, :] = 0.0
    else:
        H = torch.hamming_window(tapersize * 2, periodic=False, device=device, dtype=dtype)
        H = H[tapersize:]
        taper = torch.zeros((nz, nx), device=device, dtype=dtype)
        taper[:, :tapersize] = H.reshape(1, -1)
        taper = _torch_smooth2d(taper, span=tapersize // 2)
        taper = taper / taper.max()
        taper = taper * (1 - thred)
        taper = -taper + 1
        taper = taper * taper
    return taper


def grad_taper(nz, nx, tapersize=20, thred=0.05, marine_or_land='marine'):
    ''' Gradient taper
    '''
    # for masking the water layer, use the zero threds
    if marine_or_land in ['marine', 'Offshore']: 
        taper = np.ones((nz, nx))
        taper[:tapersize,:] = 0.0
            
    # for the land gradient damping, use the small threds
    else:
        H = _hamming_window(tapersize*2)  # gaussian window
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
                 marine_or_land="land",
                 ):
        self.grad_mute      = grad_mute
        self.grad_smooth    = grad_smooth   
        self.grad_mask      = grad_mask
        self.marine_or_land = marine_or_land
        self.norm_grad      = norm_grad
        self.forw_illumination = forw_illumination

    def forward(self,nx,nz,vmax,grad,forw=None):
        # tapper mask
        if self.grad_mute > 0:
            if self.marine_or_land.lower() in ['marine', 'offshore']:
                grad_thred = 0.0
            elif self.marine_or_land.lower() in ['land', 'onshore']:
                grad_thred = 0.001
            else:
                raise ValueError('not supported modeling marine_or_land: %s'%(self.marine_or_land))
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
                grad[self.grad_mute:,:] = smooth2d(grad[self.grad_mute:,:], span=self.grad_smooth)
            # land gradient smooth
            else:
                grad = smooth2d(grad, span=self.grad_smooth)
        
        # scale the gradient properly
        if self.norm_grad:
            grad = vmax * grad/ abs(grad).max()        
        return grad

class TorchGradProcessor(GradProcessor):
    """Device-native gradient processor compatible with GradProcessor settings.

    The legacy GradProcessor remains NumPy/SciPy based. This subclass exposes a
    forward_torch method used by the FWI runtime to avoid CPU round-trips when a
    torch-native path is explicitly requested.
    """

    def forward_torch(self, nx, nz, vmax, grad, forw=None):
        with torch.no_grad():
            processed = grad.clone()
            vmax_tensor = torch.as_tensor(vmax, device=processed.device, dtype=processed.dtype)

            if self.grad_mute > 0:
                if self.marine_or_land.lower() in ['marine', 'offshore']:
                    grad_thred = 0.0
                elif self.marine_or_land.lower() in ['land', 'onshore']:
                    grad_thred = 0.001
                else:
                    raise ValueError('not supported modeling marine_or_land: %s' % (self.marine_or_land))
                processed = processed * grad_taper_torch(
                    nz,
                    nx,
                    tapersize=self.grad_mute,
                    thred=grad_thred,
                    marine_or_land=self.marine_or_land,
                    device=processed.device,
                    dtype=processed.dtype,
                )

            if self.grad_mask is not None:
                grad_mask = torch.as_tensor(self.grad_mask, device=processed.device, dtype=processed.dtype)
                if tuple(grad_mask.shape) != tuple(processed.shape):
                    raise ValueError('Wrong size of grad mask: the size of the mask should be identical to the size of vp model')
                processed = processed * grad_mask

            if self.forw_illumination and forw is not None:
                forw_tensor = torch.as_tensor(forw, device=processed.device, dtype=processed.dtype)
                span = 40 if min(nz, nx) > 40 else int(min(nz, nx) / 2)
                forw_tensor = _torch_smooth2d(forw_tensor, span)
                epsilon = torch.as_tensor(0.0001, device=processed.device, dtype=processed.dtype)
                precond = forw_tensor / torch.max(forw_tensor + 1e-5)
                precond = torch.clamp(precond, min=epsilon)
                processed = processed / torch.pow(precond, 2)

            if self.grad_smooth > 0:
                if self.marine_or_land in ['marine', 'offshore']:
                    smoothed = processed.clone()
                    smoothed[self.grad_mute:, :] = _torch_smooth2d(smoothed[self.grad_mute:, :], span=self.grad_smooth)
                    processed = smoothed
                else:
                    processed = _torch_smooth2d(processed, span=self.grad_smooth)

            if self.norm_grad:
                processed = vmax_tensor * processed / torch.max(torch.abs(processed))
            return processed
