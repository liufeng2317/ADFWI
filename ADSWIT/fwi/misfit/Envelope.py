from .base import Misfit
import torch
import numpy as np
import torch.nn.functional as F

def hilbert(x):
    ''' Perform Hilbert transform along the last axis of x.
        Parameters:
        -------------
            x (Tensor) : The signal data. The Hilbert transform is performed along last dimension of `x`.
        Returns:
        -------------
            analytic hilbert (Tensor): A complex tensor with the same shape of `x`, representing its analytic signal. 
    '''
    device = x.device
    N = x.shape[-1]*2-1
    Xf = torch.fft.fft(x,n=N)
    h = torch.zeros((x.shape[0],N),dtype=Xf.dtype).to(device)
    if N % 2 == 0:
        h[...,0] = h[N // 2] = 1
        h[...,1:N // 2] = 2
    else:
        h[...,0] = 1
        h[...,1:(N + 1) // 2] = 2
    return torch.fft.ifft(Xf*h)

def diff(x, dim=-1, same_size=False):
    assert dim == -1, 'diff only supports dim=-1 for now'
    if same_size:
        return F.pad(x[...,1:]-x[...,:-1], (1,0))
    else:
        return x[...,1:]-x[...,:-1]
    
def unwrap(phi,dim=-1):
    assert dim == -1, 'unwrap only supports dim=-1 for now'
    dphi = diff(phi,same_size=True)
    dphi_m = ((dphi+np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m==-np.pi)&(dphi>0)] = np.pi
    phi_adj = dphi_m-dphi
    phi_adj[dphi.abs()<np.pi] = 0
    return phi + phi_adj.cumsum(dim)

class Misfit_envelope(Misfit):
    ''' Envelope difference (Wu et al., 2014; Yuan et al., 2015) 
        can be used to get the initial velocity model
    Paraemters:
    -----------
            dt (float)      : time sampling interval
            p (float)       : the norm order of the reuslt
            instaneous_phase (bool) : use instaeous phase or amplitude for the misfit
            obs (Tensors)   : the observed waveform 
            syn (Tensors)   : the synthetic waveform 
    '''
    def __init__(self,dt=1,p=1.5,instaneous_phase=False,norm="L2") -> None:
        super().__init__() 
        self.p  = p
        self.instaneous_phase = instaneous_phase
        self.dt = dt
        self.norm = norm
        
    def forward(self,obs,syn):
        device      = obs.device
        rsd         = torch.zeros((obs.shape[0],obs.shape[2],obs.shape[1])).to(device)
        for ishot in range(obs.shape[0]):
            obs_shot            = obs[ishot].T # [trace,time series]
            syn_shot            = syn[ishot].T
            analytic_signal_obs = hilbert(obs_shot)[:,:obs_shot.shape[-1]]
            analytic_signal_syn = hilbert(syn_shot)[:,:syn_shot.shape[-1]]
            envelopes_obs       = torch.abs(analytic_signal_obs)
            envelopes_syn       = torch.abs(analytic_signal_syn)
            if self.instaneous_phase:
                instaneous_phase_obs   =  unwrap(torch.angle(analytic_signal_obs))
                instaneous_phase_syn   =  unwrap(torch.angle(analytic_signal_syn))
                rsd[ishot] = instaneous_phase_obs - instaneous_phase_syn
            else:
                # squared 
                rsd[ishot] = envelopes_syn**self.p - envelopes_obs**self.p
                # squared logarithmic ratio
                # rsd[ishot] = torch.log(obs_envelope/syn_envelope)
        if self.norm == "L1":
            loss = torch.sum(torch.abs(rsd))
        else:
            loss = 0.5*torch.sum(rsd*rsd*self.dt)
        return loss