'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-06-27 19:15:39
* LastEditors: LiuFeng
* LastEditTime: 2023-07-11 23:15:25
* FilePath: /Acoustic_AD/ADinversion/utils.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import numpy as np
from math import log
from obspy.core import UTCDateTime
import obspy 
import copy
import torch

def source_wavelet(nt, dt, f0, srctype):
    ''' source time function
    '''
    # time and wavelet arrays
    t       = np.linspace(0, dt*nt, num=nt, endpoint=False)
    wavelet = np.zeros_like(t,dtype='float32')

    if srctype.lower() in ['ricker']:
        t0 = 1.2/f0
        temp = (np.pi*f0) ** 2
        wavelet = (1 - 2 * temp * (t - t0) ** 2) * np.exp(- temp * (t - t0) ** 2)
    else:
        raise ValueError('Other wavelets can be implemented here.')
    return wavelet


# def pad(c,pml,nx,nz):
#     nx_pml = nx+2*pml
#     nz_pml = nz+2*pml
#     cc = np.zeros((nx_pml,nz_pml))
#     cc[pml:nx_pml-pml,pml:nz_pml-pml] = c
#     for ix in range(0,pml):
#         cc[ix,pml:pml+nz] = cc[pml,pml:pml+nz]
#         cc[nx_pml-pml+ix,pml:pml+nz] = cc[nx_pml-pml-1,pml:pml+nz]
#     for iz in range(0,pml):
#         cc[:,iz] = cc[:,pml]
#         cc[:,nz_pml-pml+iz] = cc[:,nz_pml-pml-1]
#     return cc

def pad(c,pml,nx,nz):
    nx_pml = nx+2*pml
    nz_pml = nz+2*pml
    cc = np.zeros((nx_pml,nz_pml))
    cc[pml:nx_pml-pml,pml:nz_pml-pml] = c
    
    cc[range(0,pml),pml:pml+nz] = np.ones_like(cc[range(0,pml),pml:pml+nz])*cc[pml,pml:pml+nz]
    cc[range(nx_pml-pml,nx_pml),pml:pml+nz] = np.ones_like(cc[range(nx_pml-pml,nx_pml),pml:pml+nz])*cc[nx_pml-pml-1,pml:pml+nz]

    cc[:,range(0,pml)] = cc[:,[pml]]
    cc[:,range(nz_pml-pml,nz_pml)] = cc[:,[nz_pml-pml-1]]
    return cc

def set_damp(vmax,nx_pml,nz_pml,pml,dx):
    """
        计算PML层的吸收系数
    """
    damp_global = np.zeros((nx_pml,nz_pml))
    damp = np.zeros(pml)

    a = (pml-1)*dx
    kappa = 3.0*vmax*log(1000.0)/(2.0*a)  # Adjust the damping effect.

    for ix in range(0,pml):
        xa = ix*dx/a
        damp[ix] = kappa*xa*xa
        
    for ix in range(0,pml):
        for iz in range(0,nz_pml):
            damp_global[pml-ix-1,iz] = damp[ix]
            damp_global[nx_pml+ix-pml,iz] = damp[ix]

    for iz in range(0,pml):
        for ix in range((pml-(iz-1))-1,nx_pml-(pml-(iz))):
            damp_global[ix,pml-iz-1] = damp[iz]
            damp_global[ix,nz_pml+iz-pml] = damp[iz]
            
    return damp_global

def add_su_header(trace, nt, dt, isrc, channel):
    ''' add su header
    '''
    # get parameters

    irec = 0
    for tr in trace:
        # add headers
        tr.stats.network = 'FWI'
        tr.stats.station = '%d'%irec
        tr.stats.location = 'Source-%d' % isrc
        tr.stats.channel = channel
        tr.stats.starttime = UTCDateTime("2021-01-01T00:00:00")
        tr.stats.sampling_rate = 1./dt
        tr.stats.distance = tr.stats.su.trace_header.group_coordinate_x - tr.stats.su.trace_header.source_coordinate_x
        t0 = tr.stats.starttime 
        tr.trim(starttime=t0, endtime=t0 + dt*(nt-1), pad=True, nearest_sample=True, fill_value=0.)

        irec += 1
    return trace

def array2su(recn, dt, traces_array):
    ''' convert array data to su stream
    '''
    from obspy.core.util import get_example_file

    # get a example stream and trace
    filename = get_example_file("1.su_first_trace")
    stream = obspy.read(filename, format='SU', byteorder='<')
    tr_example = stream[0]

    traces = []
    if recn > 1:
        for irec in range(recn):
            tr = copy.deepcopy(tr_example)
            tr.data = traces_array[irec,:]
            tr.stats.sampling_rate = 1./dt
            tr.stats.starttime = UTCDateTime("2021-01-01T00:00:00")
            # add trace
            traces += [tr]
    else: # signle trace
            tr = copy.deepcopy(tr_example)
            tr.data = traces_array
            tr.stats.sampling_rate = 1./dt
            tr.stats.starttime = UTCDateTime("2021-01-01T00:00:00")
            # add trace
            traces += [tr]
    # obspy stream
    return obspy.Stream(traces = traces)


# transform dictionary to objective
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d


##########################################################################
#                          numpy <=====> tensor     
##########################################################################

def numpy2tensor(a):
    """
        transform numpy data into tensor
    """
    if not torch.is_tensor(a):
        return torch.tensor(a,requires_grad=False)
    else:
        return a

def tensor2numpy(a):
    """
        transform tensor data into numpy
    """
    if not torch.is_tensor(a):
        return a 
    else:
        return a.detach().numpy()

##########################################################################
#                          list <=====> numpy     
##########################################################################

def list2numpy(a):
    """
        transform numpy data into tensor
    """
    if isinstance(a,list):
        return np.array(a)
    else:
        return a

def numpy2list(a):
    """
        transform numpy data into tensor
    """
    if not isinstance(a,list):
        return a.tolist()
    else:
        return a