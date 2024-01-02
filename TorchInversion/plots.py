'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-06-28 15:24:45
* LastEditors: LiuFeng
* LastEditTime: 2023-12-31 10:02:53
* FilePath: /ADFWI/TorchInversion/plots.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

import matplotlib.pyplot as plt
import numpy as np
import os 

from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_model(model,save_path="",show=False):
    v = model.v
    rho = model.rho
    #Plot velocity and density after pad or before pad
    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(211)
    im = plt.imshow(v,cmap='jet_r')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im,cax=cax)
    plt.title("v(m/s)")
    
    ax = plt.subplot(212)
    im = plt.imshow(rho,cmap='jet_r')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im,cax=cax)
    plt.title("rho")
    
    if save_path != "":
        plt.savefig(save_path,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
        
        
def plot_wavelet(src,save_path="",show=False):
    stf_t = src.stf_t
    stf_val = src.stf_val
    if stf_val.ndim == 2:
        stf_val = stf_val[0,:]
    nt = stf_val.shape[0]
    #Plot wavelet after intergration
    fig = plt.figure(figsize=(8,6))
    WAVELET_EXTENT = (0, 2000*0.002, -1.2, 1.2)
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_label_text('Time (s)', fontsize=12)
    ax.yaxis.set_label_text('Normalized Amplitude', fontsize=12)
    ax.set_title('Source wavelet', fontsize=16)
    # ax.axis(WAVELET_EXTENT)
    # ax.plot(st[0:nt+1], s[0:nt+1] / abs(s[0:nt+1]).max(), 'g-')
    ax.plot(stf_t[0:nt+1], stf_val[0:nt+1], 'g-')
    
    if save_path != "":
        plt.savefig(save_path,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
        

def plot_dampRegion(model,save_path="",show=False):
    damp_global = model.damp_global
    fig = plt.figure(figsize=(12,8))
    ax = plt.axes()
    im = plt.imshow(damp_global,cmap='gray_r')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im,cax=cax)
    
    if save_path != "":
        plt.savefig(save_path,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
        
def plot_observeSystem(param,model,src,rcv,save_path="",show=False):
    v = model.v
    rcv_y = rcv.rcv_y
    rcv_x = rcv.rcv_x
    src_y = src.src_y
    src_x = src.src_x
    pml = param.pml
    
    fig = plt.figure(figsize=(12,8))
    ax = plt.axes()
    # im = plt.imshow(v,cmap='jet_r',vmin=3000,vmax=3030)
    im = plt.imshow(v,cmap='jet_r')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    # plt.colorbar(im,cax=cax)
    plt.colorbar(im,cax=cax)
    ax.scatter(rcv_y,rcv_x,20,marker="v",c='w',label="receiver")
    ax.scatter(src_y-pml,src_x-pml,20,marker='*',c='k',label="source")
    ax.legend(fontsize=12)
    if save_path != "":
        plt.savefig(save_path,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
        
        
        
def insert_zeros(trace, tt=None):
    """Insert zero locations in data trace and tt vector based on linear fit"""

    if tt is None:
        tt = np.arange(len(trace))

    # Find zeros
    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    # split tt and trace
    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    # insert zeros in tt and trace
    for i in range(len(tt_zero)):
        tt_zi = np.hstack(
            (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack(
            (trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi


def wiggle_input_check(data, tt, xx, sf, verbose):
    ''' Helper function for wiggle() and traces() to check input

    '''

    # Input check for verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    # Input check for data
    if type(data).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")

    if len(data.shape) != 2:
        raise ValueError("data must be a 2D array")

    # Input check for tt
    if tt is None:
        tt = np.arange(data.shape[0])
        if verbose:
            print("tt is automatically generated.")
            print(tt)
    else:
        if type(tt).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(tt.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")

    # Input check for xx
    if xx is None:
        xx = np.arange(data.shape[1])
        if verbose:
            print("xx is automatically generated.")
            print(xx)
    else:
        if type(xx).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(xx.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")
        if verbose:
            print(xx)

    # Input check for streth factor (sf)
    if not isinstance(sf, (int, float)):
        raise TypeError("Strech factor(sf) must be a number")

    # Compute trace horizontal spacing
    ts = np.min(np.diff(xx))

    # Rescale data by trace_spacing and strech_factor
    data_max_std = np.max(np.std(data, axis=0))
    data = data / data_max_std * ts * sf

    return data, tt, xx, ts


def wiggle(data, tt=None, xx=None, color='k', sf=0.15, verbose=False,save_path="",show=False):
    '''Wiggle plot of a sesimic data section

    Syntax examples:
        wiggle(data)
        wiggle(data, tt)
        wiggle(data, tt, xx)
        wiggle(data, tt, xx, color)
        fi = wiggle(data, tt, xx, color, sf, verbose)

    Use the column major order for array as in Fortran to optimal performance.

    The following color abbreviations are supported:

    ==========  ========
    character   color
    ==========  ========
    'b'         blue
    'g'         green
    'r'         red
    'c'         cyan
    'm'         magenta
    'y'         yellow
    'k'         black
    'w'         white
    ==========  ========


    '''

    # Input check
    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)

    # Plot data using matplotlib.pyplot
    Ntr = data.shape[1]

    ax = plt.gca()
    for ntr in range(Ntr):
        trace = data[:, ntr]
        offset = xx[ntr]

        if verbose:
            print(offset)

        trace_zi, tt_zi = insert_zeros(trace, tt)
        ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                        where=trace_zi >= 0,
                        facecolor=color)
        ax.plot(trace_zi + offset, tt_zi, color)

    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()
    if save_path != "":
        plt.savefig(save_path,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
        
        
def wiggle_cmp(data1,data2, tt=None, xx=None, color1='k',color2='r', sf=0.15, verbose=False,save_path="",show=False):
    '''Wiggle plot of a sesimic data section
    '''
    # Input check
    data1, tt, xx, ts = wiggle_input_check(data1, tt, xx, sf, verbose)
    data2, tt, xx, ts = wiggle_input_check(data2, tt, xx, sf, verbose)

    if not data1.shape == data2.shape:
        raise Exception("The size of data1 = {} not equal to data2 = {}".format(str(data1.shape),str(data2.shape)))
    
    # Plot data using matplotlib.pyplot
    Ntr = data1.shape[1]
    ax = plt.gca()
    for ntr in range(Ntr):
        trace1 = data1[:, ntr]
        trace2 = data2[:, ntr]
        offset = xx[ntr]

        if verbose:
            print(offset)

        trace_zi1, tt_zi1 = insert_zeros(trace1, tt)
        trace_zi2, tt_zi2 = insert_zeros(trace2, tt)
        # ax.fill_betweenx(tt_zi1, offset, trace_zi1 + offset,where=trace_zi1 >= 0,facecolor=color)
        # ax.fill_betweenx(tt_zi2, offset, trace_zi2 + offset,where=trace_zi2 >= 0,facecolor=color)
        ax.plot(trace_zi1 + offset, tt_zi1, color1)
        ax.plot(trace_zi2 + offset, tt_zi2, color2)

    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()
    if save_path != "":
        plt.savefig(save_path,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
        
        
##############################################
#           Inversion Result
##############################################
def plot_inversion_iter(i,v,grads,save_path):
    if i%1==0:
        plt.figure()
        plt.pcolormesh(v,cmap="jet_r")
        plt.title("Iter {}".format(i))
        plt.xlabel("x (km)")
        plt.ylabel("z (km)")
        plt.gca().invert_yaxis()
        plt.axis('scaled')
        plt.colorbar(shrink=0.5)
        plt.savefig(os.path.join(save_path,"inv/model/{}.png".format(i)),bbox_inches="tight")
        plt.close()
        
        plt.figure()
        plt.pcolormesh(grads,cmap="bwr_r")
        plt.title("Iter {}".format(i))
        plt.xlabel("x (km)")
        plt.ylabel("z (km)")
        plt.gca().invert_yaxis()
        plt.axis('scaled')
        plt.colorbar(shrink=0.5)
        plt.savefig(os.path.join(save_path,"inv/grad/{}.png".format(i)),bbox_inches="tight")
        plt.close()