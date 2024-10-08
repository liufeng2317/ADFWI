'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-04-19 22:07:06
* LastEditors: LiuFeng
* LastEditTime: 2024-05-10 14:53:29
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
import matplotlib.pyplot as plt
import numpy as np
import os 
from ADFWI.utils.utils import gpu2cpu
import warnings
warnings.filterwarnings("ignore")

def plot_bc():
    pass

def plot_bcx_bcz(bcx,bcz,dx=-1,dz=-1,figsize=(10,5),wspace=0.2,hspace=0.2,
                   cmap='gray_r',
                   tick_param     = {'labelsize':15},label_param = {'family':'Times New Roman','weight':'normal','size': 15},
                   colorbar_param = {'labelsize':15},title_param = {'family':'Times New Roman','weight':'normal','size': 20},
                   cbar_pad_fraction=0.1,cbar_height=0.05,
                   show=True,save_path="",save_dpi=300):
    """plot lambda and mu
    plot a data section using matplotlib.pcolormesh or imshow
    
    Parameters:
    ----------------
        - bcx (ndarrary or Tensor)          : the bondary condition along x axis
        - bcz (ndarrary or Tensor)          : the bondary condition along z axis
        - dx (float,optional)               : the spatial sampling interval along x-axis. Default: -1
        - dz (float,optional)               : the spatial sampling interval along z-axis. Default: -1
        - figsize (tuple,optional)          : the size of figure. Default (14,4)
        - wspace (float,optional)           : the width of the padding between subplots, as a fraction of the average Axes width.   Default: 0.2
        - hspace (float,optional)           : the height of the padding between subplots, as a fraction of the average Axes height. Default: 0.2
        - cmap (str or Colormap,optional)   : the Colormap instance or registered colormap name used to map scalar data to colors.  Default: 'gray'
        - tick_param (dict,optional)        : the fontdict for ticks
        - label_param (dict,optional)       : the fontdict for label
        - colorbar_param (dict,optional)    : the fontdict for colorbar
        - title_param (dict,optional)       : the fontdict for title
        - cbar_pad_fraction (float,optional): the padding interval for colorbar and main figure
        - cbar_height (float,optional)      : the height of the colorbar
        - show (bool,optional)              : showing the figure or not. Default True
        - save_path (str,optional)          : the saving path for the figure. Defualt:""
        - save_dpi (int,optional)           : the saving resolution for the figure. Default:300 dpi
    Retures
    -----------------
    None
    """
    bcx         = gpu2cpu(bcx)
    bxz      = gpu2cpu(bcz)
    
    nz,nx = bcx.shape
    x = np.arange(nx)*dx/1000
    z = np.arange(nz)*dz/1000
    X,Z = np.meshgrid(x,z)
    
    ######################################################
    fig,ax = plt.subplots(1,2,figsize=figsize)
    if dx>0 and dz >0:
        im1 = ax[0].pcolormesh(X,Z,bcx,cmap=cmap)
        ax[0].invert_yaxis()
        ax[0].set_xlabel("X (km)",**label_param)
        ax[0].set_ylabel("Z (km)",**label_param)
    else:
        im1 = ax[0].imshow(bcx,cmap=cmap)
        ax[0].set_xlabel("X",**label_param)
        ax[0].set_ylabel("Z",**label_param)
    cax1 = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y0-cbar_pad_fraction,
                         ax[0].get_position().width,cbar_height])
    cbar1 = plt.colorbar(im1,cax=cax1,orientation='horizontal')
    cbar1.ax.tick_params(**colorbar_param)
    ax[0].set_title("Boundary condition along X-axis",**title_param)
    ax[0].tick_params(**tick_param)
    ax[0].axis("scaled")
    
    ######################################################
    if dx>0 and dz >0:
        im2 = ax[1].pcolormesh(X,Z,bxz,cmap=cmap)
        ax[1].invert_yaxis()
        ax[1].set_xlabel("X (km)",**label_param)
        ax[1].set_ylabel("Z (km)",**label_param)
    else:
        im2 = ax[1].imshow(bxz,cmap=cmap)
        ax[1].set_xlabel("X",**label_param)
        ax[1].set_ylabel("Z",**label_param)
    cax2 = fig.add_axes([ax[1].get_position().x0,ax[1].get_position().y0-cbar_pad_fraction,
                         ax[1].get_position().width,cbar_height])
    cbar2 = plt.colorbar(im2,cax=cax2,orientation='horizontal')
    cbar2.ax.tick_params(**colorbar_param)
    ax[1].set_title("Boundary condition along Z-axis",**title_param)
    ax[1].tick_params(**tick_param)
    ax[1].axis("scaled")
    
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    if not save_path == "":
        plt.savefig(save_path,dpi=save_dpi,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_damp(damp,dx=-1,dz=-1,
                figsize=(8,8),wspace=0.2,hspace=0.2,
                cmap='gray_r',
                tick_param       = {'labelsize':15},label_param = {'family':'Times New Roman','weight':'normal','size': 15},
                colorbar_param   = {'labelsize':15},title_param = {'family':'Times New Roman','weight':'normal','size': 20},
                cbar_pad_fraction=0.12,cbar_height=0.03,
                show=True,save_path="",save_dpi=300):
    
    damp     = gpu2cpu(damp)
    nz,nx = damp.shape
    x = np.arange(nx)*dx/1000
    z = np.arange(nz)*dz/1000
    X,Z = np.meshgrid(x,z)
    
    ######################################################
    fig,ax = plt.subplots(1,1,figsize=figsize)
    if dx>0 and dz >0:
        im1 = ax.pcolormesh(X,Z,damp,cmap=cmap)
        ax.invert_yaxis()
        ax.set_xlabel("X (km)",**label_param)
        ax.set_ylabel("Z (km)",**label_param)
    else:
        im1 = ax.imshow(damp,cmap=cmap)
        ax.set_xlabel("X",**label_param)
        ax.set_ylabel("Z",**label_param)
    
    ax.set_title("ABL boundary condition",**title_param)
    ax.tick_params(**tick_param)
    ax.axis("scaled")
    
    cax1 = fig.add_axes([ax.get_position().x0,ax.get_position().y0-cbar_pad_fraction,
                         ax.get_position().width,cbar_height])
    cbar1 = plt.colorbar(im1,cax=cax1,orientation='horizontal')
    cbar1.ax.tick_params(**colorbar_param)
    
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    if not save_path == "":
        plt.savefig(save_path,dpi=save_dpi,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()