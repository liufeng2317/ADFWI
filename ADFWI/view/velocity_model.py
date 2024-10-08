'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-04-19 17:05:45
* LastEditors: LiuFeng
* LastEditTime: 2024-05-07 16:05:36
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
import matplotlib.pyplot as plt
import numpy as np
import os 
from ADFWI.utils.utils import gpu2cpu


def plot_vp_vs_rho(vp,vs,rho,dx=-1,dz=-1,figsize=(14,4),wspace=0.2,hspace=0.2,
                   cmap='jet',title="",
                   vp_min   = None  ,vp_max =None,
                   vs_min   = None  ,vs_max =None,
                   rho_min  = None  ,rho_max=None,
                   tick_param  = {'labelsize':15},label_param = {'family':'Times New Roman','weight':'normal','size': 15},
                   colorbar_param = {'labelsize':12},title_param = {'family':'Times New Roman','weight':'normal','size': 20},
                   cbar_pad_fraction=0.1,cbar_height=0.05,
                   show=True,save_path="",save_dpi=300):
    """plot vp/vs and density
        plot a data section using matplotlib.pcolormesh or imshow
    
    Parameters:
    ----------------
        - vp (ndarrary or Tensor)           : p-wave velocity (m/s)
        - vs (ndarrary or Tensor)           : s-wave velocity (m/s)
        - rho (ndarrary or Tensor)          : densicyt (kg/m^3)
        - dx (float,optional)               : the spatial sampling interval along x-axis. Default: -1
        - dz (float,optional)               : the spatial sampling interval along z-axis. Default: -1
        - figsize (tuple,optional)          : the size of figure. Default (14,4)
        - wspace (float,optional)           : the width of the padding between subplots, as a fraction of the average Axes width.   Default: 0.2
        - hspace (float,optional)           : the height of the padding between subplots, as a fraction of the average Axes height. Default: 0.2
        - cmap (str or Colormap,optional)   : the Colormap instance or registered colormap name used to map scalar data to colors.  Default: 'jet'
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
    vp = gpu2cpu(vp)
    vs = gpu2cpu(vs)
    rho = gpu2cpu(rho)
    
    nz,nx = vp.shape
    x = np.arange(nx)*dx/1000
    z = np.arange(nz)*dz/1000
    X,Z = np.meshgrid(x,z)
    
    ######################################################
    fig,ax = plt.subplots(1,3,figsize=figsize)
    if dx>0 and dz >0:
        if vp_min is not None and vp_max is not None:
            im1 = ax[0].pcolormesh(X,Z,vp,cmap=cmap,vmin=vp_min,vmax=vp_max)
        else:
            im1 = ax[0].pcolormesh(X,Z,vp,cmap=cmap)
        ax[0].invert_yaxis()
        ax[0].set_xlabel("X (km)",**label_param)
        ax[0].set_ylabel("Z (km)",**label_param)
    else:
        if vp_min is not None and vp_max is not None:
            im1 = ax[0].imshow(vp,cmap=cmap,vmin=vp_min,vmax=vp_max)
        else:
            im1 = ax[0].imshow(vp,cmap=cmap)
        ax[0].set_xlabel("X",**label_param)
        ax[0].set_ylabel("Z",**label_param)
    ax[0].set_title("vp (m/s)",**title_param)
    ax[0].tick_params(**tick_param)
    ax[0].axis("scaled")
    cax1 = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y0-cbar_pad_fraction,
                         ax[0].get_position().width,cbar_height])
    cbar1 = plt.colorbar(im1,cax=cax1,orientation='horizontal')
    cbar1.ax.tick_params(**colorbar_param)

    ######################################################
    if dx>0 and dz >0:
        if vs_min is not None and vs_max is not None:
            im2 = ax[1].pcolormesh(X,Z,vs,cmap=cmap,vmin=vs_min,vmax=vs_max)
        else:
            im2 = ax[1].pcolormesh(X,Z,vs,cmap=cmap)
        ax[1].invert_yaxis()
        ax[1].set_xlabel("X (km)",**label_param)
        ax[1].set_ylabel("Z (km)",**label_param)
    else:
        if vs_min is not None and vs_max is not None:
            im2 = ax[1].imshow(vs,cmap=cmap,vmin=vs_min,vmax=vs_max)
        else:
            im2 = ax[1].imshow(vs,cmap=cmap)
        ax[1].set_xlabel("X",**label_param)
        ax[1].set_ylabel("Z",**label_param)
    ax[1].set_title("vs (m/s)",**title_param)
    ax[1].tick_params(**tick_param)
    ax[1].axis("scaled")
    cax2 = fig.add_axes([ax[1].get_position().x0,ax[1].get_position().y0-cbar_pad_fraction,
                         ax[1].get_position().width,cbar_height])
    cbar2 = plt.colorbar(im2,cax=cax2,orientation='horizontal')
    cbar2.ax.tick_params(**colorbar_param)
    
    ######################################################
    if dx>0 and dz >0:
        if rho_min is not None and rho_max is not None:
            im3 = ax[2].pcolormesh(X,Z,rho,cmap=cmap,vmin=rho_min,vmax=rho_max)
        else:
            im3 = ax[2].pcolormesh(X,Z,rho,cmap=cmap)
        ax[2].invert_yaxis()
        ax[2].set_xlabel("X (km)",**label_param)
        ax[2].set_ylabel("Z (km)",**label_param)
    else:
        if rho_min is not None and rho_max is not None:
            im3 = ax[2].imshow(rho,cmap=cmap,vmin=rho_min,vmax=rho_max)
        else:
            im3 = ax[2].imshow(rho,cmap=cmap)
        ax[2].set_xlabel("X (km)",**label_param)
        ax[2].set_ylabel("Z (km)",**label_param)

    ax[2].set_title(r"$\rho$ (kg/m^3)",**title_param)
    ax[2].tick_params(**tick_param)
    ax[2].axis("scaled")
    cax3 = fig.add_axes([ax[2].get_position().x0,ax[2].get_position().y0-cbar_pad_fraction,
                         ax[2].get_position().width,cbar_height])
    cbar3 = plt.colorbar(im3,cax=cax3,orientation='horizontal')
    cbar3.ax.tick_params(**colorbar_param)
    
    if not title == "":
        fig.suptitle(title,**title_param)
        
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    if not save_path == "":
        plt.savefig(save_path,dpi=save_dpi,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_vp_rho(vp,rho,dx=-1,dz=-1,figsize=(10,5),wspace=0.2,hspace=0.2,
                   cmap='jet',
                   tick_param       = {'labelsize':15},label_param = {'family':'Times New Roman','weight':'normal','size': 15},
                   colorbar_param   = {'labelsize':15},title_param = {'family':'Times New Roman','weight':'normal','size': 20},
                   cbar_pad_fraction=0.1,cbar_height=0.05,
                   show=True,save_path="",save_dpi=300):
    """plot vp and rho
    plot a data section using matplotlib.pcolormesh or imshow
    
    Parameters:
    ----------------
        - vp (ndarrary or Tensor)           : p-wave velocity (m/s)
        - rho (ndarrary or Tensor)          : densicyt (kg/m^3)
        - dx (float,optional)               : the spatial sampling interval along x-axis. Default: -1
        - dz (float,optional)               : the spatial sampling interval along z-axis. Default: -1
        - figsize (tuple,optional)          : the size of figure. Default (14,4)
        - wspace (float,optional)           : the width of the padding between subplots, as a fraction of the average Axes width.   Default: 0.2
        - hspace (float,optional)           : the height of the padding between subplots, as a fraction of the average Axes height. Default: 0.2
        - cmap (str or Colormap,optional)   : the Colormap instance or registered colormap name used to map scalar data to colors.  Default: 'jet'
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
    vp          = gpu2cpu(vp)
    rho         = gpu2cpu(rho)
    
    nz,nx = vp.shape
    x = np.arange(nx)*dx/1000
    z = np.arange(nz)*dz/1000
    X,Z = np.meshgrid(x,z)
    
    ######################################################
    fig,ax = plt.subplots(1,2,figsize=figsize)
    if dx>0 and dz >0:
        im1 = ax[0].pcolormesh(X,Z,vp,cmap=cmap)
        ax[0].invert_yaxis()
        ax[0].set_xlabel("X (km)",**label_param)
        ax[0].set_ylabel("Z (km)",**label_param)
    else:
        im1 = ax[0].imshow(vp)
        ax[0].set_xlabel("X",**label_param)
        ax[0].set_ylabel("Z",**label_param)
    cax1 = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y0-cbar_pad_fraction,
                         ax[0].get_position().width,cbar_height])
    cbar1 = plt.colorbar(im1,cax=cax1,orientation='horizontal')
    cbar1.ax.tick_params(**colorbar_param)
    ax[0].set_title("vp (m/s)",**title_param)
    ax[0].tick_params(**tick_param)
    ax[0].axis("scaled")
    
    ######################################################
    if dx>0 and dz >0:
        im2 = ax[1].pcolormesh(X,Z,rho,cmap=cmap)
        ax[1].invert_yaxis()
        ax[1].set_xlabel("X (km)",**label_param)
        ax[1].set_ylabel("Z (km)",**label_param)
    else:
        im2 = ax[1].imshow(rho,cmap=cmap)
        ax[1].set_xlabel("X",**label_param)
        ax[1].set_ylabel("Z",**label_param)
    cax2 = fig.add_axes([ax[1].get_position().x0,ax[1].get_position().y0-cbar_pad_fraction,
                         ax[1].get_position().width,cbar_height])
    cbar2 = plt.colorbar(im2,cax=cax2,orientation='horizontal')
    cbar2.ax.tick_params(**colorbar_param)
    ax[1].set_title(r"$\rho$ (kg/m^3)",**title_param)
    ax[1].tick_params(**tick_param)
    ax[1].axis("scaled")
    
    
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    if not save_path == "":
        plt.savefig(save_path,dpi=save_dpi,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_eps_delta_gamma(eps,delta,gamma,dx=-1,dz=-1,figsize=(14,4),wspace=0.2,hspace=0.2,
                   cmap='jet',
                   tick_param  = {'labelsize':15},label_param = {'family':'Times New Roman','weight':'normal','size': 15},
                   colorbar_param = {'labelsize':15},title_param = {'family':'Times New Roman','weight':'normal','size': 20},
                   cbar_pad_fraction=0.1,cbar_height=0.05,
                   show=True,save_path="",save_dpi=300):
    """plot epsilon/delta and gamma
    plot a data section using matplotlib.pcolormesh or imshow
    
    Parameters:
    ----------------
        - eps   (ndarrary or Tensor)        : the anisotropic parameter
        - delta (ndarrary or Tensor)        : the anisotropic parameter
        - gamma (ndarrary or Tensor)        : the anisotropic parameter
        - dx (float,optional)               : the spatial sampling interval along x-axis. Default: -1
        - dz (float,optional)               : the spatial sampling interval along z-axis. Default: -1
        - figsize (tuple,optional)          : the size of figure. Default (14,4)
        - wspace (float,optional)           : the width of the padding between subplots, as a fraction of the average Axes width.   Default: 0.2
        - hspace (float,optional)           : the height of the padding between subplots, as a fraction of the average Axes height. Default: 0.2
        - cmap (str or Colormap,optional)   : the Colormap instance or registered colormap name used to map scalar data to colors.  Default: 'jet'
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
    eps     = gpu2cpu(eps)
    delta   = gpu2cpu(delta)
    gamma   = gpu2cpu(gamma)
    
    nz,nx = eps.shape
    x = np.arange(nx)*dx/1000
    z = np.arange(nz)*dz/1000
    X,Z = np.meshgrid(x,z)
    
    ######################################################
    fig,ax = plt.subplots(1,3,figsize=figsize)
    if dx>0 and dz >0:
        im1 = ax[0].pcolormesh(X,Z,eps,cmap=cmap)
        ax[0].invert_yaxis()
        ax[0].set_xlabel("X (km)",**label_param)
        ax[0].set_ylabel("Z (km)",**label_param)
    else:
        im1 = ax[0].imshow(eps,cmap=cmap)
        ax[0].set_xlabel("X",**label_param)
        ax[0].set_ylabel("Z",**label_param)
    cax1 = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y0-cbar_pad_fraction,
                         ax[0].get_position().width,cbar_height])
    cbar1 = plt.colorbar(im1,cax=cax1,orientation='horizontal')
    cbar1.ax.tick_params(**colorbar_param)
    ax[0].set_title("eps",**title_param)
    ax[0].tick_params(**tick_param)
    ax[0].axis("scaled")
    
    ######################################################
    if dx>0 and dz >0:
        im2 = ax[1].pcolormesh(X,Z,delta,cmap=cmap)
        ax[1].invert_yaxis()
        ax[1].set_xlabel("X (km)",**label_param)
        ax[1].set_ylabel("Z (km)",**label_param)
    else:
        im2 = ax[0].imshow(delta,cmap=cmap)
        ax[1].set_xlabel("X",**label_param)
        ax[1].set_ylabel("Z",**label_param)
    cax2 = fig.add_axes([ax[1].get_position().x0,ax[1].get_position().y0-cbar_pad_fraction,
                         ax[1].get_position().width,cbar_height])
    cbar2 = plt.colorbar(im2,cax=cax2,orientation='horizontal')
    cbar2.ax.tick_params(**colorbar_param)
    ax[1].set_title("delta",**title_param)
    ax[1].tick_params(**tick_param)
    ax[1].axis("scaled")
    
    ######################################################
    if dx>0 and dz >0:
        im3 = ax[2].pcolormesh(X,Z,gamma,cmap=cmap)
        ax[2].invert_yaxis()
        ax[2].set_xlabel("X (km)",**label_param)
        ax[2].set_ylabel("Z (km)",**label_param)
    else:
        im3 = ax[2].imshow(gamma,cmap=cmap)
        ax[2].set_xlabel("X",**label_param)
        ax[2].set_ylabel("Z",**label_param)
    cax3 = fig.add_axes([ax[2].get_position().x0,ax[2].get_position().y0-cbar_pad_fraction,
                         ax[2].get_position().width,cbar_height])
    cbar3 = plt.colorbar(im3,cax=cax3,orientation='horizontal')
    cbar3.ax.tick_params(**colorbar_param)
    ax[2].set_title("gamma",**title_param)
    ax[2].tick_params(**tick_param)
    ax[2].axis("scaled")
    
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    if not save_path == "":
        plt.savefig(save_path,dpi=save_dpi,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_lam_mu(lam,mu,dx=-1,dz=-1,figsize=(10,5),wspace=0.2,hspace=0.2,
                   cmap='jet',
                   tick_param       = {'labelsize':15},label_param = {'family':'Times New Roman','weight':'normal','size': 15},
                   colorbar_param   = {'labelsize':15},title_param = {'family':'Times New Roman','weight':'normal','size': 20},
                   cbar_pad_fraction=0.1,cbar_height=0.05,
                   show=True,save_path="",save_dpi=300):
    """plot lambda and mu
    plot a data section using matplotlib.pcolormesh or imshow
    
    Parameters:
    ----------------
        - lam (ndarrary or Tensor)          : the lame constant parameter
        - mu (ndarrary or Tensor)           : the lame constant parameter
        - dx (float,optional)               : the spatial sampling interval along x-axis. Default: -1
        - dz (float,optional)               : the spatial sampling interval along z-axis. Default: -1
        - figsize (tuple,optional)          : the size of figure. Default (14,4)
        - wspace (float,optional)           : the width of the padding between subplots, as a fraction of the average Axes width.   Default: 0.2
        - hspace (float,optional)           : the height of the padding between subplots, as a fraction of the average Axes height. Default: 0.2
        - cmap (str or Colormap,optional)   : the Colormap instance or registered colormap name used to map scalar data to colors.  Default: 'jet'
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
    lam     = gpu2cpu(lam)
    mu      = gpu2cpu(mu)
    
    nz,nx = lam.shape
    x = np.arange(nx)*dx/1000
    z = np.arange(nz)*dz/1000
    X,Z = np.meshgrid(x,z)
    
    ######################################################
    fig,ax = plt.subplots(1,2,figsize=figsize)
    if dx>0 and dz >0:
        im1 = ax[0].pcolormesh(X,Z,lam,cmap=cmap)
        ax[0].invert_yaxis()
        ax[0].set_xlabel("X (km)",**label_param)
        ax[0].set_ylabel("Z (km)",**label_param)
    else:
        im1 = ax[0].imshow(lam)
        ax[0].set_xlabel("X",**label_param)
        ax[0].set_ylabel("Z",**label_param)
    cax1 = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y0-cbar_pad_fraction,
                         ax[0].get_position().width,cbar_height])
    cbar1 = plt.colorbar(im1,cax=cax1,orientation='horizontal')
    cbar1.ax.tick_params(**colorbar_param)
    ax[0].set_title(r"$\lambda$",**title_param)
    ax[0].tick_params(**tick_param)
    ax[0].axis("scaled")
    
    ######################################################
    if dx>0 and dz >0:
        im2 = ax[1].pcolormesh(X,Z,mu,cmap=cmap)
        ax[1].invert_yaxis()
        ax[1].set_xlabel("X (km)",**label_param)
        ax[1].set_ylabel("Z (km)",**label_param)
    else:
        im2 = ax[1].imshow(mu,cmap=cmap)
        ax[1].set_xlabel("X",**label_param)
        ax[1].set_ylabel("Z",**label_param)
    cax2 = fig.add_axes([ax[1].get_position().x0,ax[1].get_position().y0-cbar_pad_fraction,
                         ax[1].get_position().width,cbar_height])
    cbar2 = plt.colorbar(im2,cax=cax2,orientation='horizontal')
    cbar2.ax.tick_params(**colorbar_param)
    ax[1].set_title(r"$\mu$",**title_param)
    ax[1].tick_params(**tick_param)
    ax[1].axis("scaled")
    
    
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    if not save_path == "":
        plt.savefig(save_path,dpi=save_dpi,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
        
def plot_model(vel_model,title,dx=-1,dz=-1,
                figsize=(8,8),wspace=0.2,hspace=0.2,
                cmap='jet',
                tick_param       = {'labelsize':15},label_param = {'family':'Times New Roman','weight':'normal','size': 15},
                colorbar_param   = {'labelsize':15},title_param = {'family':'Times New Roman','weight':'normal','size': 20},
                cbar_pad_fraction=0.12,cbar_height=0.03,
                vmin = None,vmax=None,
                show=True,save_path="",save_dpi=300):
    
    vel_model     = gpu2cpu(vel_model)
    nz,nx = vel_model.shape
    x = np.arange(nx)*dx/1000
    z = np.arange(nz)*dz/1000
    X,Z = np.meshgrid(x,z)
    
    ######################################################
    fig,ax = plt.subplots(1,1,figsize=figsize)
    if dx>0 and dz >0:
        if vmin is not None and vmax is not None:
            im1 = ax.pcolormesh(X,Z,vel_model,cmap=cmap,vmin=vmin,vmax=vmax)
        else:
            im1 = ax.pcolormesh(X,Z,vel_model,cmap=cmap)
        ax.invert_yaxis()
        ax.set_xlabel("X (km)",**label_param)
        ax.set_ylabel("Z (km)",**label_param)
    else:
        if vmin is not None and vmax is not None:
            im1 = ax.imshow(vel_model,cmap=cmap,vmin=vmin,vmax=vmax)
        else:
            im1 = ax.imshow(vel_model,cmap=cmap)
        ax.set_xlabel("X",**label_param)
        ax.set_ylabel("Z",**label_param)
    
    ax.set_title(title,**title_param)
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
    return