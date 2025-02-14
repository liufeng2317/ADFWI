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
    """
    Plot boundary conditions along X and Z axes using matplotlib.

    Parameters
    ----------
    bcx : ndarray or Tensor
        The boundary condition along the X-axis.
    bcz : ndarray or Tensor
        The boundary condition along the Z-axis.
    dx : float, optional
        The spatial sampling interval along the X-axis. Default is -1.
    dz : float, optional
        The spatial sampling interval along the Z-axis. Default is -1.
    figsize : tuple, optional
        The size of the figure. Default is (10, 5).
    wspace : float, optional
        The width of the padding between subplots, as a fraction of the average Axes width. Default is 0.2.
    hspace : float, optional
        The height of the padding between subplots, as a fraction of the average Axes height. Default is 0.2.
    cmap : str or Colormap, optional
        The colormap used to map scalar data to colors. Default is 'gray_r'.
    tick_param : dict, optional
        The font parameters for tick labels. Default is {'labelsize': 15}.
    label_param : dict, optional
        The font parameters for axis labels. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}.
    colorbar_param : dict, optional
        The font parameters for colorbar labels. Default is {'labelsize': 15}.
    title_param : dict, optional
        The font parameters for plot titles. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}.
    cbar_pad_fraction : float, optional
        The padding interval between colorbar and main figure. Default is 0.1.
    cbar_height : float, optional
        The height of the colorbar. Default is 0.05.
    show : bool, optional
        If True, displays the plot. Default is True.
    save_path : str, optional
        Path to save the figure. Default is an empty string, meaning no saving.
    save_dpi : int, optional
        The resolution (DPI) of the saved figure. Default is 300.
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
    """
    Plots the ABL (Absorbing Boundary Layer) boundary condition.

    This function visualizes the damping boundary condition in a 2D grid using either `pcolormesh` or `imshow` 
    from `matplotlib`, with optional parameters to customize the appearance and save the plot.

    Parameters
    ----------
    damp : ndarray or Tensor
        The damping boundary condition to be visualized. Should be a 2D array representing the damping values.
    dx : float, optional
        The spatial sampling interval along the x-axis in meters. Default is -1.
    dz : float, optional
        The spatial sampling interval along the z-axis in meters. Default is -1.
    figsize : tuple, optional
        The size of the figure in inches. Default is (8, 8).
    wspace : float, optional
        The width of the padding between subplots, as a fraction of the average Axes width. Default is 0.2.
    hspace : float, optional
        The height of the padding between subplots, as a fraction of the average Axes height. Default is 0.2.
    cmap : str or Colormap, optional
        The Colormap instance or registered colormap name used to map scalar data to colors. Default is 'gray_r'.
    tick_param : dict, optional
        The fontdict for ticks, e.g., `{'labelsize': 15}`.
    label_param : dict, optional
        The fontdict for labels (x and y axes), e.g., `{'family': 'Times New Roman', 'weight': 'normal', 'size': 15}`.
    colorbar_param : dict, optional
        The fontdict for colorbar labels, e.g., `{'labelsize': 15}`.
    title_param : dict, optional
        The fontdict for the title, e.g., `{'family': 'Times New Roman', 'weight': 'normal', 'size': 20}`.
    cbar_pad_fraction : float, optional
        The padding interval for colorbar and main figure, default is 0.12.
    cbar_height : float, optional
        The height of the colorbar, default is 0.03.
    show : bool, optional
        If True, the figure will be shown. Default is True.
    save_path : str, optional
        The file path to save the figure. If empty, the figure will not be saved. Default is an empty string.
    save_dpi : int, optional
        The resolution (dots per inch) for saving the figure. Default is 300 dpi.
    """
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