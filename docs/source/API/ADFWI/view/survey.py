import numpy as np
import matplotlib.pyplot as plt
from ADFWI.utils import gpu2cpu
import warnings
warnings.filterwarnings("ignore")

def plot_survey(src_x,src_z,rcv_x,rcv_z,vel_model,dx=-1,dz=-1,
                figsize=(8,8),wspace=0.2,hspace=0.2,
                cmap='jet',
                tick_param       = {'labelsize':15},label_param = {'family':'Times New Roman','weight':'normal','size': 15},
                colorbar_param   = {'labelsize':15},title_param = {'family':'Times New Roman','weight':'normal','size': 20},
                cbar_pad_fraction=0.12,cbar_height=0.03,
                show=True,save_path="",save_dpi=300):
    """
    Plot the survey setup, including the velocity model, sources, and receivers.

    Parameters
    ----------
    src_x : array-like
        x-coordinates of the sources.
    src_z : array-like
        z-coordinates of the sources.
    rcv_x : array-like
        x-coordinates of the receivers.
    rcv_z : array-like
        z-coordinates of the receivers.
    vel_model : np.ndarray
        2D array representing the velocity model.
    dx : float, optional
        Horizontal spacing of the model, in meters. Default is -1.
    dz : float, optional
        Vertical spacing of the model, in meters. Default is -1.
    figsize : tuple, optional
        Size of the figure in inches. Default is (8, 8).
    wspace : float, optional
        Horizontal space between subplots. Default is 0.2.
    hspace : float, optional
        Vertical space between subplots. Default is 0.2.
    cmap : str, optional
        Colormap for the velocity model plot. Default is 'jet'.
    tick_param : dict, optional
        Parameters for tick labels. Default is {'labelsize': 15}.
    label_param : dict, optional
        Font parameters for axis labels. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}.
    colorbar_param : dict, optional
        Font parameters for the colorbar labels. Default is {'labelsize': 15}.
    title_param : dict, optional
        Font parameters for the title. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}.
    cbar_pad_fraction : float, optional
        Fraction of the colorbar height for padding. Default is 0.12.
    cbar_height : float, optional
        Height of the colorbar. Default is 0.03.
    show : bool, optional
        If True, display the plot. Default is True.
    save_path : str, optional
        If provided, saves the figure to the specified path. Default is "" (no saving).
    save_dpi : int, optional
        Dots per inch (DPI) for saving the figure. Default is 300.
    """
    vel_model     = gpu2cpu(vel_model)
    nz,nx = vel_model.shape
    x = np.arange(nx)*dx/1000
    z = np.arange(nz)*dz/1000
    X,Z = np.meshgrid(x,z)
    
    ######################################################
    fig,ax = plt.subplots(1,1,figsize=figsize)
    if dx>0 and dz >0:
        im1 = ax.pcolormesh(X,Z,vel_model,cmap=cmap)
        ax.scatter(rcv_x*dx,rcv_z*dz,facecolor='w',edgecolors='k',marker="v",s=20,label="Receivers")
        ax.scatter(src_x*dx,src_z*dz,c='r',marker="*",s=20,label="Sources")
        ax.invert_yaxis()
        ax.set_xlabel("X (km)",**label_param)
        ax.set_ylabel("Z (km)",**label_param)
    else:
        im1 = ax.imshow(vel_model,cmap=cmap)
        ax.scatter(rcv_x,rcv_z,facecolor='w',edgecolors='k',marker="v",s=20,label="Receivers")
        ax.scatter(src_x,src_z,c='r',marker="*",s=20,label="Sources")
        ax.set_xlabel("X",**label_param)
        ax.set_ylabel("Z",**label_param)
    
    ax.legend()
    ax.set_title("Observed System",**title_param)
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
        
def plot_wavelet(tlist,wavelet,
                figsize=(6,4),
                color='k',linestyle='-',linewidth=1,
                tick_param       = {'labelsize':15},label_param = {'family':'Times New Roman','weight':'normal','size': 15},
                colorbar_param   = {'labelsize':15},title_param = {'family':'Times New Roman','weight':'normal','size': 15},
                cbar_pad_fraction=0.12,cbar_height=0.03,
                show=True,save_path="",save_dpi=300):
    """
    Plot the source wavelet.

    Parameters
    ----------
    tlist : array-like
        Time values for plotting the wavelet.
    wavelet : array-like
        Amplitude values corresponding to the wavelet at each time point.
    figsize : tuple, optional
        Size of the figure in inches. Default is (6, 4).
    color : str, optional
        Color of the wavelet plot. Default is 'k' (black).
    linestyle : str, optional
        Line style for the wavelet plot. Default is '-' (solid line).
    linewidth : float, optional
        Line width of the plot. Default is 1.
    tick_param : dict, optional
        Parameters for tick labels. Default is {'labelsize': 15}.
    label_param : dict, optional
        Font parameters for axis labels. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}.
    colorbar_param : dict, optional
        Font parameters for the colorbar labels. Default is {'labelsize': 15}.
    title_param : dict, optional
        Font parameters for the title. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}.
    cbar_pad_fraction : float, optional
        Fraction of the colorbar height for padding. Default is 0.12.
    cbar_height : float, optional
        Height of the colorbar. Default is 0.03.
    show : bool, optional
        If True, display the plot. Default is True.
    save_path : str, optional
        If provided, saves the figure to the specified path. Default is "" (no saving).
    save_dpi : int, optional
        Dots per inch (DPI) for saving the figure. Default is 300.
    """
    fig,ax = plt.subplots(1,1,figsize=figsize)

    ax.plot(tlist,wavelet,c=color,linestyle=linestyle,linewidth=linewidth)
    ax.set_xlabel("Times (s)",**label_param)
    ax.set_ylabel("Amplitude",**label_param)
    
    ax.set_title("Source Wavelets",**title_param)
    ax.tick_params(**tick_param)
    ax.grid()
    if not save_path == "":
        plt.savefig(save_path,dpi=save_dpi,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()