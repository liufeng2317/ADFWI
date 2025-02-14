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
    """
    Plot P-wave velocity (vp), S-wave velocity (vs), and density (rho) sections.

    Parameters
    ----------
    vp : ndarray or Tensor
        P-wave velocity (m/s).
    vs : ndarray or Tensor
        S-wave velocity (m/s).
    rho : ndarray or Tensor
        Density (kg/m^3).
    dx : float, optional
        Spatial sampling interval along the x-axis in meters. Default is -1.
    dz : float, optional
        Spatial sampling interval along the z-axis in meters. Default is -1.
    figsize : tuple, optional
        Size of the figure (width, height) in inches. Default is (14, 4).
    wspace : float, optional
        Width of the padding between subplots as a fraction of the average axes width. Default is 0.2.
    hspace : float, optional
        Height of the padding between subplots as a fraction of the average axes height. Default is 0.2.
    cmap : str or Colormap, optional
        Colormap used to map scalar data to colors. Default is 'jet'.
    vp_min : float, optional
        Minimum value for the P-wave velocity color scale. Default is None.
    vp_max : float, optional
        Maximum value for the P-wave velocity color scale. Default is None.
    vs_min : float, optional
        Minimum value for the S-wave velocity color scale. Default is None.
    vs_max : float, optional
        Maximum value for the S-wave velocity color scale. Default is None.
    rho_min : float, optional
        Minimum value for the density color scale. Default is None.
    rho_max : float, optional
        Maximum value for the density color scale. Default is None.
    tick_param : dict, optional
        Font parameters for tick labels. Default is {'labelsize': 15}.
    label_param : dict, optional
        Font parameters for axis labels. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}.
    colorbar_param : dict, optional
        Font parameters for the colorbar labels. Default is {'labelsize': 12}.
    title_param : dict, optional
        Font parameters for the title. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}.
    cbar_pad_fraction : float, optional
        Fraction of padding between the colorbar and the main figure. Default is 0.1.
    cbar_height : float, optional
        Height of the colorbar. Default is 0.05.
    show : bool, optional
        If True, displays the plot. Default is True.
    save_path : str, optional
        If provided, saves the figure to the specified path. Default is "" (no saving).
    save_dpi : int, optional
        DPI resolution for saving the figure. Default is 300.
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
    """
    Plot P-wave velocity (vp) and density (rho) data sections.

    Parameters
    ----------
    vp : ndarray or Tensor
        P-wave velocity in meters per second.
    rho : ndarray or Tensor
        Density in kilograms per cubic meter.
    dx : float, optional
        Spatial sampling interval along the x-axis in meters. Default is -1.
    dz : float, optional
        Spatial sampling interval along the z-axis in meters. Default is -1.
    figsize : tuple, optional
        Size of the figure. Default is (10, 5).
    wspace : float, optional
        The width of the padding between subplots, as a fraction of the average Axes width. Default is 0.2.
    hspace : float, optional
        The height of the padding between subplots, as a fraction of the average Axes height. Default is 0.2.
    cmap : str or Colormap, optional
        Colormap instance or registered colormap name. Default is 'jet'.
    tick_param : dict, optional
        Font properties for ticks. Default is {'labelsize': 15}.
    label_param : dict, optional
        Font properties for axis labels. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}.
    colorbar_param : dict, optional
        Font properties for colorbar. Default is {'labelsize': 15}.
    title_param : dict, optional
        Font properties for titles. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}.
    cbar_pad_fraction : float, optional
        Padding fraction between colorbar and the main plot. Default is 0.1.
    cbar_height : float, optional
        Height of the colorbar. Default is 0.05.
    show : bool, optional
        If True, display the plot. Default is True.
    save_path : str, optional
        Path to save the plot. If not provided, the plot will not be saved. Default is "".
    save_dpi : int, optional
        DPI for saving the plot. Default is 300.
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
    """
    Plot epsilon, delta, and gamma data sections.

    Parameters
    ----------
    eps : ndarray or Tensor
        Anisotropic parameter epsilon.
    delta : ndarray or Tensor
        Anisotropic parameter delta.
    gamma : ndarray or Tensor
        Anisotropic parameter gamma.
    dx : float, optional
        Spatial sampling interval along the x-axis in meters. Default is -1.
    dz : float, optional
        Spatial sampling interval along the z-axis in meters. Default is -1.
    figsize : tuple, optional
        Size of the figure. Default is (14, 4).
    wspace : float, optional
        Width of the padding between subplots, as a fraction of the average Axes width. Default is 0.2.
    hspace : float, optional
        Height of the padding between subplots, as a fraction of the average Axes height. Default is 0.2.
    cmap : str or Colormap, optional
        Colormap instance or registered colormap name used to map scalar data to colors. Default is 'jet'.
    tick_param : dict, optional
        Font properties for ticks. Default is {'labelsize': 15}.
    label_param : dict, optional
        Font properties for axis labels. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}.
    colorbar_param : dict, optional
        Font properties for the colorbar. Default is {'labelsize': 15}.
    title_param : dict, optional
        Font properties for titles. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}.
    cbar_pad_fraction : float, optional
        Padding fraction between the colorbar and the main plot. Default is 0.1.
    cbar_height : float, optional
        Height of the colorbar. Default is 0.05.
    show : bool, optional
        If True, display the plot. Default is True.
    save_path : str, optional
        Path to save the figure. Default is an empty string, indicating the figure will not be saved.
    save_dpi : int, optional
        Resolution of the saved figure in dots per inch. Default is 300.
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
    """
    Plot lambda and mu.

    Parameters
    ----------
    lam : ndarray or Tensor
        The lambda constant parameter.
    mu : ndarray or Tensor
        The mu constant parameter.
    dx : float, optional
        The spatial sampling interval along the x-axis. Default is -1.
    dz : float, optional
        The spatial sampling interval along the z-axis. Default is -1.
    figsize : tuple, optional
        The size of the figure. Default is (10, 5).
    wspace : float, optional
        The width of the padding between subplots, as a fraction of the average Axes width. Default is 0.2.
    hspace : float, optional
        The height of the padding between subplots, as a fraction of the average Axes height. Default is 0.2.
    cmap : str or Colormap, optional
        The Colormap instance or registered colormap name used to map scalar data to colors. Default is 'jet'.
    tick_param : dict, optional
        Font parameters for ticks. Default is {'labelsize': 15}.
    label_param : dict, optional
        Font parameters for axis labels. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}.
    colorbar_param : dict, optional
        Font parameters for colorbar. Default is {'labelsize': 15}.
    title_param : dict, optional
        Font parameters for title. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}.
    cbar_pad_fraction : float, optional
        The padding interval between the colorbar and the main figure. Default is 0.1.
    cbar_height : float, optional
        The height of the colorbar. Default is 0.05.
    show : bool, optional
        Whether to display the figure. Default is True.
    save_path : str, optional
        The file path to save the figure. Default is an empty string (not saved).
    save_dpi : int, optional
        The resolution of the saved figure. Default is 300 dpi.
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
    """
    Plot velocity model.
    
    Parameters
    ----------
    vel_model : ndarray or Tensor
        The velocity model to be plotted.
    title : str
        The title of the plot.
    dx : float, optional
        The spatial sampling interval along the x-axis. Default is -1.
    dz : float, optional
        The spatial sampling interval along the z-axis. Default is -1.
    figsize : tuple, optional
        The size of the figure. Default is (8, 8).
    wspace : float, optional
        The width of the padding between subplots, as a fraction of the average Axes width. Default is 0.2.
    hspace : float, optional
        The height of the padding between subplots, as a fraction of the average Axes height. Default is 0.2.
    cmap : str or Colormap, optional
        The Colormap instance or registered colormap name used to map scalar data to colors. Default is 'jet'.
    tick_param : dict, optional
        Font parameters for ticks. Default is {'labelsize': 15}.
    label_param : dict, optional
        Font parameters for axis labels. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}.
    colorbar_param : dict, optional
        Font parameters for colorbar. Default is {'labelsize': 15}.
    title_param : dict, optional
        Font parameters for title. Default is {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}.
    cbar_pad_fraction : float, optional
        The padding interval between the colorbar and the main figure. Default is 0.12.
    cbar_height : float, optional
        The height of the colorbar. Default is 0.03.
    vmin : float, optional
        The minimum value for the colormap. Default is None.
    vmax : float, optional
        The maximum value for the colormap. Default is None.
    show : bool, optional
        Whether to display the figure. Default is True.
    save_path : str, optional
        The file path to save the figure. Default is an empty string (not saved).
    save_dpi : int, optional
        The resolution of the saved figure. Default is 300 dpi.
    """
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