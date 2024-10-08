import numpy as np
import matplotlib.pyplot as plt
from ADFWI.utils.utils import gpu2cpu
import warnings
warnings.filterwarnings("ignore")

########################################################
#              waveform （single Trace）
########################################################
def plot_waveform_trace(data,shot=0,trace=0,dt=None,figsize=(12,4),color='k',
                    tick_param     = {'labelsize':18},label_param = {'family':'Times New Roman','weight':'normal','size': 18},
                    title_param = {'family':'Times New Roman','weight':'normal','size': 20},
                    show=True,save_path="",save_dpi=300):
    """plot single trace waveform
    Parameters
    ----------------
        - data (ndarray or Tensors)     : 3D waveform data
        - shot (int)                    : shot number 
        - trace (int)                   : trace number
        - color (str)                   : the color of lines
        - tick_param (dict,optional)    : the fontdict for ticks
        - label_param (dict,optional)   : the fontdict for label
        - title_param (dict,optional)   : the fontdict for title
        - figsize (tuple,optional)      : The size of the figure
        - show (bool,optional)          : showing the figure or not. Default True
        - save_path (str,optional)      : the saving path for the figure. Defualt:""
        - save_dpi (int,optional)       : the saving resolution for the figure. Default:300 dpi
    """
    data = gpu2cpu(data)
    
    plt.figure(figsize=figsize)
    if dt is not None:
        plt.plot(np.arange(data.shape[1])*dt,data[shot,:,trace],c=color)
        plt.xlabel("Times (s)",**label_param)
    else:
        plt.plot(data[shot,:,trace],c=color)
        plt.xlabel("Time Samples",**label_param)
    
    plt.tick_params(**tick_param)
    plt.title(f"shot:{shot} trace:{trace}",**title_param)
    
    plt.ylabel("Amplitude",**label_param)
    if not save_path == "":
        plt.savefig(save_path,dpi=save_dpi,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    

########################################################
#                   waveform 2D
########################################################
def norm_traces(data):
    """ Normalize a seismic data to its maximum amplitude trace by trace
    
    Args:
        data (ndarray): The seismic data to be normalized.
        
    Returns:
        ndarray: The normalized seismic data.
    """

    eps = 1e-20

    nr = data.shape[0]
    nt = data.shape[1]


    for i in range(nr):
        data[i,:] = data[i,:]/(abs(data[i,:]) + eps).max()

    return data

def plot_waveform2D(data, dt=None, dx=None, cmap='coolwarm', aspect='auto', clip=99.9, 
                figsize=(10,6), colorbar=False,
                type='section', norm=False, wiggle_scale=1, wiggle_interval=1,
                plot_fk=False, vel=None, fmin=1, fmax=20, kmin=-0.05, kmax=0.05,
                save_path="",
                show=False
                ):
    """ Plot two data
    
    Plot a data section using matplotlib.imshow.
    
    Parameters:
        - data (ndarray or Tensor): The data to be plotted.
        - dt (float, optional): The time sampling interval. If not provided, the x-axis will be labeled with sample numbers.
        - dx (float, optional): The spatial sampling interval. If not provided, the y-axis will be labeled with trace numbers.
        - cmap (str, optional): The colormap to be used. Default is 'gray'.
        - aspect (str, optional): The aspect ratio of the plot. Default is 'auto'.
        - clip (float, optional): The percentile value for clipping the data. Default is 99.9.
        - figsize (tuple, optional): The size of the figure. Default is (8, 8).
        - colorbar (bool, optional): Whether to show the colorbar. Default is False.
        - norm (bool, optional): Whether to normalize the data. Default is False.
        - savefig (str, optional): The file path to save the figure. If not provided, the figure will be displayed.

    Returns:
    None
    """

    plt.figure(figsize = figsize)
    
    if not isinstance(data, np.ndarray):
        try:
            data = data.cpu().detach().numpy()
        except AttributeError:
            # Handle the case where data cannot be converted to a NumPy array
            pass    
    if dt is None:
        t = np.arange(data.shape[1])
    else:
        t = np.arange(data.shape[1]) * dt
    if dx is None:
        x = np.arange(data.shape[0])
    else:
        x = np.arange(data.shape[0]) * dx

    extent = [x[0], x[-1], t[-1], t[0]]

    if plot_fk:
        plt.subplot(1,2,1)
    else:
        plt.subplot(1,1,1)

    if norm:
        data = norm_traces(data)

    vmax = np.percentile(data, clip)
    if type == 'section':
        plt.imshow(data.T, aspect=aspect, cmap=cmap, vmin=-vmax, vmax=vmax, extent = extent)
        plt.ylim([t[-1], t[0]])
        plt.xlim([x[0], x[-1]])
    
    elif type == 'wiggle':
        for i, trace in enumerate(data):
            if i % wiggle_interval != 0:
                continue
            trace = trace * wiggle_scale + i
            plt.plot(trace, t, color='black', linewidth=1.0)
            plt.fill_betweenx(t, i, trace, where=(trace > i), color='black')
        plt.ylim([t[-1], t[0]])
        plt.xlim([0-1, data.shape[0]+1])
    else:
        raise ValueError('type must be either "section" or "wiggle"')

    # lim
    if dx is not None:
        plt.xlabel('Offset (m)')
    else:
        plt.xlabel('Trace #')

    if dt is not None:
        plt.ylabel('Time (s)')
    else:
        plt.ylabel('Sample #')
    if colorbar:
        plt.colorbar()
    plt.grid(axis='y', alpha=0.8)
    plt.tight_layout()
    
    if not save_path == "":
        plt.savefig(save_path, dpi=300,bbox_inches="tight")
    
    if not show:
        plt.close()
    else:
        plt.show()

########################################################
#                waveform wiggle 2D
########################################################
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


def plot_waveform_wiggle(data, tt=None, xx=None, color='k', sf=0.15, verbose=False,save_path="",show=False):
    '''Wiggle plot of a sesimic data section
    Syntax examples:
        wiggle(data)
        wiggle(data, tt)
        wiggle(data, tt, xx)
        wiggle(data, tt, xx, color)
        fi = wiggle(data, tt, xx, color, sf, verbose)
    Parameters
    ----------------------
        - data (ndarray or Tensors)         : 2D waveform data
        - tt (ndarray or Tensors,optional)  : time list of the waveforms
        - xx (ndarray or Tensors,optional)  : offset list of the waveforms
        - color (str,optional)              : color of the waveform
        - sf (floats,optional)              : streth factor
        - verbose (bool,optional)           : show the ouputs or not
        - save_path (str,optional)          : save path for the figure
        - show (bool,optional)              : show the figure or not

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

    returns:
    --------------------
    None
    '''
    data = gpu2cpu(data)
    if tt is not None:
        tt = gpu2cpu(tt)
    if xx is not None:
        xx = gpu2cpu(xx)
    # Input check
    data, tt_new, xx_new, ts_new = wiggle_input_check(data, tt, xx, sf, verbose)

    # Plot data using matplotlib.pyplot
    Ntr = data.shape[1]

    ax = plt.gca()
    for ntr in range(Ntr):
        trace = data[:, ntr]
        offset = xx_new[ntr]

        if verbose:
            print(offset)

        trace_zi, tt_zi = insert_zeros(trace, tt_new)
        ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                        where=trace_zi >= 0,
                        facecolor=color)
        ax.plot(trace_zi + offset, tt_zi, color)

    ax.set_xlim(xx_new[0] - ts_new, xx_new[-1] + ts_new)
    ax.set_ylim(tt_new[0], tt_new[-1])
    ax.invert_yaxis()
    
    if xx is not None:
        plt.xlabel('Offset (m)')
    else:
        plt.xlabel('Trace #')

    if tt is not None:
        plt.ylabel('Time (s)')
    else:
        plt.ylabel('Sample #')
    
    if save_path != "":
        plt.savefig(save_path,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()