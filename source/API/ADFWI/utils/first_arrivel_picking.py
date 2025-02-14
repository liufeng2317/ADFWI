import numpy as np
from ADFWI.utils.utils import numpy2tensor


def brutal_picker(trace):
    """
    Picks the first arrival time of seismic traces using a simple thresholding method.

    Parameters
    ----------
    trace : numpy.ndarray
        A 2D array of seismic traces with shape (n_traces, n_samples), where each row 
        represents a single seismic trace.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the first arrival index for each trace.
    """
    threds = 0.001 * np.max(abs(trace), axis=-1)  # Compute threshold for each trace
    pick = [(abs(trace[i, :]) > threds[i]).argmax(axis=-1) for i in range(trace.shape[0])]
    return np.array(pick)


def mask(itmin, itmax, nt, length):
    """
    Constructs a tapered mask that can be applied to a seismic trace to mute early or late arrivals.

    Parameters
    ----------
    itmin : int
        The starting index for tapering.
    itmax : int
        The ending index for tapering.
    nt : int
        The total number of time samples in the trace.
    length : int
        The length of the tapering window.

    Returns
    -------
    numpy.ndarray
        A 1D mask array of shape (nt,) with tapered values applied 
        to the specified range.
    """
    mask = np.ones(nt)  # Initialize mask with ones

    # Construct tapering window
    win = np.sin(np.linspace(0, np.pi, 2 * length))  # Create a sine window
    win = win[0:length]  # Use only the first half for tapering

    if 1 < itmin < itmax < nt:
        mask[0:itmin] = 0.  # Zero out values before itmin
        mask[itmin:itmax] = win * mask[itmin:itmax]  # Apply taper
    elif itmin < 1 <= itmax:
        mask[0:itmax] = win[length - itmax:length] * mask[0:itmax]  # Handle case where itmin is out of range
    elif itmin < nt < itmax:
        mask[0:itmin] = 0.  # Zero out values before itmin
        mask[itmin:nt] = win[0:nt - itmin] * mask[itmin:nt]  # Apply taper for valid range
    elif itmin > nt:
        mask[:] = 0.  # If itmin is beyond nt, set entire mask to zero

    return mask

def mute_arrival(trace, itmin, itmax, mutetype, nt, length):
    """
    Applies a tapered mask to a seismic record section, muting early or late arrivals.

    Parameters
    ----------
    trace : torch.Tensor
        A 1D or 2D tensor representing the seismic trace or record section.
    itmin : int
        The starting index for muting.
    itmax : int
        The ending index for muting.
    mutetype : str
        Unused parameter, potentially for future expansion.
    nt : int
        The total number of time samples in the trace.
    length : int
        The length of the tapering window.

    Returns
    -------
    torch.Tensor
        The muted seismic trace with the applied mask.
    """
    win = 1 - mask(itmin, itmax, nt, length)  # Compute inverse mask
    win = numpy2tensor(win, dtype=trace.dtype, device=trace.device)  # Convert to tensor
    trace = trace * win  # Apply mask
    return trace
    
def apply_mute(mute_late_window, shot, dt):
    """
    Applies a time window mute based on the first arrival pick for each trace.

    Parameters
    ----------
    mute_late_window : float
        The time window for late arrival muting.
    shot : torch.Tensor
        A 2D tensor of shape (nt, nrcv), representing the seismic shot gather.
    dt : float
        The time sampling interval.

    Returns
    -------
    torch.Tensor
        The muted shot gather with applied time window muting.
    """
    shot_np = shot.cpu().detach().numpy()  # Convert tensor to NumPy array
    pick = brutal_picker(shot_np.T) + np.ceil(mute_late_window / dt)  # Compute mute start times per trace

    # Initialize variables
    shot_new = shot.clone()  # Clone input shot gather
    length = 100  # Taper length
    nt = shot_np.shape[0]  # Number of time samples

    # Apply mute for each trace
    for i in range(shot.shape[-1]):
        itmin = int(pick[i] - length / 2)  # Compute mute start index
        itmax = int(itmin + length)  # Compute mute end index
        shot_new[:, i] = mute_arrival(shot[:, i], itmin, itmax, 'late', nt, length)  # Apply mute

    return shot_new