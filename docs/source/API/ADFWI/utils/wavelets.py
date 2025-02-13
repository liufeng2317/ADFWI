import numpy as np

# source wavelet
def wavelet(nt, dt, f0, amp0 = 1, t0 = None, type = 'Ricker'):
    """
    Generate a source time function (wavelet).

    Parameters
    ----------
    nt : int
        The number of time steps.
    dt : float
        The time step size.
    f0 : float
        The central frequency of the wavelet.
    amp0 : float, optional
        The amplitude of the wavelet (default is 1).
    t0 : float, optional
        The center time of the wavelet. If None, defaults to `1.2 / f0` (default is None).
    type : {'Ricker', 'Gaussian', 'Ramp'}, optional
        The type of the wavelet to generate. Options are 'Ricker', 'Gaussian', or 'Ramp' (default is 'Ricker').

    Returns
    -------
    t : numpy.ndarray
        The time array.
    wavelet : numpy.ndarray
        The generated wavelet.
    
    Raises
    ------
    ValueError
        If an unsupported wavelet type is specified.
    """
    # time array
    t = np.arange(nt) * dt + 0.0
    wavelet = np.zeros_like(t)
    t0 = t0 if t0 is not None else 1.2 / f0

    # Ricker wavelet
    if type.lower() in ['ricker']:
        tau = (np.pi*f0) ** 2
        wavelet = amp0 * (1 - 2 * tau * (t - t0) ** 2) * np.exp(- tau * (t - t0) ** 2)
    
    # Gaussian wavelet
    elif type.lower() in ['gaussian']:
        tau = (np.pi*f0) ** 2
        wavelet = amp0 * (1 - 2 * tau * (t - t0) ** 2) * np.exp(- tau * (t - t0) ** 2)

        # perform integration twice to get the Gaussian wavelet
        wavelet = np.cumsum(wavelet)
        wavelet = np.cumsum(wavelet)

    # Ramp wavelet
    elif type.lower() in ['ramp']:
        wavelet = amp0 * 0.5 * (1. + np.tanh(t / t0))
        
    # Unknown source type
    else:
        msg = 'Support source types: Rikcer, Guassian, Ramp. \n'
        err = 'Unknown source type: {}'.format(type)
        raise ValueError(msg + '\n' + err)

    return t,wavelet