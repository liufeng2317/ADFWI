import numpy as np


def wavelet(nt, dt, f0, amp0 = 1, t0 = None, type = 'Ricker'):
    """Return a source time axis and wavelet values.

    Parameters
    ----------
    nt : int
        Number of time samples.
    dt : float
        Time sampling interval in seconds.
    f0 : float
        Dominant frequency in Hz.
    amp0 : float, optional
        Amplitude scale.
    t0 : float, optional
        Source delay. The historical default is ``1.2 / f0``.
    type : str, optional
        Historical public argument selecting ``"Ricker"``, ``"Gaussian"``, or
        ``"Ramp"``. ``"Gaussian"`` preserves the legacy double cumulative sum
        of the Ricker-like base expression.
    """
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
        msg = 'Supported source types: Ricker, Gaussian, Ramp. \n'
        err = 'Unknown source type: {}'.format(type)
        raise ValueError(msg + '\n' + err)

    return t,wavelet
