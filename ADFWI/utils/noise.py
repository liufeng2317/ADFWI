"""Synthetic seismic noise helpers."""

import numpy as np


def add_gaussian_noise(data, std_noise, mean_bias_factor=0.1, seed=1234):
    """Add Gaussian noise to seismic data with optional trace-mean bias.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape ``[shot, time, trace]``.
    std_noise : float
        Standard deviation passed directly to ``np.random.normal``.
    mean_bias_factor : float, optional
        Scale factor applied to each trace mean, computed over the time axis
        with shape ``[shot, 1, trace]``.
    seed : int or None, optional
        Random seed for reproducibility. ``None`` leaves the global NumPy random
        state untouched.

    Returns
    -------
    np.ndarray
        Data plus generated noise, preserving the input shape.
    """

    if seed is not None:
        np.random.seed(seed)

    trace_means = np.mean(data, axis=1, keepdims=True)
    mean_noises = trace_means * mean_bias_factor
    noise = np.random.normal(loc=mean_noises, scale=std_noise, size=data.shape)
    noisy_data = data + noise

    return noisy_data
