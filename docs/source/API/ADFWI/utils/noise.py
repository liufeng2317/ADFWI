import numpy as np

def add_gaussian_noise(data, std_noise, mean_bias_factor=0.1, seed=1234):
    """
    Add Gaussian noise to seismic data with a specified standard deviation and optional mean bias.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (shot, time, trace).
    std_noise : float
        Standard deviation of the Gaussian noise, controlling the noise intensity.
    mean_bias_factor : float, optional
        Factor to scale the mean of each trace to serve as the mean of the Gaussian noise. Default is 0.1, meaning the noise mean is 10% of the trace mean.
    seed : int or None, optional
        Random seed for reproducibility. If None, the noise will be different on each execution. Default is 1234.

    Returns
    -------
    noisy_data : np.ndarray
        Noisy data with the same shape as `data` (shot, time, trace).
    """
    
    # Optionally set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Calculate the mean of each trace [shot, 1, trace] to apply mean bias
    trace_means = np.mean(data, axis=1, keepdims=True)  # Shape: (shot, 1, trace)

    # Calculate the mean noise based on the trace means and the provided mean_bias_factor
    mean_noises = trace_means * mean_bias_factor  # Shape: (shot, 1, trace)
    
    # Generate Gaussian noise with the specified mean bias and standard deviation
    noise = np.random.normal(loc=mean_noises, scale=std_noise, size=data.shape)

    # Add the generated noise to the original data
    noisy_data = data + noise

    return noisy_data