# import numpy as np

# np.random.seed(1234)
# def add_gaussian_noise(data, std_noise, mean_factor=0.1):
#     """
#     Add Gaussian noise with a specified mean (based on each trace) and standard deviation to each trace in the data.

#     Parameters:
#     data (np.ndarray): Input data with dimensions [shot, time, trace].
#     std_noise (float): Standard deviation of the Gaussian noise, controls the noise intensity.
#     mean_factor (float): Factor to scale the mean of the Gaussian noise, controls the noise bias.

#     Returns:
#     np.ndarray: Data with added Gaussian noise.
#     """
#     noise_data = np.zeros_like(data)
#     for ishot in range(data.shape[0]):
#         for itrace in range(data.shape[2]):
#             # Calculate mean_noise based on the mean of the current trace
#             trace_mean = np.mean(data[ishot, :, itrace])
#             mean_noise = trace_mean * mean_factor
            
#             # Add noise independently to each trace with its own mean and standard deviation
#             noise_data[ishot, :, itrace] = data[ishot, :, itrace] + \
#                 np.random.normal(mean_noise, np.std(data[ishot, :, itrace]) * std_noise, data[ishot, :, itrace].shape)
#     return noise_data

import numpy as np

def add_gaussian_noise(data, std_noise, mean_bias_factor=0.1, seed=1234):
    """
    Add Gaussian noise with a specified standard deviation and an optional mean bias to each trace in the data.

    Parameters:
    data (np.ndarray): Input data with dimensions [shot, time, trace].
    std_noise (float): Standard deviation of the Gaussian noise, controls the noise intensity.
    mean_bias_factor (float): Factor to scale the mean of each trace to serve as the mean of the Gaussian noise. 
                              Default is 0.1, meaning the noise mean is 10% of the trace mean.
    seed (int, optional): Random seed for reproducibility. If None, the noise will be different each time.

    Returns:
    np.ndarray: Data with added Gaussian noise.
    """

    # Optionally set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Calculate the mean of each trace [shot, 1, trace] to apply mean bias
    trace_means = np.mean(data, axis=1, keepdims=True)  # Shape: [shot, 1, trace]

    # Calculate the mean noise based on the trace means and the provided mean_bias_factor
    mean_noises = trace_means * mean_bias_factor  # Shape: [shot, 1, trace]
    
    # Generate Gaussian noise with the specified mean bias and standard deviation
    noise = np.random.normal(loc=mean_noises, scale=std_noise, size=data.shape)

    # Add the generated noise to the original data
    noisy_data = data + noise

    return noisy_data
