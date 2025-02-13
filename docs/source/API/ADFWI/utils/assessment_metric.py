import numpy as np
from skimage.metrics import structural_similarity as ssim

def MSE(true_v, inv_v):
    """
    Computes the Mean Squared Error (MSE) between the true velocity model and the inverted velocity model.

    Parameters
    ----------
    true_v : numpy.ndarray
        The ground truth velocity model of shape (nz, nx).
    inv_v : numpy.ndarray
        The inverted velocity model of shape (nz, nx).

    Returns
    -------
    float
        The sum of squared differences between the true and inverted velocity models.
    """
    nz, nx = true_v.shape  # Get the dimensions of the velocity models
    return np.sum((true_v - inv_v) ** 2)  # Compute the sum of squared differences


def RMSE(true_v,inv_v):
    """
    Computes the Root Mean Squared Error (RMSE) between the true velocity model and the inverted velocity model.

    Parameters
    ----------
    true_v : numpy.ndarray
        The ground truth velocity model. Shape can be (nz, nx) or (batch, nz, nx).
    inv_v : numpy.ndarray
        The inverted velocity model. Shape should match `true_v`.

    Returns
    -------
    numpy.ndarray or float
        If `true_v` has one less dimension than `inv_v`, returns an array of RMSE values  for each sample in the batch. Otherwise, returns a single RMSE value.
    """
    if len(true_v.shape) != len(inv_v.shape):
        true_v = true_v[np.newaxis,:,:]
        res = np.sqrt(np.sum((true_v-inv_v)**2,axis=(1,2)))
        return res
    else:
        return np.sqrt(np.sum((true_v-inv_v)**2))

def MAPE(true_v, inv_v):
    """
    Computes the Mean Absolute Percentage Error (MAPE) between the true velocity model and the inverted velocity model.

    Parameters
    ----------
    true_v : numpy.ndarray
        The ground truth velocity model. Shape can be (nz, nx) or (batch, nz, nx).
    inv_v : numpy.ndarray
        The inverted velocity model. Shape should match `true_v`.

    Returns
    -------
    numpy.ndarray or float
        If `true_v` has one less dimension than `inv_v`, returns an array of MAPE values for each sample in the batch. Otherwise, returns a single MAPE value.
    """
    if len(true_v.shape) != len(inv_v.shape):
        nz, nx = true_v.shape  # Get spatial dimensions
        true_v = true_v[np.newaxis, :, :]  # Add batch dimension if missing
        res = 100 / (nx * nz) * np.sum(np.abs(inv_v - true_v) / true_v, axis=(1, 2))  # Compute MAPE per batch
    else:
        nz, nx = true_v.shape[-2:]  # Get spatial dimensions
        res = 100 / (nx * nz) * np.sum(np.abs(inv_v - true_v) / true_v)  # Compute single MAPE value
    
    return res

def SSIM(true_v, inv_v, win_size=11):
    """
    Computes the Structural Similarity Index (SSIM) between the true velocity model and the inverted velocity model.

    Parameters
    ----------
    true_v : numpy.ndarray
        The ground truth velocity model. Shape can be (nz, nx) or (batch, nz, nx).
    inv_v : numpy.ndarray
        The inverted velocity model. Shape should match `true_v`.
    win_size : int, optional
        The size of the sliding window used for computing SSIM, by default 11.

    Returns
    -------
    list or float
        If `true_v` has one less dimension than `inv_v`, returns a list of SSIM values for each sample in the batch. Otherwise, returns a single SSIM value.
    """
    if len(true_v.shape) != len(inv_v.shape):
        ssim_res = []
        for i in range(inv_v.shape[0]):
            vmax = np.max([true_v.max(), inv_v[i].max()])  # Compute max value for data range
            vmin = np.min([true_v.min(), inv_v[i].min()])  # Compute min value for data range
            temp_ssim = ssim(true_v, inv_v[i], data_range=vmax - vmin, win_size=win_size)
            ssim_res.append(temp_ssim)
        return ssim_res
    else:
        vmax = np.max([true_v.max(), inv_v.max()])
        vmin = np.min([true_v.min(), inv_v.min()])
        return ssim(true_v, inv_v, data_range=vmax - vmin, win_size=win_size)

def SNR(true_v, inv_v):
    """
    Computes the Signal-to-Noise Ratio (SNR) between the true velocity model and the inverted velocity model.

    Parameters
    ----------
    true_v : numpy.ndarray
        The ground truth velocity model. Shape can be (nz, nx) or (batch, nz, nx).
    inv_v : numpy.ndarray
        The inverted velocity model. Shape should match `true_v`.

    Returns
    -------
    list or float
        If `true_v` has one less dimension than `inv_v`, returns a list of SNR values for each sample in the batch. Otherwise, returns a single SNR value.
    """
    if len(true_v.shape) != len(inv_v.shape):
        snr_res = []
        for i in range(inv_v.shape[0]):
            true_norm = np.sum(true_v ** 2)  # Compute signal power
            diff_norm = np.sum((true_v - inv_v[i]) ** 2)  # Compute noise power
            temp_snr = 10 * np.log10(true_norm / diff_norm)  # Compute SNR in dB
            snr_res.append(temp_snr)
        return snr_res
    else:
        true_norm = np.sum(true_v ** 2)
        diff_norm = np.sum((true_v - inv_v) ** 2)
        temp_snr = 10 * np.log10(true_norm / diff_norm)
        return temp_snr
