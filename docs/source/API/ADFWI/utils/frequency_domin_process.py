import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def calculate_spectrum(rcv, dt):
    """
    Perform frequency spectrum analysis of seismic data and return the spectrum data.

    Parameters
    ----------
    rcv : np.ndarray
        Seismic data with shape (n_receivers, n_samples).
    dt : float
        Time sampling interval in seconds.

    Returns
    -------
    freqs : np.ndarray
        Array of positive frequencies with shape (n_samples // 2,).
    amplitude_spectrum : np.ndarray
        Amplitude spectrum of the seismic data with shape (n_receivers, n_samples // 2).
    power_spectrum : np.ndarray
        Power spectrum (amplitude squared) of the seismic data with shape (n_receivers, n_samples // 2).

    Notes
    -----
    - The function applies FFT along the last axis (time axis) of `rcv`.
    - Only the positive frequency components are returned.
    """
    # 1. Perform Fast Fourier Transform (FFT) on each receiver's data
    # rcv.shape = (n_receivers, n_samples)
    n_receivers, n_samples = rcv.shape
    fs = 1 / dt  # Sampling frequency
    
    # Apply FFT to each trace (receiver data)
    fft_data = np.fft.fft(rcv, axis=1)  # FFT along the time axis (axis=1)
    
    # Frequency axis: Frequency bins corresponding to FFT result
    freqs = np.fft.fftfreq(n_samples, dt)  # Frequency bins
    
    # Only keep positive frequencies (real-valued spectrum)
    positive_freqs = freqs[:n_samples // 2]
    fft_data = fft_data[:, :n_samples // 2]
    
    # Compute amplitude spectrum
    amplitude_spectrum = np.abs(fft_data)
    
    # Compute power spectrum (optional)
    power_spectrum = amplitude_spectrum**2
    
    return positive_freqs, amplitude_spectrum, power_spectrum


def filter_low_frequencies_zero_phase(data, dt, cutoff_freq=5):
    """
    Apply zero-phase low-pass filtering to retain frequencies below the cutoff frequency.

    Parameters
    ----------
    data : np.ndarray
        Seismic data with shape (shot, t, rcv), where:
    dt : float
        Time step (sampling interval) in seconds.
    cutoff_freq : float, optional
        The cutoff frequency in Hz. Frequencies above this value will be attenuated. Default is 5 Hz.

    Returns
    -------
    filtered_data : np.ndarray
        Zero-phase low-pass filtered seismic data with the same shape as `data` (shot, t, rcv).

    Notes
    -----
    - This function applies a **6th-order Butterworth low-pass filter**.
    - Filtering is done **trace-by-trace** using `scipy.signal.filtfilt` to ensure zero-phase distortion.
    - The cutoff frequency is normalized by the **Nyquist frequency** (`0.5 / dt`).
    """
    n_shots, n_time, n_receivers = data.shape

    # Define the Nyquist frequency
    nyquist_freq = 0.5 / dt
    normalized_cutoff = cutoff_freq / nyquist_freq  # Normalize the cutoff frequency

    # Design a Butterworth low-pass filter
    b, a = signal.butter(6, normalized_cutoff, btype='low')

    # Apply zero-phase filtering to each trace using filtfilt
    filtered_data = np.zeros_like(data)
    for i in range(n_shots):
        for j in range(n_receivers):
            filtered_data[i, :, j] = signal.filtfilt(b, a, data[i, :, j])

    return filtered_data

def plot_spectrum(positive_freqs, amplitude_spectrum, power_spectrum,
                  y1lim = [],y2lim=[],
                  save_path="",show=True):
    """
    Plot the frequency spectrum (amplitude and power) of seismic data.
    """
    # 2. Plot the frequency spectrum for one receiver (for example)
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(positive_freqs, amplitude_spectrum[0], label="Receiver 1 Amplitude Spectrum", color='blue')
    plt.title("Frequency Spectrum of Seismic Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xscale('log')
    plt.yscale('log')
    if len(y1lim) == 2:
        plt.ylim((y1lim[0],y1lim[1]))
    plt.grid(True)
    plt.legend()
    
    # 3. Plot the power spectrum (optional)
    plt.subplot(122)
    plt.plot(positive_freqs, power_spectrum[0], label="Receiver 1 Power Spectrum", color='red')
    plt.title("Power Spectrum of Seismic Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xscale('log')
    plt.yscale('log')
    if len(y2lim) == 2:
        plt.ylim((y2lim[0],y2lim[1]))
    plt.grid(True)
    plt.legend()
    if not save_path == "":
        plt.savefig(save_path,bbox_inches="tight")
    if show:
        plt.show()

def plot_frequency_distribution(positive_freqs, amplitude_spectrum,
                                xlim = [],
                                save_path="",show=True):
    """
    Plot the frequency distribution of seismic data across all receivers.
    """
    # Sum the amplitude spectrum across all receivers to get the total distribution
    total_amplitude = np.sum(amplitude_spectrum, axis=0)
    
    # Plot the frequency distribution
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, total_amplitude, label="Total Amplitude Spectrum", color='green')
    plt.title("Frequency Distribution of Seismic Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Total Amplitude")
    plt.grid(True)
    plt.legend()
    if len(xlim) == 2:
        plt.xlim((xlim[0],xlim[1]))
    if not save_path == "":
        plt.savefig(save_path,bbox_inches="tight")
    if show:
        plt.show()

def plot_filtered_data(original_data, filtered_data, shot_idx=0,cutoff_freq=5,save_path="",show=True):
    """
    Plot original and filtered seismic data for a specific shot.
    """
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_data[shot_idx], label='Original Data',aspect='auto',cmap='gray')
    plt.title(f"Original",fontsize=13)
    plt.ylabel("Time (samples)",fontsize=13)
    plt.xlabel("Receiver ID",fontsize=13)
    plt.grid(True,linestyle='--',color="silver",alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_data[shot_idx], label='Filtered Data', aspect='auto',cmap='gray')
    plt.title(f"Filtered (below {cutoff_freq}Hz removed)",fontsize=13)
    plt.ylabel("Time (samples)",fontsize=13)
    plt.xlabel("Receiver ID",fontsize=13)
    plt.grid(True,linestyle='--',color="silver",alpha=0.5)
    plt.tight_layout()
    
    if not save_path == "":
        plt.savefig(save_path,bbox_inches="tight")
    if show:
        plt.show()