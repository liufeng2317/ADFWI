import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ADFWI.utils.frequency_domin_process import (  # noqa: E402
    calculate_spectrum,
    filter_low_frequencies_zero_phase,
    plot_filtered_data,
    plot_frequency_distribution,
    plot_spectrum,
)


class UtilsFrequencyProcessTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_calculate_spectrum_matches_numpy_fft_positive_half(self):
        dt = 0.125
        n_samples = 8
        time = np.arange(n_samples) * dt
        receiver_data = np.vstack(
            [
                np.sin(2 * np.pi * 2.0 * time),
                np.cos(2 * np.pi * 1.0 * time),
            ]
        )

        freqs, amplitude, power = calculate_spectrum(receiver_data, dt)

        expected_fft = np.fft.fft(receiver_data, axis=1)[:, : n_samples // 2]
        np.testing.assert_allclose(freqs, np.fft.fftfreq(n_samples, dt)[: n_samples // 2])
        np.testing.assert_allclose(amplitude, np.abs(expected_fft))
        np.testing.assert_allclose(power, np.abs(expected_fft) ** 2)
        self.assertEqual(amplitude.shape, (2, n_samples // 2))
        self.assertEqual(power.shape, (2, n_samples // 2))

    def test_filter_low_frequencies_zero_phase_highpass_contract(self):
        dt = 0.001
        n_samples = 2000
        time = np.arange(n_samples) * dt
        low_frequency = np.sin(2 * np.pi * 2.0 * time)
        high_frequency = 0.5 * np.sin(2 * np.pi * 40.0 * time)
        data = (low_frequency + high_frequency).reshape(1, n_samples, 1)

        filtered = filter_low_frequencies_zero_phase(data, dt, cutoff_freq=5)

        self.assertEqual(filtered.shape, data.shape)
        self.assertEqual(filtered.dtype, data.dtype)
        self.assertTrue(np.isfinite(filtered).all())

        freqs = np.fft.rfftfreq(n_samples, dt)
        before = np.abs(np.fft.rfft(data[0, :, 0]))
        after = np.abs(np.fft.rfft(filtered[0, :, 0]))
        low_idx = np.argmin(np.abs(freqs - 2.0))
        high_idx = np.argmin(np.abs(freqs - 40.0))
        self.assertLess(after[low_idx] / before[low_idx], 0.01)
        self.assertGreater(after[high_idx] / before[high_idx], 0.95)

    def test_plot_helpers_save_figures_with_show_false(self):
        positive_freqs = np.arange(1.0, 5.0)
        amplitude = np.vstack([positive_freqs, positive_freqs * 2.0])
        power = amplitude**2
        original = np.arange(24, dtype=np.float64).reshape(1, 6, 4)
        filtered = original * 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            spectrum_path = tmpdir / "spectrum.png"
            distribution_path = tmpdir / "distribution.png"
            filtered_path = tmpdir / "filtered.png"

            plot_spectrum(
                positive_freqs,
                amplitude,
                power,
                y1lim=[0.5, 10.0],
                y2lim=[0.5, 100.0],
                save_path=str(spectrum_path),
                show=False,
            )
            plot_frequency_distribution(
                positive_freqs,
                amplitude,
                xlim=[1.0, 4.0],
                save_path=str(distribution_path),
                show=False,
            )
            plot_filtered_data(
                original,
                filtered,
                shot_idx=0,
                cutoff_freq=5,
                save_path=str(filtered_path),
                show=False,
            )

            self.assertTrue(spectrum_path.exists())
            self.assertTrue(distribution_path.exists())
            self.assertTrue(filtered_path.exists())


if __name__ == "__main__":
    unittest.main()
