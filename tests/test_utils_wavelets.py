import unittest

import numpy as np

from ADFWI.utils import wavelet


class UtilsWaveletTests(unittest.TestCase):
    def test_ricker_wavelet_matches_formula_with_default_t0(self):
        nt = 8
        dt = 0.004
        f0 = 5.0
        amp0 = 2.0

        time, values = wavelet(nt, dt, f0, amp0=amp0, type="Ricker")

        expected_time = np.arange(nt) * dt
        tau = (np.pi * f0) ** 2
        t0 = 1.2 / f0
        expected = amp0 * (1 - 2 * tau * (expected_time - t0) ** 2) * np.exp(
            -tau * (expected_time - t0) ** 2
        )
        np.testing.assert_allclose(time, expected_time)
        np.testing.assert_allclose(values, expected)
        self.assertEqual(values.shape, (nt,))

    def test_wavelet_accepts_case_insensitive_type_and_explicit_t0(self):
        nt = 5
        dt = 0.01
        f0 = 3.0
        t0 = 0.02

        _, lower = wavelet(nt, dt, f0, t0=t0, type="ricker")
        _, mixed = wavelet(nt, dt, f0, t0=t0, type="RiCkEr")

        np.testing.assert_allclose(lower, mixed)

    def test_gaussian_wavelet_is_double_cumsum_of_ricker_formula(self):
        nt = 8
        dt = 0.004
        f0 = 5.0
        amp0 = 1.5
        t0 = 0.02

        time, values = wavelet(nt, dt, f0, amp0=amp0, t0=t0, type="Gaussian")

        tau = (np.pi * f0) ** 2
        base = amp0 * (1 - 2 * tau * (time - t0) ** 2) * np.exp(-tau * (time - t0) ** 2)
        expected = np.cumsum(np.cumsum(base))
        np.testing.assert_allclose(values, expected)

    def test_ramp_wavelet_matches_current_tanh_formula(self):
        nt = 6
        dt = 0.01
        f0 = 2.0
        amp0 = 3.0
        t0 = 0.05

        time, values = wavelet(nt, dt, f0, amp0=amp0, t0=t0, type="Ramp")

        expected = amp0 * 0.5 * (1.0 + np.tanh(time / t0))
        np.testing.assert_allclose(values, expected)

    def test_unknown_wavelet_type_reports_requested_type(self):
        with self.assertRaisesRegex(ValueError, "Unknown source type: cosine"):
            wavelet(4, 0.001, 10.0, type="cosine")


if __name__ == "__main__":
    unittest.main()
