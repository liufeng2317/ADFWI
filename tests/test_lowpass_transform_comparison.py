import unittest

import torch

from ADFWI.fwi.multiScaleProcessing import lpass
from ADFWI.fwi.transforms import LowPassFilter


class LowPassTransformComparisonTests(unittest.TestCase):
    def _mixed_signal(self, nt=256, dt=0.001):
        time = torch.arange(nt, dtype=torch.float32) * dt
        low = torch.sin(2 * torch.pi * 20.0 * time)
        high = 0.5 * torch.sin(2 * torch.pi * 180.0 * time)
        signal = (low + high).reshape(1, nt, 1)
        return signal, signal.clone()

    def _high_band_ratio(self, before, after, dt, threshold=120.0):
        freqs = torch.fft.rfftfreq(before.shape[1], d=dt)
        high_band = freqs > threshold
        before_amp = torch.fft.rfft(before, dim=1).abs()[:, high_band, :].mean()
        after_amp = torch.fft.rfft(after, dim=1).abs()[:, high_band, :].mean()
        return float((after_amp / before_amp).item())

    def test_torch_lowpass_matches_legacy_lpass_in_trace_interior(self):
        dt = 0.001
        cutoff = 60.0
        synthetic, observed = self._mixed_signal(dt=dt)

        legacy_syn, legacy_obs = lpass(synthetic, observed, cutoff, int(1 / dt))
        torch_syn, torch_obs = LowPassFilter(cutoff_freq=cutoff, dt=dt, filter_length=101)(synthetic, observed)

        interior = slice(60, -60)
        legacy_flat = legacy_syn[:, interior, :].reshape(-1)
        torch_flat = torch_syn[:, interior, :].reshape(-1)
        correlation = torch.corrcoef(torch.stack([legacy_flat, torch_flat]))[0, 1]
        mean_abs_error = (legacy_flat - torch_flat).abs().mean()

        self.assertGreater(float(correlation.item()), 0.999)
        self.assertLess(float(mean_abs_error.item()), 1e-3)
        self.assertTrue(torch.allclose(torch_syn, torch_obs))
        self.assertTrue(torch.allclose(legacy_syn, legacy_obs))

    def test_torch_lowpass_attenuates_high_band_at_least_as_well_as_legacy(self):
        dt = 0.001
        cutoff = 60.0
        synthetic, observed = self._mixed_signal(dt=dt)

        legacy_syn, _ = lpass(synthetic, observed, cutoff, int(1 / dt))
        torch_syn, _ = LowPassFilter(cutoff_freq=cutoff, dt=dt, filter_length=101)(synthetic, observed)

        legacy_ratio = self._high_band_ratio(synthetic, legacy_syn, dt)
        torch_ratio = self._high_band_ratio(synthetic, torch_syn, dt)

        self.assertLess(torch_ratio, legacy_ratio)
        self.assertLess(torch_ratio, 0.1)

    def test_torch_lowpass_keeps_gradients_differentiable(self):
        synthetic, observed = self._mixed_signal()
        synthetic.requires_grad_(True)
        observed.requires_grad_(True)

        torch_syn, torch_obs = LowPassFilter(cutoff_freq=60.0, dt=0.001, filter_length=101)(synthetic, observed)
        loss = torch_syn.square().mean() + torch_obs.square().mean()
        loss.backward()

        self.assertIsNotNone(synthetic.grad)
        self.assertIsNotNone(observed.grad)
        self.assertTrue(torch.isfinite(synthetic.grad).all())
        self.assertTrue(torch.isfinite(observed.grad).all())


if __name__ == "__main__":
    unittest.main()
