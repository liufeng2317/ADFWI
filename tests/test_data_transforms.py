import unittest

import numpy as np
import torch

from ADFWI.fwi.normalization import normalize_waveform
from ADFWI.fwi.transforms import DataMask, DataTransformPipeline, LowPassFilter, ReceiverMask, TraceNormalize, normalize_waveform as transform_normalize_waveform


class DataTransformTests(unittest.TestCase):
    def _waveforms(self, device="cpu", dtype=torch.float32):
        synthetic = torch.tensor(
            [
                [[1.0, 0.0, -2.0], [2.0, 0.0, -4.0], [0.0, 0.0, 0.0]],
                [[-1.0, 3.0, 0.0], [-2.0, 6.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            device=device,
            dtype=dtype,
        )
        observed = synthetic * 2
        return synthetic, observed

    def test_empty_pipeline_returns_inputs(self):
        synthetic, observed = self._waveforms()
        out_syn, out_obs = DataTransformPipeline()(synthetic, observed)

        self.assertTrue(torch.equal(out_syn, synthetic))
        self.assertTrue(torch.equal(out_obs, observed))

    def test_pipeline_rejects_shape_mismatch(self):
        synthetic, observed = self._waveforms()
        with self.assertRaises(ValueError):
            DataTransformPipeline()(synthetic, observed[:, :, :2])

    def test_transform_package_exports_waveform_normalization(self):
        data = torch.tensor([[[0.0, 2.0], [2.0, -4.0]]])

        self.assertTrue(torch.equal(transform_normalize_waveform(data), normalize_waveform(data)))

    def test_trace_normalize_reuses_shared_waveform_normalization(self):
        synthetic, observed = self._waveforms()

        out_syn, out_obs = TraceNormalize()(synthetic, observed)

        self.assertTrue(torch.equal(out_syn, normalize_waveform(synthetic)))
        self.assertTrue(torch.equal(out_obs, normalize_waveform(observed)))

    def test_trace_normalize_respects_custom_dimension(self):
        synthetic = torch.tensor([[[1.0, -3.0], [2.0, 6.0]]])
        observed = synthetic * 2

        out_syn, out_obs = TraceNormalize(dim=2)(synthetic, observed)

        self.assertTrue(torch.equal(out_syn, normalize_waveform(synthetic, dim=2)))
        self.assertTrue(torch.equal(out_obs, normalize_waveform(observed, dim=2)))

    def test_trace_normalize_preserves_zero_traces(self):
        synthetic, observed = self._waveforms()
        out_syn, out_obs = TraceNormalize()(synthetic, observed)

        self.assertTrue(torch.isfinite(out_syn).all())
        self.assertTrue(torch.isfinite(out_obs).all())
        self.assertTrue(torch.equal(out_syn[:, :, 1][0], torch.zeros_like(out_syn[:, :, 1][0])))
        self.assertAlmostEqual(float(out_syn[0, :, 0].abs().max().item()), 1.0)
        self.assertAlmostEqual(float(out_obs[1, :, 1].abs().max().item()), 1.0)

    def test_receiver_mask_expands_shot_receiver_mask(self):
        synthetic, observed = self._waveforms()
        mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)
        out_syn, out_obs = ReceiverMask(mask)(synthetic, observed)

        self.assertTrue(torch.equal(out_syn[:, :, 1][0], torch.zeros_like(out_syn[:, :, 1][0])))
        self.assertTrue(torch.equal(out_syn[:, :, 0][1], torch.zeros_like(out_syn[:, :, 0][1])))
        self.assertTrue(torch.equal(out_obs[:, :, 1][0], torch.zeros_like(out_obs[:, :, 1][0])))

    def test_data_mask_applies_sample_mask(self):
        synthetic, observed = self._waveforms()
        mask = torch.ones_like(synthetic)
        mask[:, 1, :] = 0
        out_syn, out_obs = DataMask(mask)(synthetic, observed)

        self.assertTrue(torch.equal(out_syn[:, 1, :], torch.zeros_like(out_syn[:, 1, :])))
        self.assertTrue(torch.equal(out_obs[:, 1, :], torch.zeros_like(out_obs[:, 1, :])))
        self.assertTrue(torch.equal(out_syn[:, 0, :], synthetic[:, 0, :]))

    def test_data_mask_can_be_optional_noop(self):
        synthetic, observed = self._waveforms()
        out_syn, out_obs = DataMask(required=False)(synthetic, observed, context={})

        self.assertTrue(torch.equal(out_syn, synthetic))
        self.assertTrue(torch.equal(out_obs, observed))

    def test_data_mask_can_apply_to_synthetic_only(self):
        synthetic, observed = self._waveforms()
        mask = torch.ones_like(synthetic)
        mask[:, 1, :] = 0
        out_syn, out_obs = DataMask(mask, apply_to="synthetic")(synthetic, observed)

        self.assertTrue(torch.equal(out_syn[:, 1, :], torch.zeros_like(out_syn[:, 1, :])))
        self.assertTrue(torch.equal(out_obs, observed))

    def test_masks_can_come_from_context(self):
        synthetic, observed = self._waveforms()
        pipeline = DataTransformPipeline([ReceiverMask(), DataMask()])
        context = {
            "receiver_mask": np.array([[1, 1, 0], [1, 0, 1]], dtype=np.float32),
            "data_mask": torch.ones_like(synthetic),
        }
        context["data_mask"][:, 2, :] = 0
        out_syn, _ = pipeline(synthetic, observed, context=context)

        self.assertTrue(torch.equal(out_syn[:, :, 2][0], torch.zeros_like(out_syn[:, :, 2][0])))
        self.assertTrue(torch.equal(out_syn[:, :, 1][1], torch.zeros_like(out_syn[:, :, 1][1])))
        self.assertTrue(torch.equal(out_syn[:, 2, :], torch.zeros_like(out_syn[:, 2, :])))

    def test_low_pass_filter_attenuates_high_frequency_energy(self):
        dt = 0.001
        time = torch.arange(128, dtype=torch.float32) * dt
        low = torch.sin(2 * torch.pi * 20.0 * time)
        high = 0.5 * torch.sin(2 * torch.pi * 180.0 * time)
        synthetic = (low + high).reshape(1, -1, 1)
        observed = synthetic.clone()

        out_syn, out_obs = LowPassFilter(cutoff_freq=60.0, dt=dt, filter_length=51)(synthetic, observed)

        before = torch.fft.rfft(synthetic, dim=1).abs()
        after = torch.fft.rfft(out_syn, dim=1).abs()
        freqs = torch.fft.rfftfreq(synthetic.shape[1], d=dt)
        high_band = freqs > 120.0

        self.assertLess(float(after[:, high_band, :].mean().item()), float(before[:, high_band, :].mean().item()) * 0.35)
        self.assertTrue(torch.equal(out_obs, out_syn))

    def test_low_pass_filter_can_be_optional_noop(self):
        synthetic, observed = self._waveforms()
        out_syn, out_obs = LowPassFilter(required=False)(synthetic, observed, context={})

        self.assertTrue(torch.equal(out_syn, synthetic))
        self.assertTrue(torch.equal(out_obs, observed))

    def test_low_pass_filter_backward_on_cpu(self):
        synthetic, observed = self._waveforms(dtype=torch.float64)
        synthetic.requires_grad_(True)
        observed.requires_grad_(True)
        out_syn, out_obs = LowPassFilter(cutoff_freq=80.0, dt=0.001, filter_length=15)(synthetic, observed)
        loss = (out_syn.square().mean() + out_obs.square().mean())
        loss.backward()

        self.assertIsNotNone(synthetic.grad)
        self.assertIsNotNone(observed.grad)
        self.assertTrue(torch.isfinite(synthetic.grad).all())
        self.assertTrue(torch.isfinite(observed.grad).all())

    def test_transforms_preserve_dtype_and_device_on_cpu(self):
        synthetic, observed = self._waveforms(dtype=torch.float64)
        out_syn, out_obs = DataTransformPipeline([TraceNormalize(), ReceiverMask([[1, 0, 1], [1, 1, 0]])])(synthetic, observed)

        self.assertEqual(out_syn.dtype, torch.float64)
        self.assertEqual(out_obs.dtype, torch.float64)
        self.assertEqual(out_syn.device.type, "cpu")

    def test_transforms_preserve_dtype_and_device_on_npu_when_available(self):
        npu = getattr(torch, "npu", None)
        is_available = getattr(npu, "is_available", None)
        if not (callable(is_available) and is_available()):
            self.skipTest("NPU is not available on this machine")

        synthetic, observed = self._waveforms(device="npu:0", dtype=torch.float32)
        out_syn, out_obs = DataTransformPipeline([TraceNormalize(), ReceiverMask([[1, 0, 1], [1, 1, 0]])])(synthetic, observed)

        self.assertEqual(out_syn.dtype, torch.float32)
        self.assertEqual(out_obs.dtype, torch.float32)
        self.assertEqual(out_syn.device.type, "npu")
        self.assertTrue(torch.isfinite(out_syn).all().cpu().item())

    def test_low_pass_filter_runs_on_npu_when_available(self):
        npu = getattr(torch, "npu", None)
        is_available = getattr(npu, "is_available", None)
        if not (callable(is_available) and is_available()):
            self.skipTest("NPU is not available on this machine")

        synthetic, observed = self._waveforms(device="npu:0", dtype=torch.float32)
        synthetic.requires_grad_(True)
        out_syn, out_obs = LowPassFilter(cutoff_freq=80.0, dt=0.001, filter_length=15)(synthetic, observed)
        loss = out_syn.square().mean() + out_obs.square().mean()
        loss.backward()

        self.assertEqual(out_syn.device.type, "npu")
        self.assertEqual(out_obs.device.type, "npu")
        self.assertTrue(torch.isfinite(out_syn).all().cpu().item())
        self.assertIsNotNone(synthetic.grad)
        self.assertTrue(torch.isfinite(synthetic.grad).all().cpu().item())


if __name__ == "__main__":
    unittest.main()
