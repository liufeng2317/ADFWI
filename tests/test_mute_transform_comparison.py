import unittest

import numpy as np
import torch

from ADFWI.fwi.transforms import (
    DataMask,
    DataTransformPipeline,
    LegacyLateWindowMute,
    LegacyLowPassFilter,
    LegacyOffsetMute,
    TraceNormalize,
)
from ADFWI.utils.first_arrivel_picking import apply_mute
from ADFWI.utils.offset_mute import mute_offset


class MuteTransformComparisonTests(unittest.TestCase):
    def _waveforms(self):
        nt = 160
        time = torch.arange(nt, dtype=torch.float64) * 0.001
        traces = []
        for shift in [10, 18, 26]:
            trace = torch.zeros(nt, dtype=torch.float64)
            trace[shift:] = torch.sin(2 * torch.pi * 25.0 * time[: nt - shift])
            traces.append(trace)
        synthetic = torch.stack(traces, dim=-1).unsqueeze(0)
        observed = synthetic * 1.5
        return synthetic, observed


    def _legacy_receiver_x(self, synthetic, receiver_mask, rcv_x_list):
        rcv_x = torch.zeros(synthetic.shape[0], synthetic.shape[-1])
        for i in range(synthetic.shape[0]):
            active_receivers = np.argwhere(receiver_mask[i].numpy()).reshape(-1).tolist()
            rcv_x[i] = rcv_x_list[active_receivers].squeeze()
        return rcv_x

    def _legacy_full_preprocess(
        self,
        synthetic,
        observed,
        receiver_mask,
        src_x,
        rcv_x_list,
        dx,
        offset_threshold,
        late_window,
        dt,
        cutoff_freq,
        data_mask,
    ):
        rcv_x = self._legacy_receiver_x(synthetic, receiver_mask, rcv_x_list)
        synthetic = mute_offset(rcv_x, src_x, dx, synthetic, offset_threshold)
        observed = mute_offset(rcv_x, src_x, dx, observed, offset_threshold)

        synthetic_temp = synthetic.clone()
        observed_temp = observed.clone()
        for i in range(synthetic.shape[0]):
            synthetic[i] = apply_mute(late_window, synthetic_temp[i], dt)
            observed[i] = apply_mute(late_window, observed_temp[i], dt)

        from ADFWI.fwi.multiScaleProcessing import lpass

        synthetic, observed = lpass(synthetic, observed, cutoff_freq, int(round(1.0 / dt)))
        synthetic = synthetic * data_mask
        synthetic, observed = TraceNormalize()(synthetic, observed)
        return synthetic, observed

    def test_late_window_mute_matches_legacy_apply_mute_loop(self):
        synthetic, observed = self._waveforms()
        late_window = 0.018
        dt = 0.001

        expected_syn = synthetic.clone()
        expected_obs = observed.clone()
        syn_source = synthetic.clone()
        obs_source = observed.clone()
        for i in range(synthetic.shape[0]):
            expected_syn[i] = apply_mute(late_window, syn_source[i], dt)
            expected_obs[i] = apply_mute(late_window, obs_source[i], dt)

        actual_syn, actual_obs = LegacyLateWindowMute(late_window=late_window, dt=dt)(synthetic.clone(), observed.clone())

        self.assertTrue(torch.equal(actual_syn, expected_syn))
        self.assertTrue(torch.equal(actual_obs, expected_obs))

    def test_late_window_mute_optional_noop(self):
        synthetic, observed = self._waveforms()
        out_syn, out_obs = LegacyLateWindowMute(required=False)(synthetic, observed, context={})

        self.assertTrue(torch.equal(out_syn, synthetic))
        self.assertTrue(torch.equal(out_obs, observed))

    def test_offset_mute_matches_legacy_fwi_branch(self):
        synthetic = torch.ones((2, 5, 3), dtype=torch.float64)
        observed = synthetic * 2
        receiver_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.bool)
        src_x = torch.tensor([1.0, 2.0])
        rcv_x_list = torch.tensor([0.0, 1.0, 2.0, 3.0])
        dx = 10.0
        threshold = 15.0

        rcv_x = torch.zeros(synthetic.shape[0], synthetic.shape[-1])
        for i in range(synthetic.shape[0]):
            active_receivers = np.argwhere(receiver_mask[i].numpy()).reshape(-1).tolist()
            rcv_x[i] = rcv_x_list[active_receivers].squeeze()
        expected_syn = mute_offset(rcv_x, src_x, dx, synthetic.clone(), threshold)
        expected_obs = mute_offset(rcv_x, src_x, dx, observed.clone(), threshold)

        context = {
            "receiver_mask": receiver_mask,
            "src_x": src_x,
            "rcv_x": rcv_x_list,
            "dx": dx,
            "offset_mute_threshold": threshold,
        }
        actual_syn, actual_obs = LegacyOffsetMute()(synthetic.clone(), observed.clone(), context=context)

        self.assertTrue(torch.equal(actual_syn, expected_syn))
        self.assertTrue(torch.equal(actual_obs, expected_obs))

    def test_default_pipeline_matches_legacy_fwi_preprocessing_order(self):
        synthetic, observed = self._waveforms()
        synthetic = torch.cat([synthetic, synthetic * 0.75], dim=0)
        observed = torch.cat([observed, observed * 0.5], dim=0)
        receiver_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.bool)
        src_x = torch.tensor([1.0, 2.0])
        rcv_x_list = torch.tensor([0.0, 1.0, 2.0, 3.0])
        dx = 10.0
        offset_threshold = 15.0
        late_window = 0.018
        dt = 0.001
        cutoff_freq = 70.0
        # FWI loads obs_data.data_masks through numpy2tensor(), whose default dtype is float32.
        data_mask = torch.ones(synthetic.shape, dtype=torch.float32)
        data_mask[:, :7, :] = 0.0

        expected_syn, expected_obs = self._legacy_full_preprocess(
            synthetic.clone(),
            observed.clone(),
            receiver_mask,
            src_x,
            rcv_x_list,
            dx,
            offset_threshold,
            late_window,
            dt,
            cutoff_freq,
            data_mask,
        )

        pipeline = DataTransformPipeline([
            LegacyOffsetMute(required=False),
            LegacyLateWindowMute(required=False),
            LegacyLowPassFilter(required=False),
            DataMask(required=False, apply_to="synthetic"),
            TraceNormalize(),
        ])
        context = {
            "receiver_mask": receiver_mask,
            "src_x": src_x,
            "rcv_x": rcv_x_list,
            "dx": dx,
            "offset_mute_threshold": offset_threshold,
            "late_window": late_window,
            "dt": dt,
            "cutoff_freq": cutoff_freq,
            "data_mask": data_mask,
        }
        actual_syn, actual_obs = pipeline(synthetic.clone(), observed.clone(), context=context)

        self.assertTrue(torch.equal(actual_syn, expected_syn))
        self.assertTrue(torch.equal(actual_obs, expected_obs))

    def test_offset_mute_optional_noop(self):
        synthetic, observed = self._waveforms()
        out_syn, out_obs = LegacyOffsetMute(required=False)(synthetic, observed, context={})

        self.assertTrue(torch.equal(out_syn, synthetic))
        self.assertTrue(torch.equal(out_obs, observed))


if __name__ == "__main__":
    unittest.main()
