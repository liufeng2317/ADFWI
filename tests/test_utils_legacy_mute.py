import unittest

import numpy as np
import torch

from ADFWI.utils.first_arrivel_picking import apply_mute, brutal_picker, mask, mute_arrival
from ADFWI.utils.offset_mute import mute_offset


class UtilsLegacyMuteTests(unittest.TestCase):
    def test_brutal_picker_uses_trace_relative_threshold(self):
        traces = np.array(
            [
                [0.0, 0.0, 0.002, 0.0],
                [0.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        picks = brutal_picker(traces)

        np.testing.assert_array_equal(picks, np.array([2, 1, 0]))

    def test_mask_internal_window_matches_current_taper_formula(self):
        nt = 8
        length = 4
        itmin = 2
        itmax = 6

        result = mask(itmin, itmax, nt, length)

        expected = np.ones(nt)
        expected[:itmin] = 0.0
        expected[itmin:itmax] = np.sin(np.linspace(0, np.pi, 2 * length))[:length]
        np.testing.assert_allclose(result, expected)

    def test_mask_after_record_returns_zero_mask(self):
        np.testing.assert_allclose(mask(itmin=9, itmax=12, nt=8, length=4), np.zeros(8))

    def test_mute_arrival_applies_one_minus_mask_on_trace_device(self):
        trace = torch.arange(8, dtype=torch.float64)
        itmin = 2
        itmax = 6
        length = 4

        muted = mute_arrival(trace, itmin, itmax, "late", nt=8, length=length)

        expected = trace * torch.as_tensor(1.0 - mask(itmin, itmax, 8, length), dtype=trace.dtype)
        self.assertEqual(muted.dtype, trace.dtype)
        self.assertEqual(muted.device, trace.device)
        torch.testing.assert_close(muted, expected)

    def test_apply_mute_preserves_shape_dtype_and_matches_manual_trace_loop(self):
        nt = 140
        shot = torch.zeros((nt, 2), dtype=torch.float64)
        shot[20:, 0] = 1.0
        shot[40:, 1] = 2.0
        late_window = 0.0
        dt = 0.001
        length = 100
        picks = brutal_picker(shot.numpy().T) + np.ceil(late_window / dt)

        muted = apply_mute(late_window, shot, dt)

        expected = shot.clone()
        for itrace in range(shot.shape[-1]):
            itmin = int(picks[itrace] - length / 2)
            itmax = int(itmin + length)
            expected[:, itrace] = mute_arrival(shot[:, itrace], itmin, itmax, "late", nt, length)

        self.assertEqual(muted.shape, shot.shape)
        self.assertEqual(muted.dtype, shot.dtype)
        torch.testing.assert_close(muted, expected)

    def test_mute_offset_zeros_near_offsets_and_mutates_input_tensor(self):
        rcv_x = torch.tensor([[0.0, 1.0, 3.0], [1.0, 4.0, 5.0]])
        src_x = torch.tensor([1.0, 4.0])
        waveform = torch.ones((2, 4, 3), dtype=torch.float64)

        result = mute_offset(rcv_x, src_x, dx=10.0, waveform=waveform, distance_threshold=15.0)

        expected = torch.ones((2, 4, 3), dtype=torch.float64)
        expected[0, :, 0] = 0.0
        expected[0, :, 1] = 0.0
        expected[1, :, 1] = 0.0
        expected[1, :, 2] = 0.0
        self.assertIs(result, waveform)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
