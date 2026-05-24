import unittest

import numpy as np
import torch

from ADFWI.fwi.transforms import select_or_mask_receivers


def legacy_select_or_mask(synthetic, observed, receiver_mask):
    if synthetic.shape == observed.shape:
        receiver_mask_3d = receiver_mask.unsqueeze(1).expand(-1, synthetic.shape[1], -1).to(synthetic.device)
        return synthetic * receiver_mask_3d

    selected = torch.zeros_like(observed, device=synthetic.device)
    receiver_mask_2d = receiver_mask.cpu()
    for shot in range(synthetic.shape[0]):
        selected[shot] = synthetic[shot, ..., np.argwhere(receiver_mask_2d[shot]).tolist()].squeeze()
    return selected


class ReceiverSelectionTests(unittest.TestCase):
    def test_same_shape_receiver_mask_matches_legacy_multiply(self):
        synthetic = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        observed = torch.zeros_like(synthetic)
        receiver_mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)

        actual = select_or_mask_receivers(synthetic, observed, receiver_mask)
        expected = legacy_select_or_mask(synthetic, observed, receiver_mask)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.shape, observed.shape)

    def test_trace_missing_selection_matches_legacy_order(self):
        synthetic = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
        observed = torch.zeros((2, 5, 2), dtype=torch.float32)
        receiver_mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)

        actual = select_or_mask_receivers(synthetic, observed, receiver_mask)
        expected = legacy_select_or_mask(synthetic, observed, receiver_mask)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.shape, observed.shape)
        self.assertTrue(torch.equal(actual[0, :, 0], synthetic[0, :, 0]))
        self.assertTrue(torch.equal(actual[0, :, 1], synthetic[0, :, 2]))
        self.assertTrue(torch.equal(actual[1, :, 0], synthetic[1, :, 1]))
        self.assertTrue(torch.equal(actual[1, :, 1], synthetic[1, :, 3]))

    def test_single_active_trace_keeps_receiver_dimension(self):
        synthetic = torch.arange(1 * 5 * 3, dtype=torch.float32).reshape(1, 5, 3)
        observed = torch.zeros((1, 5, 1), dtype=torch.float32)
        receiver_mask = torch.tensor([[0, 1, 0]], dtype=torch.float32)

        actual = select_or_mask_receivers(synthetic, observed, receiver_mask)

        self.assertEqual(actual.shape, observed.shape)
        self.assertTrue(torch.equal(actual[0, :, 0], synthetic[0, :, 1]))

    def test_preserves_device_and_dtype_on_cpu(self):
        synthetic = torch.ones((1, 3, 2), dtype=torch.float64)
        observed = torch.zeros_like(synthetic)
        receiver_mask = torch.tensor([[1, 0]], dtype=torch.float32)

        actual = select_or_mask_receivers(synthetic, observed, receiver_mask)

        self.assertEqual(actual.dtype, synthetic.dtype)
        self.assertEqual(actual.device, synthetic.device)


if __name__ == "__main__":
    unittest.main()
