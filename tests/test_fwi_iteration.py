import unittest

import numpy as np
import torch

from ADFWI.fwi.iteration import build_batch_loss, iter_batch_ranges, set_batch_description


class DummyProgressBar:
    def __init__(self):
        self.description = None

    def set_description(self, value):
        self.description = value


class TestFWIIterationHelpers(unittest.TestCase):
    def test_none_batch_size_uses_full_batch(self):
        batches = list(iter_batch_ranges(5, None))

        self.assertEqual(len(batches), 1)
        self.assertEqual((batches[0].batch, batches[0].begin, batches[0].end), (0, 0, 5))
        np.testing.assert_array_equal(batches[0].shot_index, np.arange(5))

    def test_batch_size_larger_than_shots_uses_full_batch(self):
        batches = list(iter_batch_ranges(3, 10))

        self.assertEqual(len(batches), 1)
        self.assertEqual((batches[0].begin, batches[0].end), (0, 3))
        np.testing.assert_array_equal(batches[0].shot_index, np.arange(3))

    def test_partial_last_batch_matches_legacy_ranges(self):
        batches = list(iter_batch_ranges(5, 2))

        self.assertEqual([(b.batch, b.begin, b.end) for b in batches], [(0, 0, 2), (1, 2, 4), (2, 4, 5)])
        np.testing.assert_array_equal(batches[0].shot_index, np.arange(0, 2))
        np.testing.assert_array_equal(batches[1].shot_index, np.arange(2, 4))
        np.testing.assert_array_equal(batches[2].shot_index, np.arange(4, 5))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            list(iter_batch_ranges(0, 1))
        with self.assertRaises(ValueError):
            list(iter_batch_ranges(5, 0))
        with self.assertRaises(ValueError):
            list(iter_batch_ranges(5, -1))

    def test_build_batch_loss_without_regularization_keeps_data_loss(self):
        data_loss = torch.tensor(2.0, requires_grad=True)

        batch_loss = build_batch_loss(data_loss)
        batch_loss.tensor.backward()

        self.assertIs(batch_loss.tensor, data_loss)
        self.assertEqual(batch_loss.scalar, 2.0)
        self.assertEqual(float(data_loss.grad.item()), 1.0)

    def test_build_batch_loss_with_regularization_matches_expanded_sum(self):
        data_loss = torch.tensor(2.0, requires_grad=True)
        regularization_loss = torch.tensor(3.0, requires_grad=True)

        batch_loss = build_batch_loss(data_loss, regularization_loss)
        batch_loss.tensor.backward()

        self.assertEqual(float(batch_loss.tensor.detach().item()), 5.0)
        self.assertEqual(batch_loss.scalar, 5.0)
        self.assertEqual(float(data_loss.grad.item()), 1.0)
        self.assertEqual(float(regularization_loss.grad.item()), 1.0)

    def test_set_batch_description_single_batch_matches_legacy_label(self):
        batch_range = list(iter_batch_ranges(5, None))[0]
        progress_bar = DummyProgressBar()

        set_batch_description(progress_bar, batch_range, 1)

        self.assertEqual(progress_bar.description, "Shot:0 to 5")

    def test_set_batch_description_multi_batch_keeps_description_unchanged(self):
        batch_range = list(iter_batch_ranges(5, 2))[0]
        progress_bar = DummyProgressBar()

        set_batch_description(progress_bar, batch_range, 3)

        self.assertIsNone(progress_bar.description)


if __name__ == "__main__":
    unittest.main()
