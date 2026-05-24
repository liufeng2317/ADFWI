import unittest

import numpy as np

from ADFWI.fwi.loop import iter_batch_ranges


class TestFWILoopHelpers(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
