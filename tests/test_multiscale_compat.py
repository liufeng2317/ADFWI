import unittest

import torch

from ADFWI.fwi.multiscale import (
    Lfilter,
    adj_lowpass,
    data2d_to_3d,
    data3d_to_2d,
    lowpass,
    lpass,
)


class MultiscaleCanonicalImportTests(unittest.TestCase):
    def test_multiscale_package_exports_legacy_lowpass_objects(self):
        self.assertTrue(callable(Lfilter))
        self.assertTrue(callable(adj_lowpass))
        self.assertTrue(callable(data2d_to_3d))
        self.assertTrue(callable(data3d_to_2d))
        self.assertTrue(callable(lowpass))
        self.assertTrue(callable(lpass))

    def test_data2d_3d_round_trip_keeps_values(self):
        ns, nt, nr = 2, 4, 3
        first = torch.arange(nt * ns * nr, dtype=torch.float32).reshape(nt, ns * nr)
        second = first + 100.0

        first_3d, second_3d = data2d_to_3d(first, second, ns, nr)
        actual_first, actual_second = data3d_to_2d(first_3d, second_3d)

        self.assertTrue(torch.equal(actual_first, first))
        self.assertTrue(torch.equal(actual_second, second))


if __name__ == "__main__":
    unittest.main()
