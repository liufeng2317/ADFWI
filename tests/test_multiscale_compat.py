import unittest

import torch

from ADFWI.fwi import multiScaleProcessing as legacy
from ADFWI.fwi.multiscale import (
    Lfilter,
    adj_lowpass,
    data2d_to_3d,
    data3d_to_2d,
    lowpass,
    lpass,
)


class MultiscaleCompatibilityTests(unittest.TestCase):
    def test_legacy_multiscale_module_reexports_new_package_objects(self):
        self.assertIs(legacy.Lfilter, Lfilter)
        self.assertIs(legacy.adj_lowpass, adj_lowpass)
        self.assertIs(legacy.data2d_to_3d, data2d_to_3d)
        self.assertIs(legacy.data3d_to_2d, data3d_to_2d)
        self.assertIs(legacy.lowpass, lowpass)
        self.assertIs(legacy.lpass, lpass)

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
