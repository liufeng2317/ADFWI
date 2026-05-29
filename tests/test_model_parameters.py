import unittest

import torch

from ADFWI.model.parameters import elastic_moduli_for_TI, elastic_moduli_init


class ModelParameterFormulaTests(unittest.TestCase):
    def _stiffness_components(self):
        cc = elastic_moduli_init(2, 3, "cpu", torch.float32)
        for idx, component in enumerate(cc):
            component.fill_(idx + 1)
        return cc

    def test_elastic_moduli_for_ti_vti_layout_is_stable(self):
        cc = self._stiffness_components()

        result = elastic_moduli_for_TI(cc, "vti")

        self.assertEqual(len(result), 21)
        expected_values = [
            1, -41, 3, 4, 5, 6,
            1, 3, 9, 10, 11,
            12, 13, 14, 15,
            16, 17, 18,
            16, 20,
            21,
        ]
        for output, expected in zip(result, expected_values):
            self.assertTrue(torch.equal(output, torch.full_like(output, expected)))

    def test_elastic_moduli_for_ti_hti_layout_is_stable(self):
        cc = self._stiffness_components()

        result = elastic_moduli_for_TI(cc, "hti")

        self.assertEqual(len(result), 21)
        expected_values = [
            12, 3, 3, 4, 5, 6,
            1, -41, 9, 10, 11,
            1, 13, 14, 15,
            21, 17, 18,
            16, 20,
            16,
        ]
        for output, expected in zip(result, expected_values):
            self.assertTrue(torch.equal(output, torch.full_like(output, expected)))

    def test_elastic_moduli_for_ti_rejects_unknown_type(self):
        cc = self._stiffness_components()

        with self.assertRaisesRegex(ValueError, "expected 'vti' or 'hti'"):
            elastic_moduli_for_TI(cc, "tti")


if __name__ == "__main__":
    unittest.main()
