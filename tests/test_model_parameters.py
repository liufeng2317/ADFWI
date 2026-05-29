import unittest

import numpy as np
import torch

from ADFWI.model import AnisotropicElasticModel, IsotropicElasticModel
from ADFWI.model.parameters import (
    elastic_moduli_for_TI,
    elastic_moduli_init,
    parameter_staggered_grid,
)


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

    def test_parameter_staggered_grid_matches_current_stencil(self):
        field = torch.arange(20, dtype=torch.float32).reshape(4, 5) + 1.0

        bx, bz, muxz, c44, c55, c66 = parameter_staggered_grid(
            field, field, field, field, field, 5, 4
        )

        expected_bx = 0.5 * (field[:, 0:4] + field[:, 1:5])
        expected_bz = 0.5 * (field[0:3, :] + field[1:4, :])
        expected_staggered = 0.2 * (
            field[1:3, 1:4]
            + field[2:4, 1:4]
            + field[1:3, 2:5]
            + field[2:4, 1:4]
            + field[2:4, 2:5]
        )

        self.assertTrue(torch.equal(bx, expected_bx))
        self.assertTrue(torch.equal(bz, expected_bz))
        self.assertTrue(torch.equal(muxz, expected_staggered))
        self.assertTrue(torch.equal(c44, expected_staggered))
        self.assertTrue(torch.equal(c55, expected_staggered))
        self.assertTrue(torch.equal(c66, expected_staggered))

    def test_parameter_staggered_grid_preserves_constant_fields(self):
        mu = torch.full((4, 5), 7.0)
        b = torch.full((4, 5), 0.25)
        c44 = torch.full((4, 5), 11.0)
        c55 = torch.full((4, 5), 13.0)
        c66 = torch.full((4, 5), 17.0)

        bx, bz, muxz, c44_staggered, c55_staggered, c66_staggered = parameter_staggered_grid(
            mu, b, c44, c55, c66, 5, 4
        )

        self.assertTrue(torch.equal(bx, torch.full((4, 4), 0.25)))
        self.assertTrue(torch.equal(bz, torch.full((3, 5), 0.25)))
        self.assertTrue(torch.equal(muxz, torch.full((2, 3), 7.0)))
        self.assertTrue(torch.equal(c44_staggered, torch.full((2, 3), 11.0)))
        self.assertTrue(torch.equal(c55_staggered, torch.full((2, 3), 13.0)))
        self.assertTrue(torch.equal(c66_staggered, torch.full((2, 3), 17.0)))

    def test_elastic_model_forward_staggered_shapes(self):
        nx, nz = 5, 4
        vp = np.full((nz, nx), 2200.0, dtype=np.float32)
        vs = np.full((nz, nx), 1200.0, dtype=np.float32)
        rho = np.full((nz, nx), 1800.0, dtype=np.float32)

        iso = IsotropicElasticModel(
            0, 0, nx, nz, 10, 10, vp, vs, rho, auto_update_rho=False, device="cpu"
        )
        iso.forward()

        self.assertEqual(tuple(iso.bx.shape), (nz, nx - 1))
        self.assertEqual(tuple(iso.bz.shape), (nz - 1, nx))
        self.assertEqual(tuple(iso.muxz.shape), (nz - 2, nx - 2))
        self.assertEqual(tuple(iso.CC[18].shape), (nz - 2, nx - 2))

        eps = np.full((nz, nx), 0.05, dtype=np.float32)
        gamma = np.full((nz, nx), 0.03, dtype=np.float32)
        delta = np.full((nz, nx), 0.02, dtype=np.float32)
        ani = AnisotropicElasticModel(
            0, 0, nx, nz, 10, 10, vp, vs, rho, eps, gamma, delta, device="cpu"
        )
        ani.forward()

        self.assertEqual(tuple(ani.bx.shape), (nz, nx - 1))
        self.assertEqual(tuple(ani.bz.shape), (nz - 1, nx))
        self.assertEqual(tuple(ani.muxz.shape), (nz - 2, nx - 2))
        self.assertEqual(tuple(ani.CC[18].shape), (nz - 2, nx - 2))


if __name__ == "__main__":
    unittest.main()
