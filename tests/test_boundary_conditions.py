import unittest

import numpy as np

from ADFWI.propagator.boundary_condition import (
    bc_gerjan,
    bc_pml,
    bc_pml_xz,
    bc_sincos,
)


class BoundaryConditionTests(unittest.TestCase):
    def setUp(self):
        self.nx = 8
        self.nz = 6
        self.dx = 10.0
        self.dz = 10.0
        self.pml = 4
        self.vmax = 2500.0

    def _expected_shape(self, free_surface):
        nz_padded = self.nz + self.pml if free_surface else self.nz + 2 * self.pml
        nx_padded = self.nx + 2 * self.pml
        return nz_padded, nx_padded

    def assert_finite_shape(self, array, free_surface):
        self.assertEqual(array.shape, self._expected_shape(free_surface))
        self.assertTrue(np.isfinite(array).all())

    def test_bc_pml_shape_and_free_surface_contract(self):
        damp_free = bc_pml(
            self.nx,
            self.nz,
            self.dx,
            self.dz,
            self.pml,
            self.vmax,
            free_surface=True,
        )
        damp_absorbing_top = bc_pml(
            self.nx,
            self.nz,
            self.dx,
            self.dz,
            self.pml,
            self.vmax,
            free_surface=False,
        )

        self.assert_finite_shape(damp_free, free_surface=True)
        self.assert_finite_shape(damp_absorbing_top, free_surface=False)
        self.assertGreaterEqual(damp_free.min(), 0.0)
        self.assertGreaterEqual(damp_absorbing_top.min(), 0.0)
        self.assertEqual(float(damp_free[0, self.pml]), 0.0)
        self.assertGreater(float(damp_absorbing_top[0, self.pml]), 0.0)
        self.assertGreater(float(damp_free[-1, self.pml]), 0.0)

    def test_bc_sincos_shape_and_range_contract(self):
        damp_free = bc_sincos(
            self.nx,
            self.nz,
            self.dx,
            self.dz,
            self.pml,
            free_surface=True,
        )
        damp_absorbing_top = bc_sincos(
            self.nx,
            self.nz,
            self.dx,
            self.dz,
            self.pml,
            free_surface=False,
        )

        self.assert_finite_shape(damp_free, free_surface=True)
        self.assert_finite_shape(damp_absorbing_top, free_surface=False)
        self.assertGreaterEqual(damp_free.min(), 0.0)
        self.assertLessEqual(damp_free.max(), 1.0)
        self.assertGreaterEqual(damp_absorbing_top.min(), 0.0)
        self.assertLessEqual(damp_absorbing_top.max(), 1.0)
        self.assertEqual(float(damp_free[0, self.pml]), 1.0)
        self.assertEqual(float(damp_absorbing_top[0, self.pml]), 0.0)
        self.assertEqual(float(damp_free[-1, self.pml]), 0.0)

    def test_bc_gerjan_shape_and_range_contract(self):
        damp_free = bc_gerjan(
            self.nx,
            self.nz,
            self.dx,
            self.dz,
            self.pml,
            free_surface=True,
        )
        damp_absorbing_top = bc_gerjan(
            self.nx,
            self.nz,
            self.dx,
            self.dz,
            self.pml,
            free_surface=False,
        )

        self.assert_finite_shape(damp_free, free_surface=True)
        self.assert_finite_shape(damp_absorbing_top, free_surface=False)
        self.assertGreater(damp_free.min(), 0.0)
        self.assertLessEqual(damp_free.max(), 1.0)
        self.assertGreater(damp_absorbing_top.min(), 0.0)
        self.assertLessEqual(damp_absorbing_top.max(), 1.0)
        self.assertEqual(float(damp_free[0, self.pml]), 1.0)
        self.assertLess(float(damp_absorbing_top[0, self.pml]), 1.0)
        self.assertLess(float(damp_free[-1, self.pml]), 1.0)

    def test_bc_pml_xz_shape_and_free_surface_contract(self):
        bcx_free, bcz_free = bc_pml_xz(
            self.nx,
            self.nz,
            self.dx,
            self.dz,
            self.pml,
            self.vmax,
            free_surface=True,
        )
        bcx_absorbing_top, bcz_absorbing_top = bc_pml_xz(
            self.nx,
            self.nz,
            self.dx,
            self.dz,
            self.pml,
            self.vmax,
            free_surface=False,
        )

        for array in (bcx_free, bcz_free):
            self.assert_finite_shape(array, free_surface=True)
            self.assertGreaterEqual(array.min(), 0.0)
        for array in (bcx_absorbing_top, bcz_absorbing_top):
            self.assert_finite_shape(array, free_surface=False)
            self.assertGreaterEqual(array.min(), 0.0)

        self.assertGreater(float(bcx_free[0, 0]), 0.0)
        self.assertEqual(float(bcz_free[0, self.pml]), 0.0)
        self.assertGreater(float(bcz_free[-1, self.pml]), 0.0)
        self.assertGreater(float(bcz_absorbing_top[0, self.pml]), 0.0)


if __name__ == "__main__":
    unittest.main()
