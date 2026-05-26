import unittest
from types import SimpleNamespace

import torch

from ADFWI.fwi.runtime import align_regularization_backend, validate_model_propagator_devices


class FWIRuntimeTests(unittest.TestCase):
    def test_validate_model_propagator_devices_allows_matching_devices(self):
        model = SimpleNamespace(device=torch.device("cpu"))
        propagator = SimpleNamespace(device=torch.device("cpu"))

        validate_model_propagator_devices(model, propagator)

    def test_validate_model_propagator_devices_rejects_mismatch(self):
        model = SimpleNamespace(device=torch.device("meta"))
        propagator = SimpleNamespace(device=torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "device .* inconsistent"):
            validate_model_propagator_devices(model, propagator)

    def test_align_regularization_backend_moves_floating_tensors_and_preserves_integer_dtype(self):
        reg = SimpleNamespace(
            device=torch.device("meta"),
            dtype=torch.float64,
            floating=torch.ones(2, dtype=torch.float64),
            integer=torch.ones(2, dtype=torch.int64),
            label="regularization",
        )

        result = align_regularization_backend(reg, torch.device("cpu"), torch.float32)

        self.assertIs(result, reg)
        self.assertEqual(reg.device, torch.device("cpu"))
        self.assertEqual(reg.dtype, torch.float32)
        self.assertEqual(reg.floating.device.type, "cpu")
        self.assertEqual(reg.floating.dtype, torch.float32)
        self.assertEqual(reg.integer.device.type, "cpu")
        self.assertEqual(reg.integer.dtype, torch.int64)
        self.assertEqual(reg.label, "regularization")

    def test_align_regularization_backend_accepts_none(self):
        self.assertIsNone(align_regularization_backend(None, torch.device("cpu"), torch.float32))


if __name__ == "__main__":
    unittest.main()
