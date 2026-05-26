import unittest
from types import SimpleNamespace

import torch

from ADFWI.fwi.runtime import (
    align_regularization_backend,
    calculate_regularization_loss,
    process_parameter_gradient,
    validate_model_propagator_devices,
)


class DummyRegularization:
    def __init__(self):
        self.alphax = None
        self.alphaz = None
        self.calls = []

    def forward(self, model_param):
        self.calls.append((self.alphax, self.alphaz))
        return torch.sum(model_param * self.alphax) + torch.sum(model_param * self.alphaz * 0.1)


class DummyGradientProcessor:
    def __init__(self, scale):
        self.scale = scale
        self.calls = []

    def forward(self, nz, nx, vmax, grad, forw):
        self.calls.append(
            {
                "nz": nz,
                "nx": nx,
                "vmax": vmax,
                "grad": grad.copy(),
                "forw": forw,
            }
        )
        return grad * self.scale + vmax


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

    def test_calculate_regularization_loss_matches_legacy_formula(self):
        param = torch.ones((2, 2), requires_grad=True)
        regularization = DummyRegularization()

        loss = calculate_regularization_loss(param, 2.0, 3.0, regularization)

        expected = torch.sum(param * 2.0) + torch.sum(param * 3.0 * 0.1)
        self.assertTrue(torch.equal(loss, expected))
        self.assertEqual(regularization.calls, [(2.0, 3.0)])

    def test_calculate_regularization_loss_skips_disabled_parameter(self):
        param = torch.ones((2, 2), requires_grad=False)
        regularization = DummyRegularization()

        loss = calculate_regularization_loss(param, 2.0, 3.0, regularization)

        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(loss.device, param.device)
        self.assertEqual(regularization.calls, [])

    def test_calculate_regularization_loss_skips_zero_weights(self):
        param = torch.ones((2, 2), requires_grad=True)
        regularization = DummyRegularization()

        loss = calculate_regularization_loss(param, 0.0, 0.0, regularization)

        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual((regularization.alphax, regularization.alphaz), (0.0, 0.0))
        self.assertEqual(regularization.calls, [])

    def test_process_parameter_gradient_single_processor_matches_legacy_formula(self):
        param = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        param.grad = torch.full_like(param, 2.0)
        processor = DummyGradientProcessor(scale=0.5)
        model = SimpleNamespace(nz=2, nx=2)
        propagator = SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"))

        process_parameter_gradient(
            param,
            processor,
            model=model,
            propagator=propagator,
            forw="forward-wavefield",
            processor_type=DummyGradientProcessor,
        )

        self.assertTrue(torch.equal(param.grad, torch.full_like(param, 5.0)))
        self.assertEqual(len(processor.calls), 1)
        self.assertEqual(processor.calls[0]["nz"], 2)
        self.assertEqual(processor.calls[0]["nx"], 2)
        self.assertEqual(processor.calls[0]["vmax"], 4.0)
        self.assertEqual(processor.calls[0]["forw"], "forward-wavefield")

    def test_process_parameter_gradient_list_processor_uses_requested_index(self):
        param = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        param.grad = torch.ones_like(param)
        first = DummyGradientProcessor(scale=10.0)
        second = DummyGradientProcessor(scale=2.0)
        model = SimpleNamespace(nz=2, nx=2)
        propagator = SimpleNamespace(dtype=torch.float64, device=torch.device("cpu"))

        process_parameter_gradient(
            param,
            [first, second],
            model=model,
            propagator=propagator,
            forw=None,
            idx=1,
            processor_type=DummyGradientProcessor,
        )

        self.assertTrue(torch.equal(param.grad, torch.full_like(param, 6.0, dtype=torch.float64)))
        self.assertEqual(first.calls, [])
        self.assertEqual(len(second.calls), 1)
        self.assertEqual(param.grad.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
