import unittest
from types import SimpleNamespace

import numpy as np
import torch

from ADFWI.fwi.runtime.gradient import process_parameter_gradient
from ADFWI.propagator import GradProcessor, TorchGradProcessor


class TorchGradProcessorTests(unittest.TestCase):
    def assert_torch_matches_legacy(self, grad, *, vmax=2500.0, forw=None, rtol=1e-5, atol=2e-3, **processor_kwargs):
        legacy = GradProcessor(**processor_kwargs)
        torch_processor = TorchGradProcessor(**processor_kwargs)

        expected = legacy.forward(
            nx=grad.shape[1],
            nz=grad.shape[0],
            vmax=vmax,
            grad=grad.copy(),
            forw=None if forw is None else forw.copy(),
        )
        actual = torch_processor.forward_torch(
            nx=grad.shape[1],
            nz=grad.shape[0],
            vmax=torch.tensor(vmax),
            grad=torch.tensor(grad),
            forw=None if forw is None else torch.tensor(forw),
        ).cpu().numpy()

        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    def test_torch_processor_matches_legacy_norm_only(self):
        grad = np.array([[1.0, -2.0, 0.5], [3.0, -4.0, 2.5]], dtype=np.float32)
        self.assert_torch_matches_legacy(grad, norm_grad=True, forw_illumination=False, rtol=1e-6, atol=1e-6)

    def test_torch_processor_matches_legacy_marine_mute_mask_and_norm(self):
        grad = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        mask = np.array(
            [[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.5, 1.0, 1.0, 0.5], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        self.assert_torch_matches_legacy(
            grad,
            vmax=3000.0,
            grad_mute=1,
            grad_mask=mask,
            norm_grad=True,
            forw_illumination=False,
            marine_or_land="marine",
            rtol=1e-6,
            atol=1e-6,
        )

    def test_torch_processor_matches_legacy_marine_smoothing(self):
        grad = np.arange(1, 37, dtype=np.float32).reshape(6, 6)

        self.assert_torch_matches_legacy(
            grad,
            grad_mute=2,
            grad_smooth=1,
            grad_mask=None,
            norm_grad=True,
            forw_illumination=False,
            marine_or_land="marine",
        )

    def test_torch_processor_matches_legacy_land_mute_and_smoothing(self):
        grad = np.arange(1, 37, dtype=np.float32).reshape(6, 6)

        self.assert_torch_matches_legacy(
            grad,
            grad_mute=2,
            grad_smooth=1,
            grad_mask=None,
            norm_grad=True,
            forw_illumination=False,
            marine_or_land="land",
        )

    def test_torch_processor_matches_legacy_forward_illumination(self):
        grad = np.arange(1, 37, dtype=np.float32).reshape(6, 6)
        forw = np.linspace(0.2, 2.0, 36, dtype=np.float32).reshape(6, 6)

        self.assert_torch_matches_legacy(
            grad,
            forw=forw,
            grad_mute=0,
            grad_smooth=0,
            grad_mask=None,
            norm_grad=True,
            forw_illumination=True,
            marine_or_land="land",
        )

    def test_runtime_dispatches_torch_processor_without_changing_dtype_or_device(self):
        param = torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float64, requires_grad=True)
        param.grad = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
        processor = TorchGradProcessor(norm_grad=True, forw_illumination=False)
        model = SimpleNamespace(nz=2, nx=2)
        propagator = SimpleNamespace(dtype=torch.float64, device=torch.device("cpu"))

        process_parameter_gradient(
            param,
            processor,
            model=model,
            propagator=propagator,
            forw=None,
            processor_type=GradProcessor,
        )

        expected = GradProcessor(norm_grad=True, forw_illumination=False).forward(
            nx=2,
            nz=2,
            vmax=8.0,
            grad=np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64),
            forw=None,
        )
        self.assertEqual(param.grad.dtype, torch.float64)
        self.assertEqual(param.grad.device.type, "cpu")
        np.testing.assert_allclose(param.grad.detach().cpu().numpy(), expected, rtol=1e-12, atol=1e-12)

    def test_runtime_dispatches_torch_processor_from_list_index(self):
        param = torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True)
        param.grad = torch.tensor([[2.0, -4.0]], dtype=torch.float32)
        first = TorchGradProcessor(norm_grad=False, forw_illumination=False)
        second = TorchGradProcessor(norm_grad=True, forw_illumination=False)
        model = SimpleNamespace(nz=1, nx=2)
        propagator = SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"))

        process_parameter_gradient(
            param,
            [first, second],
            model=model,
            propagator=propagator,
            forw=None,
            idx=1,
            processor_type=GradProcessor,
        )

        expected = np.array([[1.0, -2.0]], dtype=np.float32)
        np.testing.assert_allclose(param.grad.detach().cpu().numpy(), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
