import unittest
from types import SimpleNamespace

import numpy as np
import torch

from ADFWI.fwi.runtime import process_parameter_gradient
from ADFWI.propagator import GradProcessor, TorchGradProcessor


class TorchGradProcessorTests(unittest.TestCase):
    def test_torch_processor_matches_legacy_norm_only(self):
        grad = np.array([[1.0, -2.0, 0.5], [3.0, -4.0, 2.5]], dtype=np.float32)
        legacy = GradProcessor(norm_grad=True, forw_illumination=False)
        torch_processor = TorchGradProcessor(norm_grad=True, forw_illumination=False)

        expected = legacy.forward(nx=3, nz=2, vmax=2500.0, grad=grad.copy(), forw=None)
        actual = torch_processor.forward_torch(
            nx=3,
            nz=2,
            vmax=torch.tensor(2500.0),
            grad=torch.tensor(grad),
            forw=None,
        ).cpu().numpy()

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_torch_processor_matches_legacy_marine_mute_mask_and_norm(self):
        grad = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        mask = np.array(
            [[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.5, 1.0, 1.0, 0.5], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        legacy = GradProcessor(
            grad_mute=1,
            grad_mask=mask,
            norm_grad=True,
            forw_illumination=False,
            marine_or_land="marine",
        )
        torch_processor = TorchGradProcessor(
            grad_mute=1,
            grad_mask=mask,
            norm_grad=True,
            forw_illumination=False,
            marine_or_land="marine",
        )

        expected = legacy.forward(nx=4, nz=4, vmax=3000.0, grad=grad.copy(), forw=None)
        actual = torch_processor.forward_torch(
            nx=4,
            nz=4,
            vmax=torch.tensor(3000.0),
            grad=torch.tensor(grad),
            forw=None,
        ).cpu().numpy()

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

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
