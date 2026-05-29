import unittest

import numpy as np
import torch

from ADFWI.utils import gpu2cpu, list2numpy, numpy2list, numpy2tensor, tensor2numpy


class UtilsConversionTests(unittest.TestCase):
    def test_numpy2tensor_converts_array_with_default_float32(self):
        array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        tensor = numpy2tensor(array)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertFalse(tensor.requires_grad)
        np.testing.assert_allclose(tensor.numpy(), array.astype(np.float32))

    def test_numpy2tensor_honors_requested_dtype(self):
        tensor = numpy2tensor([1, 2, 3], dtype=torch.float64)

        self.assertEqual(tensor.dtype, torch.float64)
        np.testing.assert_allclose(tensor.numpy(), np.array([1.0, 2.0, 3.0]))

    def test_numpy2tensor_returns_existing_tensor_unchanged(self):
        source = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        tensor = numpy2tensor(source, dtype=torch.float32)

        self.assertIs(tensor, source)
        self.assertEqual(tensor.dtype, torch.float64)
        self.assertTrue(tensor.requires_grad)

    def test_tensor2numpy_detaches_cpu_tensor(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)

        array = tensor2numpy(tensor)

        self.assertIsInstance(array, np.ndarray)
        np.testing.assert_allclose(array, np.array([1.0, 2.0], dtype=np.float32))

    def test_tensor2numpy_returns_non_tensor_unchanged(self):
        array = np.array([1.0, 2.0])

        result = tensor2numpy(array)

        self.assertIs(result, array)

    def test_gpu2cpu_converts_cpu_tensor_requiring_grad_to_numpy(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)

        array = gpu2cpu(tensor)

        self.assertIsInstance(array, np.ndarray)
        np.testing.assert_allclose(array, np.array([1.0, 2.0], dtype=np.float32))

    def test_gpu2cpu_converts_cpu_tensor_without_grad_to_numpy(self):
        tensor = torch.tensor([1.0, 2.0])

        array = gpu2cpu(tensor)

        self.assertIsInstance(array, np.ndarray)
        np.testing.assert_allclose(array, np.array([1.0, 2.0], dtype=np.float32))

    def test_gpu2cpu_returns_non_tensor_unchanged(self):
        value = [1, 2, 3]

        result = gpu2cpu(value)

        self.assertIs(result, value)

    def test_list2numpy_converts_python_list(self):
        array = list2numpy([[1, 2], [3, 4]])

        self.assertIsInstance(array, np.ndarray)
        np.testing.assert_array_equal(array, np.array([[1, 2], [3, 4]]))

    def test_list2numpy_returns_non_list_unchanged(self):
        array = np.array([1, 2])

        result = list2numpy(array)

        self.assertIs(result, array)

    def test_numpy2list_converts_numpy_array(self):
        result = numpy2list(np.array([[1, 2], [3, 4]]))

        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_numpy2list_returns_list_unchanged(self):
        value = [1, 2, 3]

        result = numpy2list(value)

        self.assertIs(result, value)


if __name__ == "__main__":
    unittest.main()
