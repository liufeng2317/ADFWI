import unittest

import torch

import ADFWI
from ADFWI.backends import (
    BackendUnavailableError,
    backend,
    backend_diagnostics,
    configure_backend,
    get_backend,
    resolve_backend,
    set_backend,
    use_backend,
)


class BackendTests(unittest.TestCase):
    def tearDown(self):
        configure_backend("cpu")

    def test_cpu_backend_is_available(self):
        backend = configure_backend("cpu")
        self.assertEqual(backend.name, "cpu")
        self.assertEqual(str(backend.device), "cpu")
        self.assertFalse(backend.fallback)
        self.assertEqual(get_backend().name, "cpu")



    def test_auto_select_prefers_npu_when_available(self):
        npu = getattr(torch, "npu", None)
        is_available = getattr(npu, "is_available", None)
        if not (callable(is_available) and is_available()):
            self.skipTest("NPU is not available on this machine")
        backend = configure_backend(None)
        self.assertEqual(backend.name, "npu")
        self.assertEqual(backend.index, 0)


    def test_auto_select_priority_is_user_configurable(self):
        backend = configure_backend(None, prefer=("cpu", "npu"))
        self.assertEqual(backend.name, "cpu")

        backend = configure_backend(None, prefer=("npu", "cpu"))
        npu = getattr(torch, "npu", None)
        is_available = getattr(npu, "is_available", None)
        expected = "npu" if callable(is_available) and is_available() else "cpu"
        self.assertEqual(backend.name, expected)

    def test_auto_select_can_be_forced_to_cpu(self):
        backend = configure_backend(None, prefer=("cpu",))
        self.assertEqual(backend.name, "cpu")
        self.assertEqual(str(backend.device), "cpu")

    def test_tensor_factory_uses_backend_device_and_dtype(self):
        backend = configure_backend("cpu", dtype=torch.float64)
        tensor = backend.zeros((2, 3))
        self.assertEqual(tensor.device.type, "cpu")
        self.assertEqual(tensor.dtype, torch.float64)

    def test_explicit_unavailable_backend_can_fallback(self):
        backend = resolve_backend("npu:0", fallback=True)
        if backend.name == "cpu":
            self.assertTrue(backend.fallback)
            self.assertFalse(backend.available)
        else:
            self.assertEqual(backend.name, "npu")

    def test_explicit_unavailable_backend_raises_without_fallback(self):
        npu = getattr(torch, "npu", None)
        is_available = getattr(npu, "is_available", None)
        if callable(is_available) and is_available():
            self.skipTest("NPU is available on this machine")
        with self.assertRaises(BackendUnavailableError):
            resolve_backend("npu:0", fallback=False)

    def test_scoped_backend_restores_previous_backend(self):
        configure_backend("cpu")
        original = get_backend()
        with use_backend("cpu") as scoped:
            self.assertEqual(scoped.name, "cpu")
            self.assertEqual(get_backend().name, "cpu")
        self.assertEqual(get_backend(), original)

    def test_explicit_override_does_not_change_global_backend(self):
        configure_backend("cpu")
        override = get_backend(device="cpu")
        self.assertEqual(override.name, "cpu")
        self.assertEqual(get_backend().name, "cpu")

    def test_string_dtype_is_supported(self):
        backend_obj = configure_backend("cpu", dtype="float64")
        self.assertEqual(backend_obj.dtype, torch.float64)
        self.assertEqual(get_backend(dtype="float32").dtype, torch.float32)

    def test_string_prefer_is_supported(self):
        backend_obj = configure_backend(None, prefer="cpu,npu")
        self.assertEqual(backend_obj.name, "cpu")

    def test_user_facing_backend_aliases(self):
        backend_obj = set_backend("cpu", dtype="float64")
        self.assertEqual(backend_obj.dtype, torch.float64)
        self.assertEqual(backend().dtype, torch.float64)
        diagnostics = backend_diagnostics()
        self.assertEqual(diagnostics["name"], "cpu")
        self.assertEqual(diagnostics["dtype"], "float64")

    def test_top_level_adfwi_backend_api(self):
        backend_obj = ADFWI.set_backend("cpu", dtype="float32")
        self.assertEqual(backend_obj.name, "cpu")
        self.assertEqual(ADFWI.backend().device.type, "cpu")
        self.assertEqual(ADFWI.backend_diagnostics()["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
