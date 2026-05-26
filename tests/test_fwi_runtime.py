import unittest
from types import SimpleNamespace

import numpy as np
import torch

from ADFWI.fwi.runtime import (
    acoustic_pressure_waveforms,
    accumulate_named_wavefields,
    accumulate_wavefield,
    elastic_gradient_wavefields,
    align_regularization_backend,
    append_epoch_loss,
    append_model_snapshots,
    append_required_gradient_snapshots,
    calculate_regularization_loss,
    process_named_parameter_gradients,
    process_parameter_gradient,
    select_elastic_gradient_wavefield,
    should_cache_epoch,
    snapshot_model_parameters,
    tensor_to_numpy,
    validate_model_propagator_devices,
    wavefield_to_numpy,
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

    def test_cache_helpers_record_loss_and_model_snapshots(self):
        owner = SimpleNamespace(iter_loss=[], iter_vp=[], iter_rho=[], cache_iter_index=[])
        model = SimpleNamespace(
            vp=torch.tensor([[1.0, 2.0]], requires_grad=True),
            rho=torch.tensor([[3.0, 4.0]], requires_grad=True),
        )

        append_epoch_loss(owner, 12.5)
        snapshots = snapshot_model_parameters(model, ["vp", "missing", "rho"])
        append_model_snapshots(owner, snapshots, epoch_id=4)

        self.assertEqual(owner.iter_loss, [12.5])
        self.assertEqual(owner.cache_iter_index, [4])
        self.assertEqual(set(snapshots), {"vp", "rho"})
        self.assertTrue(torch.equal(torch.tensor(owner.iter_vp[0]), torch.tensor([[1.0, 2.0]])))
        self.assertTrue(torch.equal(torch.tensor(owner.iter_rho[0]), torch.tensor([[3.0, 4.0]])))
        self.assertTrue(should_cache_epoch(4, 2))
        self.assertFalse(should_cache_epoch(5, 2))

    def test_cache_helpers_record_only_trainable_gradients(self):
        owner = SimpleNamespace(iter_vp_grad=[], iter_rho_grad=[])
        vp = torch.tensor([[1.0, 2.0]], requires_grad=True)
        rho = torch.tensor([[3.0, 4.0]], requires_grad=True)
        vp.grad = torch.tensor([[0.1, 0.2]])
        rho.grad = torch.tensor([[0.3, 0.4]])
        model = SimpleNamespace(
            vp=vp,
            rho=rho,
            get_requires_grad=lambda name: name == "vp",
        )

        snapshots = append_required_gradient_snapshots(owner, model, ["vp", "rho"])

        self.assertEqual(set(snapshots), {"vp"})
        self.assertEqual(len(owner.iter_vp_grad), 1)
        self.assertEqual(owner.iter_rho_grad, [])
        self.assertTrue(torch.equal(torch.tensor(owner.iter_vp_grad[0]), torch.tensor([[0.1, 0.2]])))

    def test_tensor_to_numpy_returns_detached_cpu_snapshot(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)

        snapshot = tensor_to_numpy(tensor)
        tensor.data.add_(10.0)

        self.assertEqual(snapshot.tolist(), [1.0, 2.0])

    def test_acoustic_pressure_waveforms_selects_loss_and_gradient_inputs(self):
        pressure = torch.tensor([[1.0]])
        forward_pressure = torch.tensor([[2.0]])
        record = {
            "p": pressure,
            "u": torch.tensor([[3.0]]),
            "forward_wavefield_p": forward_pressure,
            "forward_wavefield_u": torch.tensor([[4.0]]),
        }

        rcv_p, forward_wavefield_p = acoustic_pressure_waveforms(record)

        self.assertIs(rcv_p, pressure)
        self.assertIs(forward_wavefield_p, forward_pressure)

    def test_elastic_gradient_wavefields_selects_configured_components(self):
        record = {
            "forward_wavefield_txx": torch.tensor([[1.0, 2.0]]),
            "forward_wavefield_tzz": torch.tensor([[3.0, 4.0]]),
            "forward_wavefield_vx": torch.tensor([[5.0, 6.0]]),
            "forward_wavefield_vz": torch.tensor([[7.0, 8.0]]),
            "forward_wavefield_txz": torch.tensor([[9.0, 10.0]]),
        }

        wavefields = elastic_gradient_wavefields(record, ["pressure", "vz"])

        self.assertEqual(set(wavefields), {"pressure", "vz"})
        self.assertTrue(torch.equal(wavefields["pressure"], torch.tensor([[-4.0, -6.0]])))
        self.assertIs(wavefields["vz"], record["forward_wavefield_vz"])

    def test_wavefield_helpers_accumulate_detached_numpy_arrays(self):
        first = torch.tensor([[1.0, 2.0]], requires_grad=True)
        second = torch.tensor([[3.0, 4.0]], requires_grad=True)

        accumulator = accumulate_wavefield(None, first)
        accumulator = accumulate_wavefield(accumulator, second)

        np.testing.assert_array_equal(accumulator, np.array([[4.0, 6.0]], dtype=np.float32))
        self.assertIsInstance(wavefield_to_numpy(first), np.ndarray)

    def test_wavefield_helpers_accumulate_named_elastic_components(self):
        accumulators = {}

        accumulate_named_wavefields(
            accumulators,
            {
                "pressure": torch.tensor([[1.0]]),
                "vz": torch.tensor([[2.0]]),
            },
        )
        accumulate_named_wavefields(
            accumulators,
            {
                "pressure": torch.tensor([[3.0]]),
                "vz": torch.tensor([[4.0]]),
            },
        )

        np.testing.assert_array_equal(accumulators["pressure"], np.array([[4.0]], dtype=np.float32))
        np.testing.assert_array_equal(accumulators["vz"], np.array([[6.0]], dtype=np.float32))

    def test_select_elastic_gradient_wavefield_preserves_legacy_priority(self):
        pressure = np.array([[1.0]])
        vz = np.array([[2.0]])
        vx = np.array([[3.0]])

        self.assertIs(select_elastic_gradient_wavefield({"pressure": pressure, "vz": vz}), pressure)
        self.assertIs(select_elastic_gradient_wavefield({"vz": vz, "vx": vx}), vz)
        self.assertIs(select_elastic_gradient_wavefield({"vx": vx}), vx)
        with self.assertRaisesRegex(ValueError, "no accumulated elastic wavefield"):
            select_elastic_gradient_wavefield({})

    def test_process_named_parameter_gradients_respects_requires_grad_gate_and_indices(self):
        model = SimpleNamespace(
            vp="vp-param",
            rho="rho-param",
            get_requires_grad=lambda name: name == "vp",
        )
        calls = []

        def process_gradient(parameter, *, forw, idx):
            calls.append((parameter, forw, idx))

        processed = process_named_parameter_gradients(
            model,
            [("vp", 0), ("rho", 1)],
            process_gradient,
            forw="forward-wavefield",
        )

        self.assertEqual(processed, ["vp"])
        self.assertEqual(calls, [("vp-param", "forward-wavefield", 0)])

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
