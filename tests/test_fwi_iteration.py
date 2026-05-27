import unittest

import numpy as np
import torch

from ADFWI.fwi.iteration.epoch import apply_epoch_update_step
from ADFWI.fwi.iteration.loss import (
    apply_acoustic_batch_loss_step,
    apply_batch_loss_step,
    apply_elastic_batch_loss_step,
    build_batch_loss,
)
from ADFWI.fwi.iteration.progress import finalize_epoch_progress, set_batch_description
from ADFWI.fwi.iteration.batches import iter_batch_ranges


class DummyProgressBar:
    def __init__(self):
        self.description = None

    def set_description(self, value):
        self.description = value


class DummyOptimizer:
    def __init__(self):
        self.calls = []

    def step(self, closure=None):
        self.calls.append("optimizer.step")
        if closure is None:
            return "step-result"
        return closure()


class DummyScheduler:
    def __init__(self, call_log):
        self.call_log = call_log

    def step(self):
        self.call_log.append("scheduler.step")


class DummyModel:
    def __init__(self, call_log):
        self.call_log = call_log

    def forward(self):
        self.call_log.append("model.forward")


class DummyApplyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, synthetic, observed):
        ctx.save_for_backward(synthetic, observed)
        return torch.sum((synthetic - observed) ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        synthetic, observed = ctx.saved_tensors
        return grad_output * 2.0 * (synthetic - observed), None


class DummyCallableLoss:
    def __call__(self, synthetic, observed):
        return torch.sum((synthetic - observed) ** 2)


class DummyAcousticPropagator:
    def __init__(self, synthetic, forward_wavefield):
        self.synthetic = synthetic
        self.forward_wavefield = forward_wavefield
        self.dt = 0.002
        self.calls = []

    def forward(self, *, shot_index, checkpoint_segments):
        self.calls.append((shot_index, checkpoint_segments))
        return {
            "p": self.synthetic,
            "forward_wavefield_p": self.forward_wavefield,
        }


class DummyElasticPropagator:
    def __init__(self, record):
        self.record = record
        self.dt = 0.004
        self.calls = []

    def forward(self, *, fd_order, shot_index, checkpoint_segments):
        self.calls.append((fd_order, shot_index, checkpoint_segments))
        return self.record


class TestFWIIterationHelpers(unittest.TestCase):
    def test_none_batch_size_uses_full_batch(self):
        batches = list(iter_batch_ranges(5, None))

        self.assertEqual(len(batches), 1)
        self.assertEqual((batches[0].batch, batches[0].begin, batches[0].end), (0, 0, 5))
        np.testing.assert_array_equal(batches[0].shot_index, np.arange(5))

    def test_batch_size_larger_than_shots_uses_full_batch(self):
        batches = list(iter_batch_ranges(3, 10))

        self.assertEqual(len(batches), 1)
        self.assertEqual((batches[0].begin, batches[0].end), (0, 3))
        np.testing.assert_array_equal(batches[0].shot_index, np.arange(3))

    def test_partial_last_batch_matches_legacy_ranges(self):
        batches = list(iter_batch_ranges(5, 2))

        self.assertEqual([(b.batch, b.begin, b.end) for b in batches], [(0, 0, 2), (1, 2, 4), (2, 4, 5)])
        np.testing.assert_array_equal(batches[0].shot_index, np.arange(0, 2))
        np.testing.assert_array_equal(batches[1].shot_index, np.arange(2, 4))
        np.testing.assert_array_equal(batches[2].shot_index, np.arange(4, 5))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            list(iter_batch_ranges(0, 1))
        with self.assertRaises(ValueError):
            list(iter_batch_ranges(5, 0))
        with self.assertRaises(ValueError):
            list(iter_batch_ranges(5, -1))

    def test_build_batch_loss_without_regularization_keeps_data_loss(self):
        data_loss = torch.tensor(2.0, requires_grad=True)

        batch_loss = build_batch_loss(data_loss)
        batch_loss.tensor.backward()

        self.assertIs(batch_loss.tensor, data_loss)
        self.assertEqual(batch_loss.scalar, 2.0)
        self.assertEqual(float(data_loss.grad.item()), 1.0)

    def test_build_batch_loss_with_regularization_matches_expanded_sum(self):
        data_loss = torch.tensor(2.0, requires_grad=True)
        regularization_loss = torch.tensor(3.0, requires_grad=True)

        batch_loss = build_batch_loss(data_loss, regularization_loss)
        batch_loss.tensor.backward()

        self.assertEqual(float(batch_loss.tensor.detach().item()), 5.0)
        self.assertEqual(batch_loss.scalar, 5.0)
        self.assertEqual(float(data_loss.grad.item()), 1.0)
        self.assertEqual(float(regularization_loss.grad.item()), 1.0)

    def test_apply_batch_loss_step_runs_backward_accumulates_scalar_and_updates_progress(self):
        data_loss = torch.tensor(2.0, requires_grad=True)
        regularization_loss = torch.tensor(3.0, requires_grad=True)
        batch_range = list(iter_batch_ranges(5, None))[0]
        progress_bar = DummyProgressBar()

        epoch_loss = apply_batch_loss_step(
            10.0,
            data_loss,
            regularization_loss,
            progress_bar=progress_bar,
            batch_range=batch_range,
            batch_count=1,
        )

        self.assertEqual(epoch_loss, 15.0)
        self.assertEqual(float(data_loss.grad.item()), 1.0)
        self.assertEqual(float(regularization_loss.grad.item()), 1.0)
        self.assertEqual(progress_bar.description, "Shot:0 to 5")

    def test_apply_batch_loss_step_without_regularization_keeps_data_loss_gradient(self):
        data_loss = torch.tensor(2.0, requires_grad=True)

        epoch_loss = apply_batch_loss_step(4.0, data_loss)

        self.assertEqual(epoch_loss, 6.0)
        self.assertEqual(float(data_loss.grad.item()), 1.0)

    def test_apply_acoustic_batch_loss_step_runs_forward_loss_backward_and_wavefield_accumulation(self):
        synthetic = torch.tensor([[[1.0], [2.0]]], requires_grad=True)
        observed = torch.zeros((1, 2, 1))
        forward_wavefield = torch.tensor([[3.0, 4.0]])
        propagator = DummyAcousticPropagator(synthetic, forward_wavefield)
        regularization = torch.tensor(0.5, requires_grad=True)
        batch_range = list(iter_batch_ranges(1, None))[0]
        progress_bar = DummyProgressBar()
        prepare_calls = []

        def prepare_pair(synthetic_waveform, observed_waveform, *, shot_index, cutoff_freq, propagator_dt):
            prepare_calls.append((shot_index, cutoff_freq, propagator_dt))
            return synthetic_waveform, observed_waveform

        result = apply_acoustic_batch_loss_step(
            epoch_loss_scalar=2.0,
            accumulated_wavefield=None,
            propagator=propagator,
            batch_range=batch_range,
            checkpoint_segments=3,
            observed_pressure=observed,
            prepare_loss_pair=prepare_pair,
            loss_fn=DummyApplyLoss,
            normalization=False,
            cutoff_freq=8.0,
            regularization_loss_fn=lambda: regularization,
            progress_bar=progress_bar,
            batch_count=1,
            device=torch.device("cpu"),
        )

        self.assertEqual(result.epoch_loss_scalar, 7.5)
        self.assertTrue(torch.equal(synthetic.grad, torch.tensor([[[2.0], [4.0]]])))
        self.assertEqual(float(regularization.grad.item()), 1.0)
        np.testing.assert_array_equal(result.accumulated_wavefield, np.array([[3.0, 4.0]], dtype=np.float32))
        self.assertEqual(propagator.calls, [(batch_range.shot_index, 3)])
        self.assertEqual(prepare_calls, [(batch_range.shot_index, 8.0, 0.002)])
        self.assertEqual(progress_bar.description, "Shot:0 to 1")

    def test_apply_elastic_batch_loss_step_runs_component_losses_and_wavefield_accumulation(self):
        txx = torch.tensor([[[1.0]]], requires_grad=True)
        tzz = torch.tensor([[[2.0]]], requires_grad=True)
        vz = torch.tensor([[[4.0]]], requires_grad=True)
        record = {
            "txx": txx,
            "tzz": tzz,
            "vx": torch.tensor([[[3.0]]], requires_grad=True),
            "vz": vz,
            "forward_wavefield_txx": torch.tensor([[1.0, 2.0]]),
            "forward_wavefield_tzz": torch.tensor([[3.0, 4.0]]),
            "forward_wavefield_vx": torch.tensor([[5.0, 6.0]]),
            "forward_wavefield_vz": torch.tensor([[7.0, 8.0]]),
            "forward_wavefield_txz": torch.tensor([[9.0, 10.0]]),
        }
        observed_components = {
            "pressure": torch.zeros((1, 1, 1)),
            "vx": torch.zeros((1, 1, 1)),
            "vz": torch.zeros((1, 1, 1)),
        }
        propagator = DummyElasticPropagator(record)
        regularization = torch.tensor(0.5, requires_grad=True)
        batch_range = list(iter_batch_ranges(1, None))[0]
        progress_bar = DummyProgressBar()
        prepare_calls = []

        def prepare_pair(synthetic_waveform, observed_waveform, *, shot_index, cutoff_freq, propagator_dt):
            prepare_calls.append((shot_index, cutoff_freq, propagator_dt))
            return synthetic_waveform, observed_waveform

        result = apply_elastic_batch_loss_step(
            epoch_loss_scalar=2.0,
            accumulated_wavefields={},
            propagator=propagator,
            batch_range=batch_range,
            fd_order=6,
            checkpoint_segments=3,
            observed_components=observed_components,
            inversion_components=["pressure", "vz"],
            component_weights={"pressure": 2.0, "vz": 0.5},
            prepare_loss_pair=prepare_pair,
            loss_fn=DummyCallableLoss(),
            normalization=False,
            cutoff_freq=8.0,
            regularization_loss_fn=lambda: regularization,
            progress_bar=progress_bar,
            batch_count=1,
            device=torch.device("cpu"),
        )

        self.assertEqual(result.epoch_loss_scalar, 28.5)
        self.assertTrue(torch.equal(txx.grad, torch.tensor([[[12.0]]])))
        self.assertTrue(torch.equal(tzz.grad, torch.tensor([[[12.0]]])))
        self.assertTrue(torch.equal(vz.grad, torch.tensor([[[4.0]]])))
        self.assertEqual(float(regularization.grad.item()), 1.0)
        self.assertEqual(propagator.calls, [(6, batch_range.shot_index, 3)])
        self.assertEqual(prepare_calls, [(batch_range.shot_index, 8.0, 0.004), (batch_range.shot_index, 8.0, 0.004)])
        np.testing.assert_array_equal(result.accumulated_wavefields["pressure"], np.array([[-4.0, -6.0]], dtype=np.float32))
        np.testing.assert_array_equal(result.accumulated_wavefields["vz"], np.array([[7.0, 8.0]], dtype=np.float32))
        self.assertEqual(progress_bar.description, "Shot:0 to 1")

    def test_apply_epoch_update_step_preserves_update_order_without_closure(self):
        call_log = []
        optimizer = DummyOptimizer()
        scheduler = DummyScheduler(call_log)
        model = DummyModel(call_log)

        result = apply_epoch_update_step(optimizer, scheduler, model)

        self.assertEqual(result, "step-result")
        self.assertEqual(optimizer.calls, ["optimizer.step"])
        self.assertEqual(call_log, ["scheduler.step", "model.forward"])

    def test_apply_epoch_update_step_passes_closure_and_returns_optimizer_result(self):
        call_log = []
        optimizer = DummyOptimizer()
        scheduler = DummyScheduler(call_log)
        model = DummyModel(call_log)

        def closure():
            call_log.append("closure")
            return 7.0

        result = apply_epoch_update_step(optimizer, scheduler, model, closure=closure)

        self.assertEqual(result, 7.0)
        self.assertEqual(optimizer.calls, ["optimizer.step"])
        self.assertEqual(call_log, ["closure", "scheduler.step", "model.forward"])

    def test_finalize_epoch_progress_runs_cache_callback_and_sets_label(self):
        progress_bar = DummyProgressBar()
        calls = []

        def cache_callback(*, epoch_id, loss_epoch):
            calls.append((epoch_id, loss_epoch))

        finalize_epoch_progress(
            progress_bar,
            epoch_id=2,
            loss_epoch=3.25,
            cache_result=True,
            cache_callback=cache_callback,
        )

        self.assertEqual(calls, [(2, 3.25)])
        self.assertEqual(progress_bar.description, "Iter:3,Loss:3.25")

    def test_finalize_epoch_progress_skips_cache_when_disabled(self):
        progress_bar = DummyProgressBar()
        calls = []

        def cache_callback(*, epoch_id, loss_epoch):
            calls.append((epoch_id, loss_epoch))

        finalize_epoch_progress(
            progress_bar,
            epoch_id=0,
            loss_epoch=1.0,
            cache_result=False,
            cache_callback=cache_callback,
        )

        self.assertEqual(calls, [])
        self.assertEqual(progress_bar.description, "Iter:1,Loss:1.0")

    def test_set_batch_description_single_batch_matches_legacy_label(self):
        batch_range = list(iter_batch_ranges(5, None))[0]
        progress_bar = DummyProgressBar()

        set_batch_description(progress_bar, batch_range, 1)

        self.assertEqual(progress_bar.description, "Shot:0 to 5")

    def test_set_batch_description_multi_batch_keeps_description_unchanged(self):
        batch_range = list(iter_batch_ranges(5, 2))[0]
        progress_bar = DummyProgressBar()

        set_batch_description(progress_bar, batch_range, 3)

        self.assertIsNone(progress_bar.description)


if __name__ == "__main__":
    unittest.main()
