import unittest

import torch

from ADFWI.fwi.data.components import (
    ELASTIC_COMPONENTS,
    elastic_component_loss_inputs,
    elastic_observed_components,
    elastic_pressure,
    elastic_synthetic_components,
    normalize_elastic_component_weights,
)
from ADFWI.fwi.data.inputs import LossInput, acoustic_pressure_loss_input, elastic_loss_inputs
from ADFWI.fwi.data.loss import evaluate_loss_inputs, evaluate_misfit_loss, sum_weighted_losses
from ADFWI.fwi.data.pipeline import build_fwi_data_transform_pipeline
from ADFWI.fwi.data.preparation import build_fwi_transform_context, build_transform_context, prepare_fwi_loss_pair, prepare_loss_pair
from ADFWI.fwi.misfit import Misfit
from ADFWI.fwi.transforms import DataMask, DataTransformPipeline, LegacyLateWindowMute, LegacyLowPassFilter, LegacyOffsetMute, TraceNormalize
from ADFWI.fwi.transforms.amplitude import normalize_waveform
from ADFWI.fwi.transforms.receivers import select_or_mask_receivers


class DummyMisfit(Misfit):
    def forward(self, synthetic, observed):
        return torch.sum(synthetic - observed)


class DummyCallableLoss:
    def __call__(self, synthetic, observed):
        return torch.sum((synthetic - observed) ** 2)


class DummyApplyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, synthetic, observed):
        return torch.sum(torch.abs(synthetic - observed))


class FWIDataContractTests(unittest.TestCase):
    def test_build_fwi_data_transform_pipeline_adds_legacy_transforms_and_normalize(self):
        pipeline, waveform_normalize = build_fwi_data_transform_pipeline(None, True)

        self.assertFalse(waveform_normalize)
        self.assertIsInstance(pipeline, DataTransformPipeline)
        self.assertIsInstance(pipeline.transforms[0], LegacyOffsetMute)
        self.assertIsInstance(pipeline.transforms[1], LegacyLateWindowMute)
        self.assertIsInstance(pipeline.transforms[2], LegacyLowPassFilter)
        self.assertIsInstance(pipeline.transforms[3], DataMask)
        self.assertIsInstance(pipeline.transforms[4], TraceNormalize)

    def test_sum_weighted_losses_preserves_autograd(self):
        first = torch.tensor(1.5, requires_grad=True)
        second = torch.tensor(2.5, requires_grad=True)

        total = sum_weighted_losses([first * 2.0, second * 3.0])
        total.backward()

        self.assertTrue(torch.equal(total.detach(), torch.tensor(10.5)))
        self.assertEqual(float(first.grad), 2.0)
        self.assertEqual(float(second.grad), 3.0)

    def test_sum_weighted_losses_empty_uses_requested_device(self):
        total = sum_weighted_losses([], device=torch.device("cpu"))

        self.assertEqual(total.device.type, "cpu")
        self.assertEqual(float(total.item()), 0.0)

    def test_evaluate_misfit_loss_uses_misfit_forward(self):
        synthetic = torch.tensor([1.0, 3.0])
        observed = torch.tensor([0.5, 1.0])

        loss = evaluate_misfit_loss(DummyMisfit(), synthetic, observed)

        self.assertTrue(torch.equal(loss, torch.tensor(2.5)))

    def test_evaluate_misfit_loss_supports_callable_fallback(self):
        synthetic = torch.tensor([1.0, 3.0])
        observed = torch.tensor([0.5, 1.0])

        loss = evaluate_misfit_loss(DummyCallableLoss(), synthetic, observed, function_fallback="call")

        self.assertTrue(torch.equal(loss, torch.tensor(4.25)))

    def test_evaluate_misfit_loss_supports_apply_fallback(self):
        synthetic = torch.tensor([1.0, 3.0])
        observed = torch.tensor([0.5, 1.0])

        loss = evaluate_misfit_loss(DummyApplyLoss, synthetic, observed, function_fallback="apply")

        self.assertTrue(torch.equal(loss, torch.tensor(2.5)))

    def test_evaluate_loss_inputs_prepares_weights_and_preserves_autograd(self):
        first = torch.tensor([1.0, 2.0], requires_grad=True)
        second = torch.tensor([3.0], requires_grad=True)
        observed = torch.zeros(2)
        calls = []

        def prepare_pair(synthetic, observed_waveform, *, shot_index, cutoff_freq, propagator_dt):
            calls.append((synthetic, observed_waveform, shot_index, cutoff_freq, propagator_dt))
            return synthetic, observed_waveform

        result = evaluate_loss_inputs(
            [
                LossInput("pressure", first, observed, shot_index="shot-a", weight=2.0),
                LossInput("vz", second, torch.zeros(1), shot_index="shot-b", weight=0.5),
            ],
            prepare_loss_pair=prepare_pair,
            loss_fn=DummyCallableLoss(),
            normalization=False,
            function_fallback="call",
            cutoff_freq=8.0,
            propagator_dt=0.002,
            device=torch.device("cpu"),
        )
        result.data_loss.backward()

        self.assertEqual([item.component for item in result.component_losses], ["pressure", "vz"])
        self.assertEqual([item.weight for item in result.component_losses], [2.0, 0.5])
        self.assertTrue(torch.equal(result.data_loss.detach(), torch.tensor(14.5)))
        self.assertTrue(torch.equal(first.grad, torch.tensor([4.0, 8.0])))
        self.assertTrue(torch.equal(second.grad, torch.tensor([3.0])))
        self.assertEqual(calls[0][2:], ("shot-a", 8.0, 0.002))
        self.assertEqual(calls[1][2:], ("shot-b", 8.0, 0.002))

    def test_evaluate_loss_inputs_applies_legacy_normalization(self):
        synthetic = torch.tensor([[[2.0], [4.0]]])
        observed = torch.zeros_like(synthetic)

        result = evaluate_loss_inputs(
            [LossInput("pressure", synthetic, observed, shot_index=None)],
            prepare_loss_pair=lambda synthetic, observed, **_: (synthetic, observed),
            loss_fn=DummyCallableLoss(),
            normalization=True,
            function_fallback="call",
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(result.data_loss, torch.tensor(1.25)))

    def test_normalize_waveform_matches_legacy_trace_normalization(self):
        data = torch.tensor(
            [
                [[0.0, 2.0], [2.0, -4.0], [-1.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )

        normalized = normalize_waveform(data)

        expected = torch.tensor(
            [
                [[0.0, 0.5], [1.0, -1.0], [-0.5, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        self.assertTrue(torch.equal(normalized, expected))
        self.assertFalse(torch.isnan(normalized).any())

    def test_build_fwi_data_transform_pipeline_appends_custom_pipeline(self):
        custom_pipeline = DataTransformPipeline([TraceNormalize()])

        pipeline, waveform_normalize = build_fwi_data_transform_pipeline(custom_pipeline, True)

        self.assertTrue(waveform_normalize)
        self.assertIsInstance(pipeline.transforms[0], LegacyOffsetMute)
        self.assertIsInstance(pipeline.transforms[1], LegacyLateWindowMute)
        self.assertIsInstance(pipeline.transforms[2], LegacyLowPassFilter)
        self.assertIsInstance(pipeline.transforms[3], DataMask)
        self.assertIs(pipeline.transforms[4], custom_pipeline)

    def test_build_fwi_transform_context_selects_shot_scoped_values(self):
        shot_index = torch.tensor([0, 2])
        receiver_masks = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        src_x = torch.tensor([10.0, 20.0, 30.0])
        rcv_x = torch.tensor([100.0, 110.0])
        data_masks = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)

        context = build_fwi_transform_context(
            shot_index=shot_index,
            cutoff_freq=8.0,
            propagator_dt=None,
            default_dt=0.002,
            late_window=0.1,
            offset_mute_threshold=250.0,
            dx=10.0,
            receiver_masks_2d=receiver_masks,
            src_x=src_x,
            rcv_x=rcv_x,
            data_masks=data_masks,
        )

        self.assertIs(context["shot_index"], shot_index)
        self.assertEqual(context["cutoff_freq"], 8.0)
        self.assertEqual(context["dt"], 0.002)
        self.assertTrue(torch.equal(context["receiver_mask"], receiver_masks[shot_index]))
        self.assertTrue(torch.equal(context["src_x"], src_x[shot_index]))
        self.assertTrue(torch.equal(context["rcv_x"], rcv_x))
        self.assertTrue(torch.equal(context["data_mask"], data_masks[shot_index]))

    def test_build_fwi_transform_context_omits_shot_scoped_values_without_shot_index(self):
        context = build_fwi_transform_context(
            shot_index=None,
            cutoff_freq=None,
            propagator_dt=0.003,
            default_dt=0.002,
            late_window=None,
            offset_mute_threshold=None,
            dx=10.0,
            receiver_masks_2d=torch.ones((1, 2)),
            src_x=torch.ones(1),
            rcv_x=torch.ones(2),
            data_masks=torch.ones((1, 4, 2)),
        )

        self.assertEqual(context["dt"], 0.003)
        self.assertNotIn("receiver_mask", context)
        self.assertNotIn("src_x", context)
        self.assertNotIn("rcv_x", context)
        self.assertNotIn("data_mask", context)

    def test_build_fwi_transform_context_prefers_propagator_dt(self):
        context = build_fwi_transform_context(
            shot_index=torch.tensor([0]),
            cutoff_freq=None,
            propagator_dt=0.003,
            default_dt=0.002,
        )

        self.assertEqual(context["dt"], 0.003)

    def test_acoustic_pressure_loss_input_selects_observed_shots(self):
        shot_index = torch.tensor([0, 2])
        record = {"p": torch.full((2, 3, 2), 1.0)}
        observed = torch.arange(3 * 3 * 2, dtype=torch.float32).reshape(3, 3, 2)

        loss_input = acoustic_pressure_loss_input(record, observed, shot_index)

        self.assertIsInstance(loss_input, LossInput)
        self.assertEqual(loss_input.component, "pressure")
        self.assertIs(loss_input.synthetic, record["p"])
        self.assertTrue(torch.equal(loss_input.observed, observed[shot_index]))
        self.assertIs(loss_input.shot_index, shot_index)
        self.assertEqual(loss_input.weight, 1.0)

    def test_elastic_loss_inputs_select_active_components_and_observed_shots(self):
        shot_index = torch.tensor([1])
        record = {
            "txx": torch.full((1, 2, 2), 1.0),
            "tzz": torch.full((1, 2, 2), 2.0),
            "vx": torch.full((1, 2, 2), 3.0),
            "vz": torch.full((1, 2, 2), 4.0),
        }
        observed_components = {
            "pressure": torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2),
            "vx": torch.full((3, 2, 2), 5.0),
            "vz": torch.full((3, 2, 2), 6.0),
        }

        inputs = elastic_loss_inputs(record, observed_components, ["vz", "pressure"], {"pressure": 2.0, "vz": 0.5}, shot_index)

        self.assertEqual([item.component for item in inputs], ["pressure", "vz"])
        self.assertTrue(torch.equal(inputs[0].synthetic, torch.full((1, 2, 2), -3.0)))
        self.assertTrue(torch.equal(inputs[0].observed, observed_components["pressure"][shot_index]))
        self.assertEqual(inputs[0].weight, 2.0)
        self.assertIs(inputs[0].shot_index, shot_index)
        self.assertIs(inputs[1].synthetic, record["vz"])
        self.assertTrue(torch.equal(inputs[1].observed, observed_components["vz"][shot_index]))
        self.assertEqual(inputs[1].weight, 0.5)

    def test_prepare_fwi_loss_pair_matches_manual_context_path(self):
        shot_index = torch.tensor([0, 1])
        synthetic = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        observed = torch.ones((2, 4, 2), dtype=torch.float32)
        receiver_masks = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        data_masks = torch.ones_like(observed)
        data_masks[:, 0, :] = 0
        pipeline = DataTransformPipeline([DataMask(apply_to="synthetic")])

        actual_syn, actual_obs = prepare_fwi_loss_pair(
            synthetic,
            observed,
            shot_index=shot_index,
            default_dt=0.002,
            receiver_masks_2d=receiver_masks,
            data_masks=data_masks,
            data_transform_pipeline=pipeline,
        )

        context = build_fwi_transform_context(
            shot_index=shot_index,
            default_dt=0.002,
            receiver_masks_2d=receiver_masks,
            data_masks=data_masks,
        )
        expected_syn, expected_obs = prepare_loss_pair(
            synthetic,
            observed,
            receiver_mask=context["receiver_mask"],
            data_transform_pipeline=pipeline,
            context=context,
        )
        self.assertTrue(torch.equal(actual_syn, expected_syn))
        self.assertTrue(torch.equal(actual_obs, expected_obs))

    def test_prepare_loss_pair_selects_receivers_before_pipeline(self):
        synthetic = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
        observed = torch.ones((1, 4, 2), dtype=torch.float32)
        receiver_mask = torch.tensor([[1, 0, 1]], dtype=torch.float32)
        data_mask = torch.ones_like(observed)
        data_mask[:, 0, :] = 0
        pipeline = DataTransformPipeline([DataMask(apply_to="synthetic")])
        context = build_transform_context(receiver_mask=receiver_mask, data_mask=data_mask)

        actual_syn, actual_obs = prepare_loss_pair(
            synthetic,
            observed,
            receiver_mask=receiver_mask,
            data_transform_pipeline=pipeline,
            context=context,
        )

        expected_syn = select_or_mask_receivers(synthetic, observed, receiver_mask) * data_mask
        self.assertTrue(torch.equal(actual_syn, expected_syn))
        self.assertTrue(torch.equal(actual_obs, observed))
        self.assertEqual(actual_syn.shape, observed.shape)

    def test_prepare_loss_pair_can_run_pipeline_without_receiver_selection(self):
        synthetic = torch.tensor([[[1.0, 2.0], [2.0, 4.0]]])
        observed = synthetic * 2
        pipeline = DataTransformPipeline([TraceNormalize()])

        actual_syn, actual_obs = prepare_loss_pair(synthetic, observed, data_transform_pipeline=pipeline)
        expected_syn, expected_obs = pipeline(synthetic, observed)

        self.assertTrue(torch.equal(actual_syn, expected_syn))
        self.assertTrue(torch.equal(actual_obs, expected_obs))

    def test_build_transform_context_omits_optional_none_values(self):
        context = build_transform_context(shot_index=[0, 1], cutoff_freq=5.0, dt=0.001, receiver_mask=None)

        self.assertEqual(context["shot_index"], [0, 1])
        self.assertEqual(context["cutoff_freq"], 5.0)
        self.assertEqual(context["dt"], 0.001)
        self.assertNotIn("receiver_mask", context)
        self.assertNotIn("data_mask", context)

    def test_elastic_pressure_is_negative_stress_sum(self):
        txx = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
        tzz = torch.tensor([[[0.5, 2.0], [-1.0, 6.0]]])

        pressure = elastic_pressure(txx, tzz)

        self.assertTrue(torch.equal(pressure, -(txx + tzz)))

    def test_elastic_synthetic_components_use_stable_names(self):
        record = {
            "txx": torch.ones((1, 2, 2)),
            "tzz": torch.full((1, 2, 2), 2.0),
            "vx": torch.full((1, 2, 2), 3.0),
            "vz": torch.full((1, 2, 2), 4.0),
        }

        components = elastic_synthetic_components(record)

        self.assertEqual(tuple(components.keys()), ELASTIC_COMPONENTS)
        self.assertTrue(torch.equal(components["pressure"], torch.full((1, 2, 2), -3.0)))
        self.assertTrue(torch.equal(components["vx"], record["vx"]))
        self.assertTrue(torch.equal(components["vz"], record["vz"]))

    def test_elastic_component_loss_inputs_filters_active_components_in_stable_order(self):
        synthetic_components = {
            "pressure": torch.full((1, 2, 2), 1.0),
            "vx": torch.full((1, 2, 2), 2.0),
            "vz": torch.full((1, 2, 2), 3.0),
        }
        observed_components = {
            "pressure": torch.full((1, 2, 2), 4.0),
            "vx": torch.full((1, 2, 2), 5.0),
            "vz": torch.full((1, 2, 2), 6.0),
        }
        weights = {"pressure": 1.0, "vx": 2.0, "vz": 3.0}

        inputs = elastic_component_loss_inputs(
            synthetic_components,
            observed_components,
            ["vz", "pressure"],
            weights,
        )

        self.assertEqual([item[0] for item in inputs], ["pressure", "vz"])
        self.assertIs(inputs[0][1], synthetic_components["pressure"])
        self.assertIs(inputs[0][2], observed_components["pressure"])
        self.assertEqual(inputs[0][3], 1.0)
        self.assertIs(inputs[1][1], synthetic_components["vz"])
        self.assertIs(inputs[1][2], observed_components["vz"])
        self.assertEqual(inputs[1][3], 3.0)

    def test_elastic_component_loss_inputs_returns_empty_for_no_active_components(self):
        synthetic_components = {component: torch.ones((1, 1, 1)) for component in ELASTIC_COMPONENTS}
        observed_components = {component: torch.ones((1, 1, 1)) for component in ELASTIC_COMPONENTS}
        weights = {component: 1.0 for component in ELASTIC_COMPONENTS}

        inputs = elastic_component_loss_inputs(synthetic_components, observed_components, [], weights)

        self.assertEqual(inputs, [])

    def test_elastic_observed_components_match_legacy_pressure_rule(self):
        data = {
            "txx": torch.tensor([[[1.0, 2.0]]]),
            "tzz": torch.tensor([[[3.0, 4.0]]]),
            "vx": torch.tensor([[[5.0, 6.0]]]),
            "vz": torch.tensor([[[7.0, 8.0]]]),
        }

        components = elastic_observed_components(data)

        self.assertEqual(tuple(components.keys()), ELASTIC_COMPONENTS)
        self.assertTrue(torch.equal(components["pressure"], -(data["txx"] + data["tzz"])))
        self.assertTrue(torch.equal(components["vx"], data["vx"]))
        self.assertTrue(torch.equal(components["vz"], data["vz"]))

    def test_normalize_elastic_component_weights_defaults_active_components_to_one(self):
        weights = normalize_elastic_component_weights(["pressure", "vx"], None)

        self.assertEqual(weights, {"pressure": 1.0, "vx": 1.0})

    def test_normalize_elastic_component_weights_uses_explicit_values(self):
        weights = normalize_elastic_component_weights(["pressure", "vx", "vz"], {"pressure": 2.0, "vz": 0.25})

        self.assertEqual(weights, {"pressure": 2.0, "vx": 1.0, "vz": 0.25})

    def test_normalize_elastic_component_weights_rejects_unknown_names(self):
        with self.assertRaises(ValueError):
            normalize_elastic_component_weights(["pressure", "ux"], None)
        with self.assertRaises(ValueError):
            normalize_elastic_component_weights(["pressure"], {"ux": 1.0})

    def test_normalize_elastic_component_weights_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            normalize_elastic_component_weights(["pressure"], {"pressure": -1.0})


if __name__ == "__main__":
    unittest.main()
