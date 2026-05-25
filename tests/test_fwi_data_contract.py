import unittest

import torch

from ADFWI.fwi.data import (
    ELASTIC_COMPONENTS,
    build_fwi_data_transform_pipeline,
    build_fwi_transform_context,
    build_transform_context,
    elastic_observed_components,
    elastic_pressure,
    elastic_synthetic_components,
    normalize_elastic_component_weights,
    prepare_loss_pair,
)
from ADFWI.fwi.transforms import DataMask, DataTransformPipeline, LegacyLateWindowMute, LegacyLowPassFilter, LegacyOffsetMute, TraceNormalize
from ADFWI.fwi.transforms.receivers import select_or_mask_receivers


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
