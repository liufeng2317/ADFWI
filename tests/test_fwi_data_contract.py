import unittest

import torch

from ADFWI.fwi.data import (
    ELASTIC_COMPONENTS,
    build_transform_context,
    elastic_observed_components,
    elastic_pressure,
    elastic_synthetic_components,
    normalize_elastic_component_weights,
    prepare_loss_pair,
)
from ADFWI.fwi.transforms import DataMask, DataTransformPipeline, TraceNormalize
from ADFWI.fwi.transforms.receivers import select_or_mask_receivers


class FWIDataContractTests(unittest.TestCase):
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
