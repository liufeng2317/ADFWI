import unittest

import ADFWI.fwi.data as data_api
from ADFWI.fwi.data import (
    ELASTIC_COMPONENTS,
    ComponentLoss,
    LossEvaluation,
    LossInput,
    acoustic_pressure_loss_input,
    build_fwi_data_transform_pipeline,
    build_fwi_transform_context,
    build_transform_context,
    elastic_component_loss_inputs,
    elastic_loss_inputs,
    elastic_observed_components,
    elastic_pressure,
    elastic_synthetic_components,
    evaluate_loss_inputs,
    evaluate_misfit_loss,
    normalize_elastic_component_weights,
    prepare_fwi_loss_pair,
    prepare_loss_pair,
    sum_weighted_losses,
)
from ADFWI.fwi.data import components, inputs, loss, pipeline, preparation


class FWIDataPublicAPITests(unittest.TestCase):
    def test_public_facade_exports_curated_data_contract_symbols(self):
        expected = {
            "ELASTIC_COMPONENTS",
            "ComponentLoss",
            "LossEvaluation",
            "LossInput",
            "acoustic_pressure_loss_input",
            "build_fwi_data_transform_pipeline",
            "build_fwi_transform_context",
            "build_transform_context",
            "elastic_component_loss_inputs",
            "elastic_loss_inputs",
            "elastic_observed_components",
            "elastic_pressure",
            "elastic_synthetic_components",
            "evaluate_loss_inputs",
            "evaluate_misfit_loss",
            "normalize_elastic_component_weights",
            "prepare_fwi_loss_pair",
            "prepare_loss_pair",
            "sum_weighted_losses",
        }

        self.assertEqual(set(data_api.__all__), expected)

    def test_public_facade_points_to_owner_module_objects(self):
        self.assertIs(ELASTIC_COMPONENTS, components.ELASTIC_COMPONENTS)
        self.assertIs(ComponentLoss, loss.ComponentLoss)
        self.assertIs(LossEvaluation, loss.LossEvaluation)
        self.assertIs(LossInput, inputs.LossInput)
        self.assertIs(acoustic_pressure_loss_input, inputs.acoustic_pressure_loss_input)
        self.assertIs(build_fwi_data_transform_pipeline, pipeline.build_fwi_data_transform_pipeline)
        self.assertIs(build_fwi_transform_context, preparation.build_fwi_transform_context)
        self.assertIs(build_transform_context, preparation.build_transform_context)
        self.assertIs(elastic_component_loss_inputs, components.elastic_component_loss_inputs)
        self.assertIs(elastic_loss_inputs, inputs.elastic_loss_inputs)
        self.assertIs(elastic_observed_components, components.elastic_observed_components)
        self.assertIs(elastic_pressure, components.elastic_pressure)
        self.assertIs(elastic_synthetic_components, components.elastic_synthetic_components)
        self.assertIs(evaluate_loss_inputs, loss.evaluate_loss_inputs)
        self.assertIs(evaluate_misfit_loss, loss.evaluate_misfit_loss)
        self.assertIs(normalize_elastic_component_weights, components.normalize_elastic_component_weights)
        self.assertIs(prepare_fwi_loss_pair, preparation.prepare_fwi_loss_pair)
        self.assertIs(prepare_loss_pair, preparation.prepare_loss_pair)
        self.assertIs(sum_weighted_losses, loss.sum_weighted_losses)


if __name__ == "__main__":
    unittest.main()
