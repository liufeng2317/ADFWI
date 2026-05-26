# 58 - FWI Loss Input Record

## Goal

Move raw synthetic/observed loss-input pairing into the FWI data contract layer while preserving the existing transform, misfit, regularization, and optimizer behavior in the drivers.

## Change

- Added data-layer LossInput records that carry component, synthetic waveform, observed waveform, shot_index, and component weight.
- Added acoustic_pressure_loss_input(record_waveform, observed_pressure, shot_index) for acoustic pressure inversion batches.
- Added elastic_loss_inputs(record_waveform, observed_components, inversion_components, component_weights, shot_index) for active elastic component batches.
- Updated AcousticFWI.forward and AcousticFWI.forward_closure to use acoustic_pressure_loss_input before transform preparation.
- Updated ElasticFWI.forward to use elastic_loss_inputs before per-component transform preparation and weighted loss summation.
- Added focused data-contract tests for observed shot selection, elastic component order, pressure sign convention, and component weights.

## Directory Contract

- ADFWI.fwi.runtime.forward owns propagator execution and ForwardBatchRecord construction.
- ADFWI.fwi.runtime.wavefield owns forward-wavefield selection and accumulation for gradient processing.
- ADFWI.fwi.data.inputs owns raw loss-input pairing before transform preparation.
- ADFWI.fwi.data.preparation continues to own receiver selection, masks, transform context, and transform pipeline execution.
- AcousticFWI and ElasticFWI still own loss evaluation, regularization, gradient processing, and epoch updates.

## Scientific Contract

- Acoustic synthetic pressure still comes from record_waveform["p"].
- Acoustic observed pressure still uses obs_p[shot_index].
- Elastic synthetic components still use elastic_synthetic_components(record_waveform), including pressure = -(txx + tzz).
- Elastic observed components still use observed_components[component][shot_index].
- Elastic component order and weights still follow elastic_component_loss_inputs.
- Transform preparation, receiver masking, low-pass filtering, normalization, misfit evaluation, backward propagation, gradient processing, and model updates are unchanged.

## Validation

Completed validation:

- python -m py_compile ADFWI/fwi/data/inputs.py ADFWI/fwi/data/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py: passed.
- conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_fwi_runtime.py tests/test_fwi_iteration.py tests/test_backend_integration.py: 81 tests passed.
- conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0: passed.
  - Acoustic CPU/NPU relative differences: loss 0.0, vp_grad_norm 0.0, vp_update_norm 0.0.
  - Elastic CPU/NPU relative differences: loss 4.880620563312549e-06, vp_grad_norm 1.2590211728446073e-06, vp_update_norm 0.0.

## Next Optimization Direction

Continue the staged FWI-loop cleanup by extracting a small loss-evaluation helper that consumes prepared synthetic/observed tensors and returns weighted component losses, while keeping misfit dispatch and component weighting transparent for acoustic and elastic review.
