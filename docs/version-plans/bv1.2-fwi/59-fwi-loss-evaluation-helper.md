# 59 - FWI Loss Evaluation Helper

## Optimization Path

Continue the staged FWI-loop cleanup from step 58 by moving the repeated
prepared-pair misfit evaluation into the FWI data contract layer.

The change is intentionally narrow:

- keep AcousticFWI and ElasticFWI responsible for choosing acoustic pressure or
  elastic component loss inputs;
- keep receiver selection, mute, low-pass filtering, data masks, and transform
  execution in `ADFWI.fwi.data.preparation`;
- keep misfit formula implementations in `ADFWI.fwi.misfit`;
- share only the mechanical path from `LossInput` records to weighted data loss.

## Change

- Added `ComponentLoss` and `LossEvaluation` records in `ADFWI.fwi.data.loss`.
- Added `evaluate_loss_inputs(...)` to prepare each `LossInput`, optionally
  apply legacy waveform normalization, dispatch the misfit, apply component
  weights, and return a summed `data_loss`.
- Updated `AcousticFWI.forward` and `AcousticFWI.forward_closure` to evaluate
  the acoustic pressure loss through `evaluate_loss_inputs`.
- Updated `ElasticFWI.forward` to evaluate active elastic component losses
  through `evaluate_loss_inputs`.
- Kept the acoustic historical fallback as `function_fallback="apply"` and the
  elastic historical fallback as `function_fallback="call"`.
- Added data-contract unit tests for weight accumulation, prepare-callback
  metadata forwarding, legacy normalization, and autograd preservation.

## Scientific Contract

- Acoustic loss inputs still come from `acoustic_pressure_loss_input`.
- Elastic loss inputs still come from `elastic_loss_inputs`, including pressure
  sign convention and component ordering.
- Transform preparation, receiver masking, low-pass filtering, normalization
  formula, misfit dispatch, regularization, backward propagation, gradient
  processing, optimizer stepping, and model constraints are unchanged.
- Component weights remain transparent in `LossEvaluation.component_losses` for
  future diagnostics.

## Validation

Completed validation:

- `python -m py_compile ADFWI/fwi/data/loss.py ADFWI/fwi/data/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py`: 27 tests passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_fwi_iteration.py tests/test_backend_integration.py`: 56 tests passed.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`: passed.
  - Acoustic CPU/NPU relative differences: loss 0.0, `vp_grad_norm` 0.0, `vp_update_norm` 0.0.
  - Elastic CPU/NPU relative differences: loss `4.880620563312549e-06`, `vp_grad_norm` `1.2590211728446073e-06`, `vp_update_norm` 0.0.

## Next Optimization Direction

Continue reducing duplicated acoustic batch-loop code by extracting a small
acoustic batch loss step used by both `forward` and `forward_closure`, while
keeping closure ownership and optimizer behavior visible in `AcousticFWI`.
