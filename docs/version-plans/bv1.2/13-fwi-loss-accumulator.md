# FWI Loss Accumulator

## Goal

Make the acoustic and elastic batch loops less repetitive by moving only the
small, shared loss-composition pattern into a helper. This step intentionally
does not move data-loss calculation, model-regularization calculation, backward
propagation, gradient processing, optimizer steps, or scheduler steps.

## Current Step

`ADFWI.fwi.iteration` now provides `BatchLoss` and `build_batch_loss(data_loss,
regularization_loss=None)`. The helper returns:

- `tensor`: `data_loss` when no regularization is used, otherwise
  `data_loss + regularization_loss`;
- `scalar`: `data_loss.item()` when no regularization is used, otherwise
  `data_loss.item() + regularization_loss.item()`.

AcousticFWI normal/closure paths and ElasticFWI now call this helper before
calling `batch_loss.tensor.backward()` in the same loop location as before.

## Validation

Validation covers both helper semantics and end-to-end behavior:

- unit tests confirm scalar values and gradients match the previous expanded
  expression with and without regularization;
- syntax compilation includes loop, acoustic, and elastic FWI modules;
- existing backend/data/regularization tests remain green;
- acoustic and elastic CPU/NPU smoke comparisons remain within the established
  tolerances.

Validation run after the change:

- `python -m py_compile ADFWI/fwi/iteration.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_iteration.py` passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_iteration.py tests/test_backend_integration.py tests/test_fwi_data_contract.py` passed: 34 tests OK.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0` passed. Acoustic CPU/NPU drift was zero for loss, `vp_grad_norm`, and `vp_update_norm`; elastic maximum relative drift was `4.880620563312549e-06`, within the `1e-5` tolerance.

## Next Steps

1. Consider extracting per-batch progress-label handling if the smoke baseline
   remains stable.
2. Keep closure-specific optimizer behavior untouched until normal-loop cleanup
   is finished.
3. Avoid a broad base FWI class until the small helpers have stabilized.
