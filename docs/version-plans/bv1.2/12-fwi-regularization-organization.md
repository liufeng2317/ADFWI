# FWI Regularization Organization

## Goal

Move repeated model-regularization bookkeeping out of the acoustic and elastic
FWI batch loops while preserving the exact loss construction sequence. This is a
small structural step toward cleaner FWI loop orchestration.

## Current Step

AcousticFWI now uses `calculate_model_regularization_loss()` for the vp/rho
regularization sum in both the normal optimizer path and the LBFGS/NLCG closure
path.

ElasticFWI now uses `calculate_model_regularization_loss()` for vp/vs/rho and,
when applicable, eps/delta/gamma. The helper keeps parameter order aligned with
`regularization_weights_x` and `regularization_weights_z`.

The batch loops still own the same numerical sequence:

1. compute data loss;
2. compute model regularization loss when `regularization_fn` is present;
3. add scalar loss history from `.item()` values;
4. build `loss = data_loss + regularization_loss`;
5. call `loss.backward()` in the same loop position.

## Validation

Validation proves both helper behavior and end-to-end smoke behavior:

- unit tests compare helper output against the previous expanded regularization
  sums for acoustic vp/rho and elastic vp/vs/rho;
- syntax compilation covers acoustic and elastic FWI files;
- existing backend/data/loop tests remain green;
- acoustic and elastic CPU/NPU smoke comparisons remain within the established
  tolerances.

Validation run after the change:

- `python -m py_compile ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_backend_integration.py` passed.
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py tests/test_fwi_loop.py tests/test_fwi_data_contract.py` passed: 32 tests OK.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0` passed. Acoustic CPU/NPU drift was zero for loss, `vp_grad_norm`, and `vp_update_norm`; elastic maximum relative drift was `4.880620563312549e-06`, within the `1e-5` tolerance.

## Next Steps

1. If this remains stable, extract a tiny loss accumulator helper that only owns
   scalar history updates, not tensor loss construction.
2. Keep closure-specific logic separate until regular optimizer loop cleanup is
   fully validated.
3. Consider moving shared regularization helper patterns into a common mixin only
   after acoustic and elastic behavior stays stable through multiple smoke cases.
