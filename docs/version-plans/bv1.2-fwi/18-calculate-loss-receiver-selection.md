# Calculate Loss Receiver Selection

## Goal

Make direct `calculate_loss(..., apply_transforms=True)` calls use the same
receiver-selection and transform-pipeline order as the main AcousticFWI and
ElasticFWI training loops. This closes the gap left by the earlier data-contract
refactor, where the training path used `_prepare_loss_pair()` but direct
`calculate_loss()` calls only executed the transform pipeline.

## Current Step

AcousticFWI and ElasticFWI now call their `_prepare_loss_pair()` wrappers inside
`calculate_loss(..., apply_transforms=True)`. The main training loops are not
changed because they already prepare pairs first and call
`calculate_loss(..., apply_transforms=False)`.

The effective order for direct `calculate_loss()` calls is now:

1. Build the FWI transform context.
2. Select/mask receivers when a shot-scoped receiver mask is present.
3. Run the data transform pipeline.
4. Apply optional waveform normalization.
5. Evaluate the requested misfit.

## Validation

The public API behavior and end-to-end smoke baseline were both checked:

- added AcousticFWI and ElasticFWI backend integration tests where synthetic
  data has all receiver traces and observed data contains only active traces;
- those direct `calculate_loss()` calls return zero loss after receiver
  selection;
- `python -m py_compile ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_backend_integration.py`
  passed;
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py`
  passed with 43 tests;
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`
  passed.

CPU/NPU smoke drift stayed within the established bv1.2 baseline:

- acoustic loss, gradient norm, and update norm drift were all zero;
- elastic loss relative drift was `4.880620563312549e-06`;
- elastic gradient relative drift was `1.2590211728446073e-06`;
- elastic update norm drift was zero.

## Next Steps

1. Consider moving duplicated waveform normalization into a shared helper once
   direct loss preparation is stable.
2. Keep direct `calculate_loss()` tests in place as API guards for notebook and
   research-script usage.
3. Avoid changing `apply_transforms=False`, which is now the explicit signal that
   the caller has already prepared same-shape loss tensors.
