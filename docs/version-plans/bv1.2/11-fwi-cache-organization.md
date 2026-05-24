# FWI Cache Organization

## Goal

Keep FWI main loops focused on numerical work by moving repeated result-cache
bookkeeping into explicit helper methods. This is a structural cleanup only; it
does not change propagation, data transforms, misfit calculation, regularization,
backward propagation, optimizer steps, or scheduler steps.

## Current Step

AcousticFWI now has `save_model_and_gradients(epoch_id, loss_epoch)`, matching
the existing ElasticFWI style. The normal optimizer path and the LBFGS/NLCG
closure path both route cache writes through this method.

The helper preserves the previous cache order:

1. copy `vp` and `rho` to numpy;
2. append cached models and epoch index according to `cache_result_epoch`;
3. append scalar loss;
4. save model figures;
5. append and save available gradients.

## Validation

Because this change only touches cache bookkeeping, validation focused on syntax,
existing unit coverage, and the acoustic smoke path where `cache_result=True` is
used by the mini inversion script.

Validation run after the change:

- `python -m py_compile ADFWI/fwi/acoustic_fwi.py` passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_loop.py tests/test_backend_integration.py tests/test_fwi_data_contract.py` passed: 30 tests OK.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic --cases trace-missing --devices cpu,npu:0` passed. CPU/NPU drift was zero for loss, `vp_grad_norm`, and `vp_update_norm`.

## Next Steps

1. If this remains stable, consider a shared cache helper for acoustic and
   elastic with model-parameter descriptors.
2. Keep closure refactoring separate from normal optimizer loop refactoring.
3. Only extract regularization/loss accumulation after cache and batch loop
   organization are fully validated.
