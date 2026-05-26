# FWI Runtime Gradient Dispatch Helper

## Goal

Keep the FWI loops focused on the inversion algorithm by moving repeated
`model.get_requires_grad(name)` gradient-dispatch branches into a small runtime
helper. The FWI drivers still own the physical parameter order and index mapping
used by their `GradProcessor` configuration.

## Change

- Added `ADFWI.fwi.runtime.process_named_parameter_gradients`.
- Updated `AcousticFWI.forward()` and `AcousticFWI.forward_closure()` to dispatch
  `vp` and `rho` gradients through the helper.
- Updated `ElasticFWI.forward()` to build the elastic parameter/index list in the
  driver, then dispatch through the helper.
- Added focused runtime coverage for the `get_requires_grad(name)` gate, idx
  propagation, and processed parameter reporting.

## Scientific Contract

This is a structural dispatch refactor. It must preserve:

- which model parameters receive `GradProcessor` calls;
- AcousticFWI parameter order: `vp -> 0`, `rho -> 1`;
- ElasticFWI parameter order: `vp -> 0`, `vs -> 1`, `rho -> 2`, optional
  anisotropic `eps -> 3`, `delta -> 4`;
- the forward wavefield object passed to every processed parameter;
- the existing `process_parameter_gradient` CPU NumPy processor contract;
- optimizer, scheduler, model constraint, cache, and plotting order.

## Validation

Completed on 2026-05-26.

Syntax check:

```bash
python -m py_compile ADFWI/fwi/runtime/gradient.py ADFWI/fwi/runtime/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py
```

Result: passed.

Focused runtime tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py
```

Result: passed, 16 tests.

Broader runtime/backend/data/iteration tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py
```

Result: passed, 69 tests.

Acoustic/elastic CPU/NPU smoke comparison:

```bash
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0
```

Result: passed.

| Problem | Metric | Relative difference | Status |
| --- | --- | ---: | --- |
| acoustic | loss | 0.0 | pass |
| acoustic | vp_grad_norm | 0.0 | pass |
| acoustic | vp_update_norm | 0.0 | pass |
| elastic | loss | 4.880620563312549e-06 | pass |
| elastic | vp_grad_norm | 1.2590211728446073e-06 | pass |
| elastic | vp_update_norm | 0.0 | pass |

The elastic CPU/NPU drift remains inside the established `1e-5` relative
tolerance and matches the previous bv1.2 backend smoke behavior.

## Next Optimization Direction

The next useful cleanup is the per-batch loss-step block shared by AcousticFWI
and ElasticFWI: combine data loss, optional regularization, scalar accumulation,
backward, and progress description into a small iteration helper. Keep component
selection and synthetic/observed pair construction in the drivers so the code
still reads as a physical FWI workflow.
