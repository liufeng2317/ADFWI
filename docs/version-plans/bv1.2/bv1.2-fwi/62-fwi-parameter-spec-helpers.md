# 62 - FWI Parameter Spec Helpers

## Optimization Path

Continue from step 61 by centralizing the small parameter-name and
gradient-processor index lists used by AcousticFWI and ElasticFWI.

The helper design keeps the physical model choice explicit:

- runtime helpers define stable acoustic and elastic parameter order;
- ElasticFWI still decides whether anisotropic parameters are active with
  `isinstance(model, AnisotropicElasticModel)`;
- gradient specs remain plain `(name, idx)` pairs for the legacy list-style
  `GradProcessor` interface.

## Change

- Added parameter helper constants/functions in `ADFWI.fwi.runtime.gradient`:
  - `parameter_specs(...)`;
  - `acoustic_parameter_names()`;
  - `acoustic_gradient_parameter_specs()`;
  - `elastic_parameter_names(include_anisotropic=False)`;
  - `elastic_gradient_parameter_specs(include_anisotropic=False)`.
- Exported these helpers through `ADFWI.fwi.runtime`.
- Updated AcousticFWI regularization and gradient dispatch to use acoustic
  parameter helpers.
- Updated ElasticFWI regularization, cache/gradient snapshot parameter names,
  and gradient dispatch to use elastic parameter helpers.
- Added focused runtime tests for acoustic order, isotropic elastic order, and
  anisotropic elastic behavior.

Follow-up note: step 120 moved acoustic/elastic physical parameter-order
constants and convenience helpers from `ADFWI.fwi.runtime.gradient` into the
owning FWI drivers. `runtime.gradient` now keeps only the generic
`parameter_specs(...)` helper and gradient processor dispatch.

## Scientific Contract

- Acoustic parameter order remains `vp`, `rho`.
- Acoustic gradient processor indices remain `vp -> 0`, `rho -> 1`.
- Isotropic elastic parameter order remains `vp`, `vs`, `rho`.
- Anisotropic elastic regularization/cache order remains `vp`, `vs`, `rho`,
  `eps`, `delta`, `gamma`.
- Anisotropic elastic gradient processor specs remain `vp -> 0`, `vs -> 1`,
  `rho -> 2`, `eps -> 3`, `delta -> 4`; `gamma` remains excluded from the
  current ElasticFWI gradient processor path, matching the previous loop.
- Regularization formulas, cache contents, gradient processing, optimizer
  stepping, scheduler stepping, model constraints, and loss construction are
  unchanged.

## Validation

Completed validation:

- `python -m py_compile ADFWI/fwi/runtime/gradient.py ADFWI/fwi/runtime/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_runtime.py`: 23 tests passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_fwi_iteration.py tests/test_backend_integration.py`: 65 tests passed.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`: passed.
  - Acoustic CPU/NPU relative differences: loss 0.0, `vp_grad_norm` 0.0, `vp_update_norm` 0.0.
  - Elastic CPU/NPU relative differences: loss `4.880620563312549e-06`, `vp_grad_norm` `1.2590211728446073e-06`, `vp_update_norm` 0.0.

## Next Optimization Direction

Extract a small model-regularization summation helper that consumes ordered
parameter names and regularization weights. This can reduce the remaining
regularization loop duplication in AcousticFWI and ElasticFWI while preserving
parameter order through the new parameter helpers.
