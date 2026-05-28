# 32. FWI Runtime Gradient Helper

## Goal

AcousticFWI and ElasticFWI used the same gradient preconditioning sequence:

1. move `parameter.grad` to CPU NumPy,
2. compute `vmax` from the current model parameter,
3. call either one shared `GradProcessor` or the indexed processor from a list,
4. convert the processed gradient back to the propagator dtype/device.

This update moves that sequence into `ADFWI.fwi.runtime.process_parameter_gradient` while keeping the public `process_gradient(...)` methods on both FWI drivers unchanged.

## Implementation

- Added `ADFWI/fwi/runtime/gradient.py`.
- Exported `process_parameter_gradient` from `ADFWI.fwi.runtime`.
- Replaced duplicate AcousticFWI/ElasticFWI method bodies with thin calls to the shared helper.
- Kept the legacy `GradProcessor` dispatch rule unchanged:
  - a single processor instance is used directly;
  - a processor list is indexed by `idx`.

## Numerical Contract

This is a structural refactor only. The helper intentionally preserves:

- CPU NumPy gradient processor input,
- `vmax = np.max(parameter.cpu().detach().numpy())`,
- output conversion through `numpy2tensor(..., dtype=propagator.dtype).to(propagator.device)`,
- existing failure behavior when `gradient_processor` is missing or an indexed processor is unavailable.

## Validation Plan

- `python -m py_compile` on modified FWI/runtime/test modules.
- `tests/test_fwi_runtime.py` covers single-processor and list-processor dispatch with exact processed-gradient checks.
- Backend integration/data/iteration tests ensure this extraction does not disturb surrounding runtime helpers.
- Acoustic/elastic backend smoke comparison checks loss/gradient/update drift after the refactor.

## Results

Validated on 2026-05-26 in the `adfwi` conda environment.

- `python -m py_compile ADFWI/fwi/runtime/gradient.py ADFWI/fwi/runtime/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py`: passed.
- `python -m unittest tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py`: 62 tests passed.
- `scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`: passed.

CPU/NPU drift summary:

| Problem | loss rel diff | vp_grad_norm rel diff | vp_update_norm rel diff | Status |
| --- | ---: | ---: | ---: | --- |
| acoustic | 0.0 | 0.0 | 0.0 | passed |
| elastic | 4.880620563312549e-06 | 1.2590211728446073e-06 | 0.0 | passed |

The observed elastic CPU/NPU drift remains inside the existing `1e-5` relative tolerance and is unchanged in character from prior backend smoke behavior.
