# FWI Runtime Cache Helper

## Goal

Keep `AcousticFWI` and `ElasticFWI` readable as geophysical inversion drivers by
moving repeated result-cache bookkeeping into `ADFWI.fwi.runtime.cache`. The
helper owns only mechanical recording behavior: loss history, model snapshots,
gradient snapshots, and cached epoch indices. Each FWI driver still decides which
physical parameters are meaningful for its workflow.

## Change

- Added `ADFWI.fwi.runtime.tensor_to_numpy` for detached CPU NumPy snapshots.
- Added `append_epoch_loss`, `should_cache_epoch`, `snapshot_model_parameters`,
  `append_model_snapshots`, and `append_required_gradient_snapshots`.
- Updated `AcousticFWI.save_model_and_gradients()` to use the shared cache
  helpers for `vp` and `rho`.
- Updated `ElasticFWI.save_model_and_gradients()` to use the shared cache
  helpers for isotropic parameters and optional anisotropic parameters.
- Kept figure-saving decisions inside each FWI driver because plot grouping is
  tied to acoustic, elastic, and anisotropic model semantics.

## Robustness Note

The shared snapshot helper returns a copied NumPy array. This avoids historical
CPU-side aliasing where a cached `.detach().numpy()` view could be mutated by a
later in-place optimizer step. This changes cache storage robustness only; it
does not change loss calculation, backpropagation, gradient processing,
regularization, optimizer steps, device placement, or model constraints.

## Scientific Contract

This is a structural runtime refactor. It must preserve:

- model parameter tensor values used by the optimizer;
- gradient tensors before and after `GradProcessor`;
- loss accumulation and scheduler/optimizer order;
- AcousticFWI cached parameter names: `vp`, `rho`;
- ElasticFWI cached parameter names: `vp`, `vs`, `rho`, plus `eps`, `delta`,
  `gamma` for anisotropic models;
- cached gradient names gated by `model.get_requires_grad(name)`;
- CPU/NPU smoke drift within the existing bv1.2 tolerances.

## Validation

Completed on 2026-05-26.

Syntax check:

```bash
python -m py_compile ADFWI/fwi/runtime/cache.py ADFWI/fwi/runtime/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py
```

Result: passed.

Focused runtime tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py
```

Result: passed, 12 tests. The new cache-helper tests cover copied NumPy
snapshots, loss recording, epoch-index recording, model parameter snapshots, and
`get_requires_grad(name)`-gated gradient snapshots.

Broader runtime/backend/data/iteration tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py
```

Result: passed, 65 tests.

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
tolerance and matches the prior bv1.2 backend smoke behavior.
