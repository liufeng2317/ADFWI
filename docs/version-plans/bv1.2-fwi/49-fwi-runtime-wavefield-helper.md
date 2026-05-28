# FWI Runtime Wavefield Helper

## Goal

Keep the FWI loops focused on the inversion algorithm by moving repeated
forward-wavefield accumulation into `ADFWI.fwi.runtime.wavefield`. This helper is
execution glue only: it converts forward wavefield tensors to the legacy detached
CPU NumPy format expected by `GradProcessor` and accumulates them over shot
batches.

## Change

- Added `ADFWI.fwi.runtime.wavefield_to_numpy`.
- Added `accumulate_wavefield` for acoustic pressure-wavefield accumulation.
- Added `accumulate_named_wavefields` for elastic component wavefield
  accumulation.
- Added `select_elastic_gradient_wavefield` with the legacy priority
  `pressure -> vz`, plus a `vx` fallback for vx-only component inversions.
- Updated `AcousticFWI.forward()` and `AcousticFWI.forward_closure()` to use the
  shared accumulator.
- Updated `ElasticFWI.forward()` to use a named wavefield accumulator while
  keeping the physical pressure definition `-(txx + tzz)` in the elastic driver.

## Scientific Contract

This is a structural refactor of gradient-processor inputs. It must preserve:

- the tensor values used for waveform misfit and backpropagation;
- detached CPU NumPy wavefields passed to legacy `GradProcessor`;
- acoustic pressure wavefield accumulation over shot batches;
- elastic pressure wavefield definition as `-(forward_wavefield_txx +
  forward_wavefield_tzz)`;
- legacy elastic gradient-wavefield priority: pressure first, otherwise `vz`;
- optimizer, scheduler, model constraint, cache, and plotting order.

The `vx` fallback is a robustness improvement for vx-only elastic component
inversions. It does not alter pressure or vz behavior.

## Validation

Completed on 2026-05-26.

Syntax check:

```bash
python -m py_compile ADFWI/fwi/runtime/wavefield.py ADFWI/fwi/runtime/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py
```

Result: passed.

Focused runtime tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py
```

Result: passed, 15 tests. The new tests cover unnamed and named wavefield
accumulation, detached NumPy conversion, elastic gradient-wavefield priority,
and the vx-only fallback.

Broader runtime/backend/data/iteration tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py
```

Result: passed, 68 tests.

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
