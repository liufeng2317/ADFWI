# 73. Example Gradient Processor Comparison

## Optimization Path

Continue from the opt-in `TorchGradProcessor` example support by making the
examples smoke suite compare legacy and torch gradient processing in one run.
This turns the next optimization direction from record 72 into a repeatable
validation command before larger benchmark work.

## Change

- Added `--example-gradient-processors legacy,torch` to
  `scripts/smoke/run_backend_smoke_suite.py`.
- The existing `--example-gradient-processor legacy|torch` single-path option
  remains supported.
- The examples suite now records the gradient processor used for each run.
- CPU-vs-device comparisons are grouped by problem and gradient processor.
- New legacy-vs-torch comparisons are grouped by problem and device and report
  drift for `loss`, `vp_grad_norm`, and `vp_update_norm`.
- Updated backend/example documentation with the new command.

## Scientific Contract

- No propagator, loss, transform, optimizer, or FWI driver code changed.
- The legacy gradient processor remains the default path.
- This is validation and orchestration work: it exposes numerical drift between
  the existing legacy and opt-in torch-native gradient processor paths without
  changing either implementation.

## Validation

Run in the `adfwi` conda environment:

```bash
conda run -n adfwi python -m unittest tests/test_backend_smoke_suite.py
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites examples --devices cpu --example-problems acoustic --example-gradient-processors legacy,torch
conda run -n adfwi python -m py_compile scripts/smoke/run_backend_smoke_suite.py tests/test_backend_smoke_suite.py
git diff --check
```

Because this change touches smoke orchestration only, FWI numerical precision is
validated through the reported legacy-vs-torch example metrics rather than by
changing core numerical code.

Standard acoustic CPU example metrics:

| Metric | Legacy | Torch | Abs diff | Rel diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| loss | 3.8032811744415085e-07 | 3.8032811744415085e-07 | 0.0 | 0.0 | pass |
| vp_grad_norm | 1.8180083216634557e-08 | 1.8180083216634557e-08 | 0.0 | 0.0 | pass |
| vp_update_norm | 1.8180636167526245 | 1.8180636167526245 | 0.0 | 0.0 | pass |

## Next Step

Extend `scripts/benchmark/acoustic_backend_benchmark.py` so benchmark rows can
compare legacy and torch gradient processor runtime and memory on larger acoustic
grids.
