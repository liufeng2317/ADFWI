# 75. Marmousi2 Iteration Smoke Control

## Optimization Path

Move from tiny synthetic examples and one-step real-case checks toward real-case
multi-iteration validation. The reduced Marmousi2 acoustic script now supports
configurable iteration counts so 2, 10, and later 100 iteration runs can use the
same reproducible entry point.

## Change

- Added `--iterations` to
  `scripts/examples/marmousi2_acoustic_reduced_inversion.py`.
- The script now records `loss_history`, `initial_loss`, `loss_min`, and
  `loss_max` in its JSON output.
- Added positive iteration validation.
- Added `--case-inversion-iterations` to
  `scripts/smoke/run_backend_smoke_suite.py`.
- Updated backend and example documentation with 10-iteration real-case
  commands.

## Scientific Contract

- No propagator, misfit, transform, gradient processor, optimizer, or FWI driver
  formula changed.
- The default remains one iteration, preserving existing smoke behavior.
- Multi-iteration validation uses the existing reduced Marmousi2 subset:
  one shot, 300 time samples, pressure observations, safe squared-L2 loss, and
  legacy `GradProcessor`.

## Validation

Run in the `adfwi` conda environment:

```bash
conda run -n adfwi python -m unittest tests/test_backend_smoke_suite.py
conda run -n adfwi python -m py_compile scripts/examples/marmousi2_acoustic_reduced_inversion.py scripts/smoke/run_backend_smoke_suite.py tests/test_backend_smoke_suite.py
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device cpu --shot-count 1 --nt-samples 300 --iterations 2
git diff --check
```

CPU reduced Marmousi2 2-iteration result:

| Metric | Value |
| --- | ---: |
| seconds | 109.2927095592022 |
| initial_loss | 4.9887580644281115e-06 |
| final_loss | 4.9887580644281115e-06 |
| loss_min | 4.9887580644281115e-06 |
| loss_max | 4.9887580644281115e-06 |
| vp_grad_norm | 4.904892342435206e-14 |
| vp_update_norm | 0.09812843799591064 |

A CPU 10-iteration run on the same 1-shot, 300-sample subset was stopped after
more than 3.5 minutes without JSON output. Treat 10 and 100 iterations as
long-run real-case gates, preferably on `npu:0` or in a scheduled benchmark
session rather than as default per-change validation.

## Next Step

Run the reduced Marmousi2 `--iterations 10` gate on `npu:0`, then use the result
to decide whether a 100-iteration baseline should run with one shot or a smaller
time window.
