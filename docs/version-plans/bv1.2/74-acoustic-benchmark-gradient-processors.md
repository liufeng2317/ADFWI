# 74. Acoustic Benchmark Gradient Processors

## Optimization Path

Continue from the examples-suite legacy-vs-torch comparison by extending the
acoustic benchmark scaffold to measure gradient post-processing runtime and
memory metadata. This prepares the benchmark path for larger CPU/NPU comparisons
without changing FWI core numerics.

## Change

- Added `--gradient-processors none|legacy|torch` support to
  `scripts/benchmark/acoustic_backend_benchmark.py`.
- Kept the historical default behavior as `--gradient-processors none`, so the
  benchmark still measures forward/backward only unless a processor is selected.
- Added a timed `gradient_seconds` stage after backward.
- Added `metrics_by_gradient_processor` and `comparisons` JSON blocks.
- Added legacy-vs-torch drift checks for `loss`, `vp_grad_norm`, and
  `pressure_l2`.
- Extended `tests/test_acoustic_benchmark.py` to cover the default path and the
  `legacy,torch` comparison path.

## Scientific Contract

- No propagator, misfit, transform, optimizer, or FWI driver code changed.
- The benchmark applies gradient processors after autograd backward on the same
  synthetic acoustic case.
- The legacy and torch processors are configured with
  `norm_grad=False, forw_illumination=False`, matching the current minimal smoke
  comparison path and isolating post-processing dispatch overhead.

## Validation

Run in the `adfwi` conda environment:

```bash
conda run -n adfwi python -m unittest tests/test_acoustic_benchmark.py
conda run -n adfwi python -m py_compile scripts/benchmark/acoustic_backend_benchmark.py tests/test_acoustic_benchmark.py
conda run -n adfwi python scripts/benchmark/acoustic_backend_benchmark.py --device cpu --warmup 1 --repeat 1 --gradient-processors legacy,torch
git diff --check
```

The focused unit test passed with 2 tests. The CPU benchmark command passed on
the default `nx=24, nz=20, nt=30` case after one warmup.

Post-warm CPU benchmark metrics:

| Metric | Legacy | Torch | Abs diff | Rel diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| loss | 3.7251948192817963e-09 | 3.7251948192817963e-09 | 0.0 | 0.0 | pass |
| vp_grad_norm | 7.296129846123822e-12 | 7.296129846123822e-12 | 0.0 | 0.0 | pass |
| pressure_l2 | 0.0005790229188278317 | 0.0005790229188278317 | 0.0 | 0.0 | pass |

Runtime means from the same `repeat=1` run:

| Stage | Legacy seconds | Torch seconds |
| --- | ---: | ---: |
| forward | 0.017988404259085655 | 0.01797321066260338 |
| backward | 0.07877816446125507 | 0.07857020385563374 |
| gradient processing | 0.00022765249013900757 | 0.00012052059173583984 |
| total | 0.09699422121047974 | 0.09666393510997295 |

## Next Step

Run the same benchmark on `npu:0` and with larger acoustic grids using
`--repeat 3` or more, then decide whether the torch-native processor should be
expanded to smoothing and illumination validation.
