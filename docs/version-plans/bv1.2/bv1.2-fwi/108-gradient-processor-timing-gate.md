# 108. Gradient Processor Timing Gate

## Goal

Continue convergence-focused optimization by adding a cheap timing and parity
gate for the legacy NumPy/SciPy `GradProcessor` and opt-in `TorchGradProcessor`.
This runs before any fixed Marmousi2 baseline comparison so performance work has
a small, reproducible signal.

## Optimization Path

1. Add `scripts/benchmark/gradient_processor_benchmark.py`.
2. Benchmark deterministic gradient arrays without running a propagator.
3. Report JSON with:
   - backend/environment/git metadata;
   - shape, dtype, seed, repeat/warmup;
   - per-case legacy/torch timing;
   - max absolute and relative drift;
   - speedup ratio.
4. Cover the same convergence cases as the parity tests:
   - `norm`
   - `marine_smooth`
   - `land_smooth`
   - `illumination`
5. Add `tests/test_gradient_processor_benchmark.py` as a CPU JSON smoke.

## Numerical Contract

The benchmark does not change FWI behavior. It runs both processors on the same
deterministic arrays and fails if parity exceeds explicit tolerances
(`rtol=1e-5`, `atol=2e-3` by default).

## Validation Result

CLI smoke passed on CPU:

```bash
conda run -n adfwi python -m unittest tests/test_gradient_processor_benchmark.py
# Ran 1 test in 22.716s, OK
```

Direct CPU benchmark passed:

```bash
conda run -n adfwi python scripts/benchmark/gradient_processor_benchmark.py --device cpu --warmup 0 --repeat 1 --nx 8 --nz 6
# status: ok
# max_abs_diff by case:
#   norm: 0.0
#   marine_smooth: 6.103515625e-04
#   land_smooth: 4.15171704162276e-04
#   illumination: 7.880099251451611e-04
```

Related parity and syntax checks passed:

```bash
conda run -n adfwi python -m unittest tests/test_torch_grad_processor.py tests/test_import_surface_policy.py
# Ran 9 tests in 3.032s, OK

conda run -n adfwi python -m py_compile scripts/benchmark/gradient_processor_benchmark.py tests/test_gradient_processor_benchmark.py
# OK
```

A small `npu:0` smoke was also run:

```bash
conda run -n adfwi python scripts/benchmark/gradient_processor_benchmark.py --device npu:0 --warmup 0 --repeat 1 --nx 8 --nz 6
# status: failed
```

The NPU result is useful as a convergence gate: the `norm` case passed, but
`marine_smooth`, `land_smooth`, and `illumination` exceeded the strict CPU
parity tolerance. Observed NPU drift:

| Case | Max abs diff | Max rel diff |
| --- | ---: | ---: |
| norm | `2.44140625e-04` | `9.765625e-08` |
| marine_smooth | `3.133544921875e-01` | `1.25341796875e-04` |
| land_smooth | `3.3236476662841596e-01` | `1.3294590665136638e-04` |
| illumination | `1.5536359614952744e-01` | `6.214543845981098e-05` |

This means the torch gradient path should not move to a Marmousi2 full-case gate
yet. The next step is to isolate whether the drift comes from NPU convolution
precision, reduction order in normalization, or dtype/device conversion.
```

## Next Direction

Investigate the NPU drift in the smoothing and illumination cases before running
the fixed Marmousi2 3-shot baseline with `TorchGradProcessor`.
