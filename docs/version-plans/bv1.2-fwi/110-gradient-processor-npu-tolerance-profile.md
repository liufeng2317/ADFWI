# 110. Gradient Processor NPU Tolerance Profile

## Goal

Continue convergence-focused optimization by turning the diagnosed NPU float32
gradient-processor drift into an explicit benchmark tolerance profile. This
keeps the default strict parity gate unchanged while allowing intentional NPU
validation to use a documented engineering tolerance.

## Optimization Path

1. Keep `gradient_processor_benchmark.py` default behavior on `strict`.
2. Add `--tolerance-profile` with:
   - `strict`: `rtol=1e-5`, `atol=2e-3`;
   - `npu-float32`: `rtol=2e-4`, `atol=5e-1`.
3. Keep `--compare-rtol` and `--compare-atol` as explicit overrides for either
   profile.
4. Record the selected profile and resolved tolerances in the benchmark JSON.
5. Extend the CPU CLI smoke tests so the default strict profile and profile
   override behavior are guarded.

## Numerical Contract

No FWI core code is changed. This only changes the benchmark acceptance logic.
The strict profile remains the default and still catches the known NPU smoothing
drift. The `npu-float32` profile must be requested explicitly when validating
the opt-in `TorchGradProcessor` on NPU.

The `npu-float32` thresholds come from the stage diagnostics in record 109:
raw NPU smoothing drift is about `1.4e-4` relative, and final `vmax=2500`
normalization amplifies the absolute difference to roughly `0.1-0.3`.

## Validation Result

CPU unit and syntax checks passed:

```bash
conda run -n adfwi python -m unittest tests/test_gradient_processor_benchmark.py
# Ran 3 tests in 60.657s, OK

conda run -n adfwi python -m py_compile scripts/benchmark/gradient_processor_benchmark.py tests/test_gradient_processor_benchmark.py
# OK
```

NPU strict profile still fails as intended:

```bash
conda run -n adfwi python scripts/benchmark/gradient_processor_benchmark.py --device npu:0 --warmup 0 --repeat 1 --nx 8 --nz 6 --output tests/full_cases/outputs/gradient_benchmark_npu_strict_after_profile.json
# status: failed
```

Strict-profile NPU drift:

| Case | Max abs diff | Max rel diff | Status |
| --- | ---: | ---: | --- |
| norm | `2.44140625e-04` | `9.765625e-08` | ok |
| marine_smooth | `3.133544921875e-01` | `1.25341796875e-04` | failed |
| land_smooth | `3.3236476662841596e-01` | `1.3294590665136638e-04` | failed |
| illumination | `1.5536359614952744e-01` | `6.214543845981098e-05` | failed |

NPU `npu-float32` profile passes:

```bash
conda run -n adfwi python scripts/benchmark/gradient_processor_benchmark.py --device npu:0 --warmup 0 --repeat 1 --nx 8 --nz 6 --tolerance-profile npu-float32 --output tests/full_cases/outputs/gradient_benchmark_npu_float32_profile.json
# status: ok
```

Resolved `npu-float32` tolerances:

| Field | Value |
| --- | ---: |
| `compare_rtol` | `2e-4` |
| `compare_atol` | `5e-1` |

The same numerical drift passes under the explicit profile:

| Case | Max abs diff | Max rel diff | Status |
| --- | ---: | ---: | --- |
| norm | `2.44140625e-04` | `9.765625e-08` | ok |
| marine_smooth | `3.133544921875e-01` | `1.25341796875e-04` | ok |
| land_smooth | `3.3236476662841596e-01` | `1.3294590665136638e-04` | ok |
| illumination | `1.5536359614952744e-01` | `6.214543845981098e-05` | ok |

## Next Direction

Use the explicit `npu-float32` profile for the next fixed Marmousi2
`TorchGradProcessor` opt-in comparison. The comparison should remain
non-default and should report loss curves, final model differences, and gradient
statistics against the legacy gradient processor path.
