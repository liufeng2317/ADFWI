# 109. Gradient Processor Stage Diagnostics

## Goal

Continue convergence-focused optimization by isolating the NPU drift observed in
the previous legacy-vs-torch gradient processor timing gate. This step does not
change FWI core numerics; it adds a small diagnostic tool that separates raw
smoothing, post-smoothing normalization, and illumination preconditioning.

## Optimization Path

1. Add `scripts/benchmark/gradient_processor_stage_diagnostics.py`.
2. Compare deterministic legacy NumPy/SciPy outputs against torch CPU and the
   requested torch device.
3. Report JSON for:
   - raw `smooth2d` parity;
   - marine smoothing before normalization;
   - marine smoothing after `vmax` normalization;
   - illumination preconditioner parity;
   - illumination processing before and after normalization.
4. Add `tests/test_gradient_processor_stage_diagnostics.py` as a CPU JSON smoke.
5. Run the same small diagnostic on CPU and `npu:0`.

## Numerical Contract

The diagnostic script is read-only with respect to FWI behavior. It uses the same
deterministic input generator as `gradient_processor_benchmark.py` and reports
max absolute and relative differences instead of replacing tolerances in the
existing gate.

## Validation Result

CPU diagnostic smoke passed:

```bash
conda run -n adfwi python -m unittest tests/test_gradient_processor_stage_diagnostics.py
# Ran 1 test in 26.623s, OK

conda run -n adfwi python -m py_compile scripts/benchmark/gradient_processor_stage_diagnostics.py tests/test_gradient_processor_stage_diagnostics.py
# OK
```

Direct CPU diagnostic passed and matched the previous CPU gate:

```bash
conda run -n adfwi python scripts/benchmark/gradient_processor_stage_diagnostics.py --device cpu --nx 8 --nz 6 --smooth-span 2 --output tests/full_cases/outputs/gradient_stage_diag_cpu.json
# status: ok
```

Key CPU drift:

| Stage | Max abs diff | Max rel diff |
| --- | ---: | ---: |
| raw smooth, legacy vs torch CPU | `1.4901161193847656e-07` | `1.4428851793529365e-07` |
| marine smooth, no norm | `1.7881393432617188e-07` | `2.8450289946388275e-07` |
| marine smooth, with norm | `6.103515625e-04` | `2.44140625e-07` |
| illumination preconditioner | `2.8112152772319376e-07` | `2.811236758592523e-07` |
| illumination, no norm | `1.8134833683625118e-06` | `4.997472190113465e-07` |
| illumination, with norm | `7.880099251451611e-04` | `3.1520397005806444e-07` |

NPU diagnostic:

```bash
conda run -n adfwi python scripts/benchmark/gradient_processor_stage_diagnostics.py --device npu:0 --nx 8 --nz 6 --smooth-span 2 --output tests/full_cases/outputs/gradient_stage_diag_npu.json
# status: ok
```

Key NPU drift:

| Stage | Max abs diff | Max rel diff |
| --- | ---: | ---: |
| raw smooth, legacy vs torch CPU | `1.4901161193847656e-07` | `1.4428851793529365e-07` |
| raw smooth, legacy vs torch NPU | `1.4579296112060547e-04` | `1.411718859478913e-04` |
| raw smooth, torch CPU vs torch NPU | `1.4591217041015625e-04` | `1.4128733307115145e-04` |
| marine smooth, no norm | `5.7831406593322754e-05` | `9.201297940161074e-05` |
| marine smooth, with norm | `3.133544921875e-01` | `1.25341796875e-04` |
| illumination preconditioner | `4.3039370986375225e-05` | `4.3039699863446655e-05` |
| illumination, no norm | `3.939914576767123e-04` | `1.0857344419207512e-04` |
| illumination, with norm | `1.5536359614952744e-01` | `6.214543845981098e-05` |

The drift source is now localized: NPU `conv2d` used by `_torch_smooth2d`
introduces about `1e-4` relative drift on this small float32 case, while torch
CPU remains aligned with the legacy SciPy implementation. The larger absolute
differences in the processor output are mainly the expected effect of final
`vmax=2500` normalization.

An NPU `float64` diagnostic was also attempted:

```bash
conda run -n adfwi python scripts/benchmark/gradient_processor_stage_diagnostics.py --device npu:0 --dtype float64 --nx 8 --nz 6 --smooth-span 2
# failed: NPU convolution does not support DT_DOUBLE
```

The error reports that NPU convolution supports `DT_FLOAT`, `DT_FLOAT16`, and
`DT_BFLOAT16`, so `float64` is not a viable precision switch for this path.

## Next Direction

Do not promote `TorchGradProcessor` to the default gradient processor yet. The
next convergence step should either:

1. keep the torch path opt-in and define a documented NPU tolerance for
   smoothing/illumination diagnostics; or
2. prototype a dedicated NPU smoothing implementation and compare its drift and
   timing against the current `conv2d` path before any full Marmousi2 gate.
