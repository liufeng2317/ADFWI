# Acoustic Pressure Update Variants

Date: 2026-06-01

## Purpose

This round tested whether pressure-stencil expression rewrites can reduce the
dominant production checkpoint replay cost identified in the stencil breakdown.

No production kernel change was made.

## Benchmark

Added:

```text
scripts/benchmark/acoustic_pressure_update_variant_microbenchmark.py
```

The benchmark compares formula-equivalent pressure update variants against the
production-style sliced assignment:

```text
reference
explicit_div
split_div
addcmul
```

Shape:

```text
device: npu:0
dtype: float32
shots: 3
nx/nz: 200/88
nabc: 30
receivers: 200
```

## Numerical Result

| Variant | Output max abs diff | p grad max abs diff | u grad max abs diff | w grad max abs diff |
| --- | ---: | ---: | ---: | ---: |
| explicit div | `0.0` | `0.0` | `0.0` | `0.0` |
| split div | `2.328e-10` | `1.243e-14` | `5.204e-18` | `5.204e-18` |
| addcmul | `4.657e-10` | `3.553e-14` | `1.735e-17` | `1.388e-17` |

All variants are numerically safe at microbenchmark scale.

## Timing Result

Speedup is relative to the production-style pressure update expression.

| Variant | Forward speedup mean | Backward speedup mean | Total speedup mean |
| --- | ---: | ---: | ---: |
| explicit div | `0.970x` | `1.035x` | `1.017x` |
| split div | `0.924x` | `0.880x` | `0.890x` |
| addcmul | `0.961x` | `0.855x` | `0.880x` |

## Decision

Do not promote any pressure-update expression variant to production.

Reason:

```text
The only exact variant, explicit_div, improves total time by only about 1.7%,
well below the 10% microbenchmark threshold for entering a full production
checkpoint gate. The other variants are slower.
```

## Interpretation

Simple Python/TorchScript expression rearrangement is not enough to reduce the
pressure-stencil replay cost on the tested NPU stack.

The remaining cost appears to be the actual finite-difference slice arithmetic
and copy-slice autograd work rather than the superficial expression layout.

## Next Direction

Stop pressure-expression rewrites in Python.

The next meaningful options are:

```text
1. Lower-level fused stencil implementation for pressure/u/w updates, with a
   strict output/loss/raw-gradient parity gate.
2. Boundary-saving research path, treated separately from production checkpoint.
3. Defer further acoustic kernel performance work until a lower-level backend
   implementation path is available.
```

For this branch, the practical recommendation is to stop production
`acoustic_kernels.py` micro-optimizations here unless a lower-level fused
operator path is selected.
