# Ascend Pressure-Update Prototype Compile Gate

Date: 2026-06-01

## Purpose

This is the first custom-op prototype gate after the empty scaffold compiled.
The goal is to verify that a minimal real acoustic pressure-update body can be
generated and compiled by the current Ascend/CANN toolchain.

This step still does not modify `ADFWI/propagator/acoustic_kernels.py` and does
not connect the operator to production FWI.

## Prototype Scope

Implemented in the generated scaffold only:

```text
p_next = (1 - kappa1) * p - alpha1 * div(u, w)
```

The prototype covers the interior pressure update for tensors shaped:

```text
p, u, w: [src_n, nz_pml, nx_pml]
kappa1, alpha1: [nz_pml, nx_pml]
```

Intentionally excluded from this gate:

- source injection;
- free-surface writeback;
- `u` and `w` velocity updates;
- receiver sampling;
- autograd/backward integration;
- production `forward_kernel` hook.

These exclusions keep the numerical contract small enough to validate before
any production integration.

## Script

```text
scripts/benchmark/ascend_fused_pressure_update_prototype.py
```

The script:

1. generates the same `FusedPressureUpdateForward` scaffold with `msopgen`;
2. patches tiling data to carry `src_n`, `nz_pml`, `nx_pml`, and
   `free_surface_start`;
3. replaces the generated TODO kernel with a minimal scalar AscendC pressure
   update;
4. compiles and packages the custom op;
5. validates the PyTorch pressure-update reference against an independent
   scalar Python implementation on a deterministic tiny tensor case.

## Command

```bash
conda run -n adfwi python -m py_compile \
  scripts/benchmark/ascend_fused_pressure_update_prototype.py

conda run -n adfwi python scripts/benchmark/ascend_fused_pressure_update_prototype.py \
  --compile \
  --output docs/version-plans/bv1.2-propagator-performance/ascend_fused_pressure_update_compile_20260601.json
```

## Result

| Check | Result |
| --- | --- |
| Python syntax | passed |
| scaffold generation | passed |
| generated Python patch | passed |
| pressure-update project patch | passed |
| PyTorch vector-vs-scalar reference | `max_abs_diff = 0.0` |
| custom-op compile | `returncode = 0` |
| package output | `custom_opp_ubuntu_aarch64.run` |

Reference tensor case:

```text
shape: [2, 9, 10]
free_surface_start: 2
max_abs_diff: 0.0
allclose: true
```

Patched generated files:

```text
op_host/fused_pressure_update_forward_tiling.h
op_kernel/fused_pressure_update_forward.cpp
op_host/fused_pressure_update_forward.cpp
```

## Decision

The real pressure-update kernel body is compile-feasible.

This does not yet prove runtime correctness, because the next gate is PyTorch
or ACL runtime invocation of the generated custom op and direct output
comparison with the PyTorch reference. Until that invocation gate passes, the
operator must stay outside production.

## Next Direction

Build a standalone runtime invocation gate:

1. install the generated package into a temporary `ASCEND_CUSTOM_OPP_PATH`;
2. determine the correct PyTorch/ACL invocation path for
   `FusedPressureUpdateForward`;
3. compare custom-op output against the PyTorch pressure-update reference on
   the same tiny tensor case;
4. only after exact or tolerance-approved parity, run a small timing comparison.

If the runtime invocation path cannot be made reproducible, stop the custom-op
line instead of editing production acoustic kernels.
