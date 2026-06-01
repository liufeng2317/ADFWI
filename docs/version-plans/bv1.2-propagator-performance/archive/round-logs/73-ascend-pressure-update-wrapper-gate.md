# 73 - Ascend Pressure Update Wrapper Gate

## Goal

Check whether the compiled AscendC pressure-update prototype can be called from a PyTorch/NPU tensor interface.

This is a feasibility gate only. It does not modify `ADFWI/propagator/acoustic_kernels.py` or the production FWI path.

## What Changed

- Added `scripts/benchmark/ascend_pressure_update_wrapper_probe.py`.
- The script performs four controlled steps:
  1. generate and compile the AscendC pressure-update custom-op package,
  2. install it into a temporary `ASCEND_CUSTOM_OPP_PATH`,
  3. build a minimal `torch_npu` C++ extension wrapper,
  4. run a tiny `[shot, nz, nx] = [2, 9, 10]` numerical comparison against the PyTorch reference.
- Updated the default Ascend compute unit from `ai_core-ascend910b` to `ai_core-ascend910b2`, matching the current device reported by `torch_npu` as `Ascend910B2`.

## Result

Command:

```bash
conda run -n adfwi python scripts/benchmark/ascend_pressure_update_wrapper_probe.py \
  --wrapper-build-timeout 600 \
  --runtime-timeout 240 \
  --output docs/version-plans/bv1.2-propagator-performance/ascend_pressure_update_wrapper_probe_20260601.json
```

Observed result:

- custom-op compile: OK
- custom-op install: OK
- PyTorch NPU wrapper build: OK
- wrapper runtime: OK
- numerical comparison: failed

Numerical mismatch:

```text
max_abs_diff = 2.4019694328308105
allclose(atol=1e-6, rtol=1e-6) = false
```

Additional localization on the generated wrapper workspace showed that the output contains zero blocks where the PyTorch reference retains boundary/input values. The wrapper enters the ACLNN custom-op path, so the current failure is in the AscendC prototype kernel behavior rather than Python import, package visibility, or PyTorch extension wiring.

## Decision

Do not connect this custom-op path to production.

The useful progress is that the full call chain is now proven reachable:

```text
PyTorch Tensor -> torch_npu C++ extension -> generated ACLNN API -> AscendC custom op
```

The blocker is numerical correctness of the kernel prototype.

## Next Step

Stay on this custom-op feasibility line, but narrow the next task to kernel correctness only:

1. replace the current scalar `GlobalTensor::GetValue/SetValue` block loop with a minimal vectorized AscendC copy/update pattern,
2. first validate pure copy `out = p` over the full tensor,
3. only after full-tensor copy parity, re-enable the pressure-update interior formula.

If pure copy cannot pass exact parity, the AscendC route should be paused before spending more time on production integration.
