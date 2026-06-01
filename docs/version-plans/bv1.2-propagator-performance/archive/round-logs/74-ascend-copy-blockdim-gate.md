# 74 - Ascend Copy/BlockDim Correctness Gate

## Goal

Separate three possible error sources in the AscendC pressure-update prototype:

1. PyTorch/NPU wrapper wiring,
2. basic global-memory read/write,
3. pressure-update formula and block partitioning.

This round still does not modify production propagator code.

## What Changed

- Added `--kernel-mode {pressure,copy}` to `scripts/benchmark/ascend_fused_pressure_update_prototype.py`.
- Added `--block-dim` to the generated host tiling patch.
- Propagated both options through `scripts/benchmark/ascend_pressure_update_wrapper_probe.py`.

The new `copy` kernel mode performs only:

```text
out[index] = p[index]
```

This gives a minimal correctness gate for AscendC global-memory read/write before testing any stencil formula.

## Results

### Copy, block_dim=8

Command:

```bash
conda run -n adfwi python scripts/benchmark/ascend_pressure_update_wrapper_probe.py \
  --kernel-mode copy \
  --wrapper-build-timeout 600 \
  --runtime-timeout 240 \
  --output docs/version-plans/bv1.2-propagator-performance/ascend_pressure_update_copy_wrapper_probe_20260601.json
```

Result:

```text
status = wrapper_numerical_mismatch
max_abs_diff = 2.5216896533966064
```

Interpretation: the previous mismatch is not caused by the pressure-update formula. The multi-block scalar-loop path itself is not numerically correct.

### Copy, block_dim=1

Command:

```bash
conda run -n adfwi python scripts/benchmark/ascend_pressure_update_wrapper_probe.py \
  --kernel-mode copy \
  --block-dim 1 \
  --wrapper-build-timeout 600 \
  --runtime-timeout 240 \
  --output docs/version-plans/bv1.2-propagator-performance/ascend_pressure_update_copy_block1_wrapper_probe_20260601.json
```

Result:

```text
status = ok
max_abs_diff = 0.0
allclose(atol=0, rtol=0) = true
```

Interpretation: the wrapper and basic AscendC global-memory read/write are valid when executed as a single block.

### Pressure Update, block_dim=1

Command:

```bash
conda run -n adfwi python scripts/benchmark/ascend_pressure_update_wrapper_probe.py \
  --kernel-mode pressure \
  --block-dim 1 \
  --wrapper-build-timeout 600 \
  --runtime-timeout 240 \
  --output docs/version-plans/bv1.2-propagator-performance/ascend_pressure_update_pressure_block1_wrapper_probe_20260601.json
```

Result:

```text
status = ok
max_abs_diff = 2.9802322387695312e-08
allclose(atol=1e-6, rtol=1e-6) = true
```

Interpretation: the pressure-update formula is consistent with the PyTorch reference at float32 precision when the block partition is removed.

## Decision

The current production-facing blocker is not the formula and not Python/NPU wrapper wiring. It is the multi-block AscendC execution strategy used by the scalar `GlobalTensor::GetValue/SetValue` loop.

Do not connect this custom-op path to production yet.

## Next Step

Continue the same line with a narrow kernel-level task:

1. replace the scalar multi-block loop with a standard AscendC vectorized copy/update pattern using local tensors and explicit tiling,
2. first require `copy + block_dim > 1` to pass exact parity,
3. then re-enable the pressure-update formula and require `max_abs_diff <= 1e-6`.

Only after that should the work return to performance benchmarking.
