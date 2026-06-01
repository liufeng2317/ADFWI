# 75 - Ascend Multi-Block Copy Attempt

## Goal

Continue the AscendC correctness line with a strict boundary:

- do not touch production propagator code,
- do not benchmark FWI,
- only test whether the failed `copy + block_dim=8` path can be made correct.

The previous gate showed:

- `copy + block_dim=1`: exact parity,
- `pressure + block_dim=1`: float32 parity,
- `copy + block_dim=8`: numerical mismatch.

## Attempted Change

The first hypothesis was that `GetBlockNum()` did not match the host `SetBlockDim(8)` in this generated custom-op path. I tested an explicit block count strategy by compiling the requested block count directly into the prototype kernel.

This was intentionally kept inside the standalone benchmark/probe scripts and was not connected to production.

## Result

Command:

```bash
conda run -n adfwi python scripts/benchmark/ascend_pressure_update_wrapper_probe.py \
  --kernel-mode copy \
  --block-dim 8 \
  --compile-timeout 600 \
  --wrapper-build-timeout 600 \
  --runtime-timeout 240 \
  --output docs/version-plans/bv1.2-propagator-performance/ascend_pressure_update_copy_block8_const_wrapper_probe_20260601.json
```

Result:

```text
custom-op compile: ok
install: ok
wrapper build: ok
runtime: failed
```

The runtime probe timed out before completing `torch_npu`/wrapper execution. This makes the explicit compiled block-count variant unsuitable as a correctness fix.

## Decision

Do not keep the explicit compiled block-count kernel change.

The benchmark scripts keep timeout handling so failed compile/runtime gates are recorded instead of aborting without a report, but the kernel prototype itself is restored to the previous safe experimental state.

## Next Step

The next meaningful path is not another scalar-loop tweak. It should be a standard AscendC vectorized copy kernel:

1. use local tensor buffers and `DataCopy`,
2. handle block partition with explicit per-core offset/count,
3. first require `copy + block_dim=8` exact parity,
4. only then re-enable the pressure update.

If vectorized copy cannot pass, pause the AscendC custom-op route.
