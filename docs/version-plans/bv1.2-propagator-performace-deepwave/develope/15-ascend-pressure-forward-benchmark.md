# Ascend Pressure Forward Benchmark

## Focus

This round tested whether the aligned Ascend pressure-update prototype is
actually faster than the PyTorch pressure formula for forward-only execution.

The production propagator was not changed.

## Benchmark

Command:

```bash
conda run -n adfwi python scripts/benchmark/ascend_pressure_update_forward_benchmark.py \
  --output docs/version-plans/bv1.2-propagator-performace-deepwave/develope/ascend_pressure_update_forward_benchmark_20260605.json
```

Timing method:

- device: `npu:0`;
- warmup: 10 calls;
- repeats: 50 calls;
- `torch.npu.synchronize()` around each timed call;
- timing excludes custom-op compile/install/wrapper build;
- comparison target: PyTorch slicing implementation of the same pressure update.

## Results

| Shape `[shot, nz, nx]` | PyTorch median | Ascend median | Speedup | Max abs diff |
| --- | ---: | ---: | ---: | ---: |
| `[3, 64, 80]` | `6.89e-04 s` | `4.02e-04 s` | `1.71x` | `2.38e-07` |
| `[40, 64, 80]` | `6.47e-04 s` | `2.68e-03 s` | `0.24x` | `2.38e-07` |

Both cases pass `atol=1e-6, rtol=1e-6`.

## Interpretation

The aligned Ascend kernel is numerically usable as a forward prototype, but its
current implementation does not scale with shot count.

The likely reason is that the current AscendC kernel is still a scalar
global-memory loop using `GetValue` / `SetValue`. It removes PyTorch launch
overhead for small tensors, but it does not provide the vectorized memory and
compute pattern needed for larger shot batches.

This means:

- the wrapper and aligned chunk boundary are useful infrastructure;
- the scalar pressure kernel is not yet a production performance path;
- integrating it into `ADFWI/propagator` now would add complexity without
  reliable speed.

## Decision

Do not integrate this prototype into production.

The next useful optimization must change the kernel execution model, not only
the Python wrapper:

1. replace scalar `GetValue` / `SetValue` with an AscendC tiled/vectorized
   compute pattern;
2. test forward speed on `[3,64,80]` and `[40,64,80]`;
3. only if the 40-shot case beats PyTorch, continue to backward/gradient design.

If vectorized AscendC pressure update is still slower, stop this micro-kernel
route and move to a larger fused time-segment operator, where launch overhead
and memory traffic can be amortized across multiple time steps.
