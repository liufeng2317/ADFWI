# Ascend Pressure BlockDim Sweep

## Focus

The previous benchmark showed that `pressure_aligned_chunks` was fast for a
small batch but slow for 40 shots. This round tested the most direct hypothesis:
the prototype kernel was under-parallelized because `block_dim=8` split the
full tensor into too few chunks.

Production propagator code was not changed.

## Tests

All tests used:

- device: `npu:0`;
- kernel: `pressure_aligned_chunks`;
- warmup: 10;
- repeats: 50;
- numerical gate: `atol=1e-6, rtol=1e-6`.

## Results

| BlockDim | Shape `[shot,nz,nx]` | PyTorch median | Ascend median | Speedup | Max abs diff |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | `[3,64,80]` | `6.89e-04 s` | `4.02e-04 s` | `1.71x` | `2.38e-07` |
| 8 | `[40,64,80]` | `6.47e-04 s` | `2.68e-03 s` | `0.24x` | `2.38e-07` |
| 32 | `[3,64,80]` | `6.76e-04 s` | `3.94e-04 s` | `1.72x` | `2.38e-07` |
| 32 | `[40,64,80]` | `6.58e-04 s` | `9.62e-04 s` | `0.68x` | `2.38e-07` |
| 64 | `[3,64,80]` | `6.97e-04 s` | `4.34e-04 s` | `1.60x` | `2.38e-07` |
| 64 | `[40,64,80]` | `6.54e-04 s` | `9.62e-04 s` | `0.68x` | `2.38e-07` |

## Interpretation

Increasing `block_dim` fixes part of the 40-shot bottleneck:

```text
40-shot speedup:
  block_dim=8   -> 0.24x
  block_dim=32  -> 0.68x
  block_dim=64  -> 0.68x
```

The plateau at 32/64 means simple parallelism tuning is not enough. The
remaining gap is likely from the scalar global-memory loop:

```text
for each element:
  GetValue(...)
  GetValue(...)
  ...
  SetValue(...)
```

This is not an efficient AscendC execution model for larger shot batches.

## Decision

Stop tuning the scalar aligned pressure kernel.

The current prototype is useful as a correctness and wrapper boundary, but it
is not a production performance candidate.

## Next Step

Use one of two higher-value routes:

1. implement a tiled/vectorized AscendC pressure update that loads contiguous
   data through local tensors and performs vector operations where possible;
2. if vectorizing a single pressure update is too limited, move to a fused
   multi-time-step segment so launch overhead and memory traffic are amortized
   across recurrence steps.

Do not integrate `pressure_aligned_chunks` into `ADFWI/propagator`.
