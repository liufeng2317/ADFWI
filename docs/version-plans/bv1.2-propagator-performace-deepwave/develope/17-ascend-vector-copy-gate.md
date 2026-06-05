# Ascend Vector Copy Gate

## Focus

After the scalar aligned pressure kernel plateaued below PyTorch for 40 shots,
this round tested the next high-value prerequisite:

```text
Can an aligned vector/DataCopy custom op provide a stable memory path?
```

The test intentionally used copy instead of the pressure formula. If vector
copy is not stable, vector pressure should not be attempted in the same ad hoc
kernel scaffold.

Production propagator code was not changed.

## Test

Prototype mode:

```text
copy_vector_aligned_chunks
```

Configuration:

- device: `npu:0`;
- block_dim: `64`;
- warmup: `10`;
- repeats: `50`;
- shapes: `[3,64,80]` and `[40,64,80]`;
- PyTorch reference: `p.clone()`.

## Result

The runtime failed during the 40-shot case with an Ascend vector core exception:

```text
The DDR address of the MTE instruction is out of range
```

The failed JSON record is:

```text
ascend_copy_vector_aligned_forward_benchmark_block64_20260605.json
```

## Interpretation

The current quick DataCopy implementation is not a stable path. This is
different from the scalar aligned kernel:

```text
scalar aligned copy
  |
  +-- numerically stable
  |
  +-- too slow at 40 shots

ad hoc vector/DataCopy copy
  |
  +-- runtime crash at 40 shots
```

This means the next useful step is not to keep modifying this quick kernel.
The vector path needs to be rebuilt from an official AscendC tiling/DataCopy
template, with explicit local buffer layout and tail handling.

## Decision

Do not keep `copy_vector_aligned_chunks` as an active benchmark mode.

The failed mode was removed from the scripts to avoid leaving a kernel option
that can crash the NPU runtime.

## Next Step

Choose one of two bounded routes:

1. implement a clean official-template AscendC vector copy first, then pressure;
2. skip single-step micro-kernels and design a fused multi-time-step segment,
   where launch overhead and global-memory traffic can be amortized.

Given the current evidence, route 2 is likely higher value for ADFWI.
