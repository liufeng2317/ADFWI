# Ascend Wrapper Call Boundary

## Focus

This round focused on the blocker from the previous gate: the generated custom
OPP package was not visible through `torch.ops.npu`.

The goal was to determine whether the pressure-update custom op can be called
from Python at all.

## What Was Tested

The wrapper probe builds a minimal `torch_npu` `NpuExtension` that calls the
generated ACLNN API:

```text
Python
  |
  +-- NpuExtension wrapper
       |
       +-- EXEC_NPU_CMD_EXT(aclnnFusedPressureUpdateForward, ...)
            |
            +-- generated libcust_opapi.so
                 |
                 +-- AscendC fused_pressure_update_forward kernel
```

## Results

| Test | Result |
| --- | --- |
| wrapper build | ok |
| wrapper runtime call | ok |
| `copy`, `block_dim=1` | exact |
| `pressure`, `block_dim=1`, shape `[2, 9, 10]` | ok, max abs diff `2.98e-08` |
| `pressure`, `block_dim=1`, shape `[2, 16, 20]` | ok, max abs diff `1.19e-07` |
| `copy`, `block_dim=2/4/8` | mismatch |
| `copy_vector`, `block_dim=8` | mismatch |
| `copy_aligned_chunks`, `block_dim=8` | exact, max abs diff `0.0` |
| `pressure_aligned_chunks`, `block_dim=8`, shape `[2, 16, 20]` | ok, max abs diff `1.19e-07` |

## Interpretation

The Python-callable wrapper problem is solved:

```text
custom OPP package
  |
  +-- libcust_opapi.so exposes ACLNN symbols
  |
  +-- NpuExtension wrapper can call those symbols from Python
```

The first multi-block blocker was inside the AscendC kernel implementation:

```text
block_dim=1
  |
  +-- correct

naive block_dim>1
  |
  +-- partial/mismatched output

32-float aligned chunks
  |
  +-- copy exact
  |
  +-- pressure formula ok at 1e-6
```

The failed copy tests showed the problem before the pressure formula was
involved. The successful aligned-copy and aligned-pressure tests indicate that
the missing output ranges came from unsafe non-aligned block boundaries in the
prototype scalar global-memory write path.

## Boundary

Do not integrate this wrapper into `ADFWI/propagator` yet.

The wrapper can call the generated ACLNN API and the aligned multi-block kernel
now passes the first pressure formula parity gate. It is still a prototype:

- no backward operator exists;
- no FWI loop has used it;
- no performance benchmark has compared it against production pressure updates;
- the current aligned chunk size is a correctness fix, not a tuned tiling
  strategy.

## Next Step

Move from callability to usefulness:

1. keep `pressure_aligned_chunks` as the only active AscendC pressure prototype;
2. benchmark one-step pressure update against the PyTorch formula on realistic
   small/medium shapes;
3. only if forward speed is meaningful, design the matching backward or a
   segment-level compiled operator;
4. do not touch production `ADFWI/propagator` until forward speed and gradient
   parity have both passed.
