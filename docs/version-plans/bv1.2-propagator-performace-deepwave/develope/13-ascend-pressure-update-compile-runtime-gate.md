# Ascend Pressure Update Compile/Runtime Gate

## Focus

This round moved from Python custom-autograd prototypes to the lower-level
Ascend/NPU route. The target is a fused pressure-update custom op, which is
closer to Deepwave's useful design: put hot stencil work below the PyTorch
time-step graph.

## Tests

Compile gate:

```bash
conda run -n adfwi python scripts/benchmark/ascend_fused_pressure_update_prototype.py \
  --output docs/version-plans/bv1.2-propagator-performace-deepwave/develope/ascend_fused_pressure_update_compile_20260605.json \
  --kernel-mode pressure \
  --compile \
  --compile-timeout 300
```

Runtime visibility gate:

```bash
conda run -n adfwi python scripts/benchmark/ascend_custom_op_runtime_probe.py \
  --output docs/version-plans/bv1.2-propagator-performace-deepwave/develope/ascend_custom_op_runtime_probe_20260605.json \
  --compile-timeout 300 \
  --install-timeout 120 \
  --runtime-timeout 120
```

## Results

| Gate | Result |
| --- | --- |
| reference formula contract | exact, `max_abs_diff=0.0` |
| AscendC project generation | ok |
| package compile | ok, return code `0` |
| package install | ok, return code `0` |
| PyTorch dispatch visibility | not exposed to `torch.ops.npu` |

Runtime probe detail:

```text
dispatch_matches = []
torch.ops.npu.FusedPressureUpdateForward = null
torch.ops.npu.fused_pressure_update_forward = null
torch.ops.npu.aclnn_fused_pressure_update_forward = null
```

## Interpretation

The lower-level custom-op path is feasible at the CANN package level, but not
yet callable from PyTorch.

This is now the main blocker:

```text
AscendC kernel compiles
  |
  +-- custom OPP package installs
       |
       +-- missing PyTorch/torch_npu dispatch binding
```

## Boundary

Do not continue Python remat or Python segment optimization.

Do not integrate this op into ADFWI until a Python-visible call path exists and
pressure-update parity is tested against the PyTorch reference.

## Next Step

Resolve the PyTorch/NPU call boundary:

1. inspect generated package files under the runtime workspace;
2. determine whether the generated op exposes an ACLNN API, custom OPP only, or
   requires a separate torch extension wrapper;
3. build the smallest Python-callable wrapper;
4. run pressure-update output parity against the PyTorch reference.

Only after that should this route move toward a multi-step pressure segment.
