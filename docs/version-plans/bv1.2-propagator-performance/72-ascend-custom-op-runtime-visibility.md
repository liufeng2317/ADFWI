# Ascend Custom-Op Runtime Visibility Gate

Date: 2026-06-01

## Purpose

This gate checks whether the compiled `FusedPressureUpdateForward` custom-op
package is directly callable from the current Python/PyTorch NPU runtime.

This is still outside production. No `ADFWI/propagator` runtime code was
changed.

## Script

```text
scripts/benchmark/ascend_custom_op_runtime_probe.py
```

The script performs the full standalone runtime visibility sequence:

1. rebuilds the minimal pressure-update custom-op package;
2. installs the package into a temporary `ASCEND_CUSTOM_OPP_PATH`;
3. starts a fresh Python process with the generated `set_env.bash` equivalent;
4. imports `torch` and `torch_npu`;
5. checks both PyTorch dispatcher names and `torch.ops.npu` visibility.

## Command

```bash
conda run -n adfwi python -m py_compile \
  scripts/benchmark/ascend_custom_op_runtime_probe.py

conda run -n adfwi python scripts/benchmark/ascend_custom_op_runtime_probe.py \
  --output docs/version-plans/bv1.2-propagator-performance/ascend_custom_op_runtime_probe_20260601.json
```

## Result

| Check | Result |
| --- | --- |
| pressure-update package compile | `ok` |
| package exists | `true` |
| temporary package install | returncode `0` |
| Python runtime probe | returncode `0` |
| `ASCEND_CUSTOM_OPP_PATH` visible | yes |
| custom op visible in PyTorch dispatcher | no |
| custom op visible in `torch.ops.npu` | no |

Runtime visibility payload:

```text
dispatch_matches: []
torch_ops_npu_matches:
  FusedPressureUpdateForward: null
  fused_pressure_update_forward: null
  aclnn_fused_pressure_update_forward: null
```

## Decision

The generated package installs successfully, but it is not automatically exposed
as a Python-callable `torch.ops.npu` operator in this environment.

This means the next useful work is not changing the acoustic propagator. The
next gate is a Python/PyTorch wrapper route:

```text
compiled AscendC kernel package
  -> ACLNN symbols are present in libcust_opapi.so
  -> need a PyTorch extension or ACL runtime wrapper
  -> only then can we compare output against the PyTorch reference
```

Confirmed exported ACLNN symbols from `libcust_opapi.so`:

```text
aclnnFusedPressureUpdateForwardGetWorkspaceSize
aclnnFusedPressureUpdateForward
```

## Next Direction

Build a separate wrapper feasibility gate. The bounded goal is:

1. create a minimal Python-callable wrapper around the generated ACLNN API;
2. call it on tiny NPU tensors;
3. compare output against the PyTorch pressure-update reference;
4. stop if wrapper registration is not reproducible.

Do not connect the custom op to `ADFWI/propagator/acoustic_kernels.py` until the
wrapper gate produces a direct numerical parity result.
