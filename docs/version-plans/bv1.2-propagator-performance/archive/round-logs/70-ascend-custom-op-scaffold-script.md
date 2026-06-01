# Ascend Custom-Op Scaffold Script

Date: 2026-06-01

## Purpose

This record turns the previous manual Ascend/CANN custom-op scaffold probe into
a reproducible script. The goal is still feasibility gating, not production
kernel implementation.

No `ADFWI/propagator` runtime code was changed in this step.

## Script

```text
scripts/benchmark/ascend_custom_op_scaffold_probe.py
```

The script:

1. writes a minimal `FusedPressureUpdateForward` IR JSON;
2. runs `msopgen gen` for `ai_core-ascend910b`;
3. checks that the generated project contains the expected host/kernel files;
4. patches generated build scripts away from base `python3` to the active
   `adfwi` Python;
5. records environment dependencies needed by the build path;
6. skips compile explicitly when `google.protobuf` is missing.

This keeps the custom-op route measurable without silently relying on shell
state or temporary manual edits.

## Commands

Syntax check:

```bash
conda run -n adfwi python -m py_compile \
  scripts/benchmark/ascend_custom_op_scaffold_probe.py
```

Generation-only probe:

```bash
conda run -n adfwi python scripts/benchmark/ascend_custom_op_scaffold_probe.py \
  --output docs/version-plans/bv1.2-propagator-performance/ascend_custom_op_scaffold_probe_20260601.json
```

Compile-gated probe:

```bash
conda run -n adfwi python scripts/benchmark/ascend_custom_op_scaffold_probe.py \
  --compile \
  --output docs/version-plans/bv1.2-propagator-performance/ascend_custom_op_scaffold_probe_compile_skip_20260601.json
```

## Results

| Probe | Status | Key result |
| --- | --- | --- |
| generation only | `ok` | `msopgen` generated the required project files |
| compile requested | `compile_skipped` | skipped because `google.protobuf` is missing in `adfwi` |
| compile after protobuf install | `ok` | generated scaffold compiled and packaged successfully |

Generated required files were present in both runs:

```text
CMakeLists.txt
CMakePresets.json
build.sh
op_host/fused_pressure_update_forward.cpp
op_host/fused_pressure_update_forward_tiling.h
op_kernel/fused_pressure_update_forward.cpp
```

Generated Python build references patched successfully:

```text
CMakePresets.json
build.sh
cmake/util/ascendc_compile_kernel.py
```

Dependency check:

```text
numpy: true
google: false
google.protobuf: false
```

After installing `protobuf` in the `adfwi` environment:

```text
google.protobuf: true
protobuf version: 5.29.3
compile returncode: 0
```

The generated scaffold package was created successfully:

```text
custom_opp_ubuntu_aarch64.run
```

## Decision

The Ascend custom-op route is now feasible at the generated scaffold compile
level.

The environment-readiness gate is cleared:

```text
conda run -n adfwi python scripts/benchmark/ascend_custom_op_scaffold_probe.py --compile
```

The next gate is no longer package generation. The next gate is to replace the
generated placeholder kernel body with the smallest real pressure-update kernel
and validate it against the PyTorch reference on a tiny deterministic tensor
case before connecting it to any production propagator path.

## Next Direction

Continue the lower-level fused-stencil line with a separate prototype kernel.
Do not edit `ADFWI/propagator/acoustic_kernels.py` until a standalone custom op
matches the PyTorch pressure update numerically and demonstrates a meaningful
runtime benefit on NPU.
