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

## Decision

The Ascend custom-op route remains feasible at the scaffold generation level,
but production implementation should not start yet.

The next gate is environment readiness:

```text
Install or provide google.protobuf inside the adfwi environment, then rerun the
same script with --compile.
```

Only after the generated empty scaffold compiles reproducibly should we write
the fused pressure-update kernel body.

## Next Direction

Continue the lower-level fused-stencil line only if the compile gate is cleared.
Otherwise, stop acoustic Python-kernel optimization and move to another measured
bottleneck. The previous Python rematerialization/cache line has already been
closed because its speed/memory tradeoff is not sufficient for default use.
