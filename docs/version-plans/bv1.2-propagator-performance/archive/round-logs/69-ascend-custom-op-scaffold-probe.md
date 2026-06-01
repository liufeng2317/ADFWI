# Ascend Custom Op Scaffold Probe

Date: 2026-06-01

## Purpose

This round started the next step after Gate 0: verify whether the Ascend/CANN
custom-operator route can scaffold a minimal fused pressure-update operator.

No ADFWI production code was changed.

## Prototype Target

Minimal operator name:

```text
FusedPressureUpdateForward
```

This is intentionally forward-only and narrower than the full acoustic
propagator.

Initial IR JSON shape:

```text
inputs:  p, u, w, kappa1, alpha1
output:  p_next
attr:    free_surface_start
dtype:   fp32
format:  ND
target:  ai_core-ascend910b
```

The goal of this probe was only to test scaffold/build viability, not to
implement the finite-difference formula.

## Scaffold Result

Command shape:

```bash
msopgen gen \
  -i fused_pressure_update_forward.json \
  -f pytorch \
  -c ai_core-ascend910b \
  -out <tmp>/out \
  -op FusedPressureUpdateForward \
  -lan cpp
```

Result:

```text
pass
```

Generated files included:

```text
CMakeLists.txt
CMakePresets.json
build.sh
op_host/fused_pressure_update_forward.cpp
op_host/fused_pressure_update_forward_tiling.h
op_kernel/fused_pressure_update_forward.cpp
framework/CMakeLists.txt
scripts/install.sh
```

Important detail:

```text
msopgen rejects input files that are group/other writable.
The JSON input must be chmod 600, and the temp/project directory should be
chmod 700.
```

## Build Probe Result

The generated project starts compiling, but does not complete in the current
environment without additional Python dependency setup.

### Attempt 1: Default Generated Build

Default `build.sh` uses:

```text
python3
```

On this machine, that resolves to:

```text
/liufeng1afs/software/miniconda3/bin/python3
Python 3.13.11
NumPy 2.4.4
```

Failure:

```text
AttributeError: np.float_ was removed in the NumPy 2.0 release.
```

Interpretation:

```text
CANN/TBE tooling is not compatible with the base Python 3.13 + NumPy 2.x
environment.
```

### Attempt 2: Force `adfwi` Python

The generated project was patched in the temporary directory so that:

```text
ASCEND_PYTHON_EXECUTABLE=/liufeng1afs/software/miniconda3/envs/adfwi/bin/python
HI_PYTHON=/liufeng1afs/software/miniconda3/envs/adfwi/bin/python
```

This uses:

```text
Python 3.9.25
NumPy 1.26.4
```

The NumPy error disappears, but compilation then fails at:

```text
ModuleNotFoundError: No module named 'google'
```

The missing module is required by:

```text
google.protobuf
```

inside CANN's `opc_tool/opc.py`.

## Decision

The Ascend custom-op route is viable enough to scaffold a project, but the
current environment is not yet ready to compile even an empty generated kernel.

This is an environment/setup blocker, not an acoustic formula blocker.

Required before implementation:

```text
1. Ensure the generated custom-op build uses the adfwi Python executable, not
   the base Python 3.13 executable.
2. Install or expose protobuf/google for that Python environment.
3. Re-run the empty scaffold build before writing pressure-update code.
```

## Next Step

Do not implement `fused_pressure_update_forward` yet.

Next useful task:

```text
Create a small, reproducible custom-op scaffold script that:
  1. writes the IR JSON with safe permissions;
  2. runs msopgen gen;
  3. patches generated Python executable settings to the active adfwi Python;
  4. checks required Python modules, especially google.protobuf;
  5. stops before compilation if dependencies are missing.
```

Only after this scaffold script reports a clean environment should the actual
pressure-update kernel be implemented.
