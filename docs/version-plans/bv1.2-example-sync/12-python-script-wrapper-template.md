# Python Script Wrapper Template

## Boundary

```text
Goal:
Align example `.py` script wrappers with the bv1.2 notebook usage style.

Scope:
Start from one representative DR-FWI script:
`examples/DR-FWI/Inductive_bias/imputation_bias/low-frequency-missing/Marmousi2-nowater-f0=10/01_0_inversion-vp-baseline.py`

Stop:
Do not change inversion parameters, model geometry, survey geometry, optimizer,
loss function, regularization strength, iteration count, or saved result names.
```

## Applied Pattern

- Remove repo-root `sys.path.append(...)`.
- Replace `from ADFWI... import *` with the smallest explicit imports required
  by the script.
- Configure the ADFWI backend once:

```python
device = "npu:0"
dtype = torch.float32
backend = ADFWI.set_backend(device, dtype=dtype)
```

- Let `AcousticModel` and `AcousticPropagator` inherit the active backend
  instead of passing `device=device` or `dtype=dtype`.
- Use script-local paths based on `Path(__file__).resolve().parent`, so the
  script can be launched from any working directory.
- Keep script-specific Matplotlib backend handling in the script, and set it
  before importing `matplotlib.pyplot`.

## Validation

```text
conda run -n adfwi python -m py_compile \
  examples/DR-FWI/Inductive_bias/imputation_bias/low-frequency-missing/Marmousi2-nowater-f0=10/01_0_inversion-vp-baseline.py
```

Result: passed.

The same wrapper rule was then applied to all discovered inversion Python
scripts under `examples/`:

```text
inversion_py_files 1072
py_compile_bad 0
old_sys_path_append 0
old_adfwi_wildcard_imports 0
missing_import_adfwi_for_set_backend 0
cuda_device_binding_left 0
```

`examples/validation/marmousi2_acoustic_reduced/scripts/inversion.py` is a special
validation driver that imports runtime modules through
`forward_modeling.import_runtime_modules()`. It was left in its existing
validated structure instead of being forced into the simple standalone script
template.

## Git Boundary

This representative script is currently ignored by `.gitignore`, so the code
change is local unless the file is explicitly force-added. Do not force-add a
large set of DR-FWI research scripts without a release-scope decision.

The repository already tracks many DR-FWI inversion scripts, so those tracked
files can be committed normally with `git add -u`. Ignored research scripts are
updated locally by the batch pass but remain outside git unless explicitly
force-added.

## Next Direction

Apply the same per-file analysis to tracked `.py` scripts first. For ignored
DR-FWI research scripts, decide whether they are release examples or local
experiment drivers before committing them.
