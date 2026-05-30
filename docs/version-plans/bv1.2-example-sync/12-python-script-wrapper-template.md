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

## Git Boundary

This representative script is currently ignored by `.gitignore`, so the code
change is local unless the file is explicitly force-added. Do not force-add a
large set of DR-FWI research scripts without a release-scope decision.

## Next Direction

Apply the same per-file analysis to tracked `.py` scripts first. For ignored
DR-FWI research scripts, decide whether they are release examples or local
experiment drivers before committing them.
