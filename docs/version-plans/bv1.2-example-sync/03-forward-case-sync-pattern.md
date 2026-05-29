# Forward Case Sync Pattern

This document records the reusable update pattern learned from
`examples/acoustic/01-model-test/01-Marmousi2/01_forward.ipynb`.

The purpose is to support later semi-automatic or automatic synchronization of
other forward examples. This is a pattern definition, not a request to rewrite
all examples at once.

## Target Pattern

Forward examples should follow this high-level shape:

```python
import ADFWI

device = "npu:0"          # or "cuda:0" / "cpu"
dtype = torch.float32
backend = ADFWI.set_backend(device, dtype=dtype)

model = AcousticModel(..., nabc=nabc)
propagator = AcousticPropagator(model, survey)
record_waveform = propagator.forward()
```

The key idea is:

```text
set backend once -> model inherits backend -> propagator inherits model backend
```

## Required Updates For Forward Cases

| Area | Old pattern | New pattern |
| --- | --- | --- |
| Package import | `sys.path.append(...)` | Editable install, direct `import ADFWI` |
| Namespace import | `from ADFWI.xxx import *` | Explicit imports used by the case |
| Device declaration | `device = "cuda:0"` only | `device`, `dtype`, `backend = ADFWI.set_backend(...)` |
| Model construction | `AcousticModel(..., device=device, dtype=dtype)` | `AcousticModel(...)` when backend is already set |
| Propagator construction | `AcousticPropagator(model, survey, device=device)` | `AcousticPropagator(model, survey)` |
| Dataset path | Implicit working-directory dependent path | Case-local path that matches intended notebook execution directory |
| Output directory | Repeated `if not exists` blocks | Small local directory loop with `exist_ok=True` |
| Notebook outputs | Runtime warnings and embedded run outputs mixed into diffs | Clear outputs unless intentionally updating reference figures |

## Validation Contract

Before applying this pattern broadly, each forward case should pass the smallest
matching validation:

1. notebook JSON is valid;
2. import/setup cell runs in the `adfwi` environment;
3. `ADFWI.set_backend(device, dtype=dtype)` succeeds;
4. `AcousticModel(..., no device/dtype)` lands tensors on the backend device and
   dtype;
5. `AcousticPropagator(model, survey)` lands damping, wavelet, source, and
   receiver tensors on the expected device and dtype;
6. a short forward run or full intended forward run produces finite waveform
   tensors;
7. output figure changes are reviewed separately from code/notebook logic.

For Marmousi2 acoustic, the backend inheritance check passed:

```text
backend npu npu:0 torch.float32
model.device npu:0
model.dtype torch.float32
model.vp.device npu:0
model.vp.dtype torch.float32
prop.device npu:0
prop.dtype torch.float32
prop.damp.device npu:0
prop.damp.dtype torch.float32
prop.wavelet.device npu:0
prop.wavelet.dtype torch.float32
inherits_backend True
```

## Automation Notes

Automation should be conservative:

- identify candidate notebooks/scripts by searching for `AcousticPropagator`,
  `ElasticPropagator`, `sys.path.append`, and wildcard `ADFWI` imports;
- apply import/backend/model/propagator rewrites only when the pattern is
  unambiguous;
- leave case-specific physics parameters, source/receiver layout, wavelets, and
  plotting choices unchanged;
- do not update generated figures in the same automated patch unless the task is
  explicitly a result refresh;
- record skipped cases and the reason instead of forcing a rewrite.

## Current Forward Case Notes

For `01_forward.ipynb`, the meaningful user-facing changes are:

- direct package import through editable install;
- explicit `import ADFWI`;
- backend declaration with `ADFWI.set_backend(device, dtype=dtype)`;
- model and propagator construction no longer manually pass `device/dtype`;
- dataset path was adjusted for notebook-local execution;
- generated plots were refreshed by running the notebook.

The refreshed plot files should be reviewed and committed separately from
structural notebook edits if they are intended to become reference outputs.

## Next Direction

Use this pattern to synchronize
`examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb`, then build a
small audit script that lists other examples matching the old forward patterns.
