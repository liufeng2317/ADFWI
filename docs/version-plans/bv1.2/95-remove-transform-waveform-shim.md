# 95. Remove Transform Waveform Shim

## Optimization Path

Continue the bv1.2 canonical API cleanup after removing root-level FWI shims:

```text
scan transform imports -> remove pure waveform re-export -> validate transform package API
```

`ADFWI.fwi.transforms.waveform` was a pure compatibility re-export module. Active
code, scripts, examples, and tests already import from `ADFWI.fwi.transforms` or
the responsibility-focused transform modules directly.

## Removed File

| Removed path | Replacement |
| --- | --- |
| `ADFWI/fwi/transforms/waveform.py` | `ADFWI.fwi.transforms` or specific modules such as `ADFWI.fwi.transforms.amplitude` |

## Active Import Scan

```bash
rg -n "ADFWI\\.fwi\\.transforms\\.waveform" ADFWI scripts tests examples docs \
  --glob '!tests/full_cases/outputs/**' \
  --glob '!**/__pycache__/**'
```

Result before removal: no active code, script, example, or test imports. Only
historical bv1.2 planning documents referenced the old path.

## Numerical Precision

No FWI numerical path changed. This removes an import surface only. Transform
implementations remain in:

- `ADFWI.fwi.transforms.amplitude`
- `ADFWI.fwi.transforms.filters`
- `ADFWI.fwi.transforms.masks`
- `ADFWI.fwi.transforms.mutes`
- `ADFWI.fwi.transforms.receivers`
- `ADFWI.fwi.transforms.__init__`

## Validation

Focused transform and data-path checks:

```bash
conda run -n adfwi python -m unittest \
  tests/test_data_transforms.py \
  tests/test_backend_integration.py \
  tests/test_fwi_data_contract.py
```

Compile check:

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/fwi/transforms/__init__.py \
  ADFWI/fwi/transforms/amplitude.py \
  ADFWI/fwi/transforms/filters.py \
  ADFWI/fwi/transforms/masks.py \
  ADFWI/fwi/transforms/mutes.py \
  ADFWI/fwi/transforms/receivers.py
```

## Next Direction

Continue the cleanup audit for other pure compatibility surfaces. The remaining
high-risk items are not import shims but legacy numerical methods, especially
`LegacyLowPassFilter`, `Misfit_waveform_L2`, and `GradProcessor`; those require
focused numerical comparisons before any replacement.
