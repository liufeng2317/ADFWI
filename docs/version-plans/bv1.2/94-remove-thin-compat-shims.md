# 94. Remove Thin Compatibility Shims

## Optimization Path

`bv1.1` is retained as the legacy compatibility branch, so `bv1.2` can now move
from compatibility preservation to canonical framework APIs:

```text
old thin import shims -> canonical transform/multiscale imports -> focused tests
```

This removes thin compatibility files while preserving the actual legacy
numerical implementations that still matter for comparison and full-case
baselines.

## Removed Files

| Removed path | Replacement |
| --- | --- |
| `ADFWI/fwi/normalization.py` | `ADFWI.fwi.transforms.amplitude.normalize_waveform` or `ADFWI.fwi.transforms.normalize_waveform` |
| `ADFWI/fwi/multiScaleProcessing.py` | `ADFWI.fwi.multiscale` |

## Updated Code Paths

| Path | Change |
| --- | --- |
| `ADFWI/fwi/data/loss.py` | Import `normalize_waveform` from `ADFWI.fwi.transforms.amplitude`. |
| `tests/test_data_transforms.py` | Use canonical normalization imports only. |
| `tests/test_multiscale_compat.py` | Convert compatibility re-export test into canonical multiscale export test. |
| `tests/test_lowpass_transform_comparison.py` | Import `lpass` from `ADFWI.fwi.multiscale`. |
| `tests/test_mute_transform_comparison.py` | Import `lpass` from `ADFWI.fwi.multiscale`. |
| `ADFWI/fwi/multiscale/legacy_lowpass.py` | Clarify that it owns the historical low-pass behavior, not the removed shim. |

## Preserved Numerical Paths

The following were not removed or changed:

- `ADFWI.fwi.multiscale.legacy_lowpass`
- `ADFWI.fwi.multiscale.lpass`
- `LegacyLowPassFilter`
- `Misfit_waveform_L2`
- `GradProcessor`

This commit removes import compatibility surfaces, not numerical methods.

## Test Comparison

Focused transform and low-pass tests:

```bash
conda run -n adfwi python -m unittest \
  tests/test_data_transforms.py \
  tests/test_multiscale_compat.py \
  tests/test_lowpass_transform_comparison.py \
  tests/test_mute_transform_comparison.py
```

Result:

```text
Ran 29 tests in 11.473s
OK
```

Data contract and runtime tests:

```bash
conda run -n adfwi python -m unittest \
  tests/test_fwi_data_contract.py \
  tests/test_fwi_runtime.py
```

Result:

```text
Ran 51 tests in 0.219s
OK
```

Example smoke comparison:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py \
  --suites examples \
  --devices cpu \
  --example-problems acoustic \
  --example-gradient-processors legacy,torch
```

Result:

```text
status: ok
runs: 2
failed: 0
max_abs_diff: 0.0
max_rel_diff: 0.0
```

Compile check:

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/fwi/data/loss.py \
  ADFWI/fwi/transforms/amplitude.py \
  ADFWI/fwi/transforms/filters.py \
  ADFWI/fwi/multiscale/__init__.py \
  ADFWI/fwi/multiscale/legacy_lowpass.py
```

Result: passed.

## Numerical Precision

No FWI numerical implementation changed. The legacy low-pass implementation is
still owned by `ADFWI.fwi.multiscale.legacy_lowpass`, and focused tests confirm
`LegacyLowPassFilter` still matches `lpass` exactly. Full-case rerun was not
required for this import-surface cleanup.

## Next Direction

Continue removing purely historical import surfaces from `bv1.2` while keeping
actual legacy numerical methods explicit. The next candidate is the thin
`ADFWI.fwi.transforms.waveform` export module, after checking whether examples
or tests still import it directly.
