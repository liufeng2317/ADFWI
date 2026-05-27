# 96. Remove Data Normalize Re-Export

## Optimization Path

Continue canonical API cleanup by removing the data-layer re-export of waveform
normalization:

```text
ADFWI.fwi.data.normalize_waveform -> ADFWI.fwi.transforms.amplitude.normalize_waveform
```

`normalize_waveform` is a waveform transform helper, not a data-contract helper.
The data package still uses it internally for loss evaluation, but it no longer
exports it as part of `ADFWI.fwi.data`.

## Updated Paths

| Path | Change |
| --- | --- |
| `ADFWI/fwi/data/__init__.py` | Remove `normalize_waveform` import and `__all__` export. |
| `ADFWI/fwi/acoustic_fwi.py` | Import `normalize_waveform` from `ADFWI.fwi.transforms.amplitude`. |
| `ADFWI/fwi/elastic_fwi.py` | Import `normalize_waveform` from `ADFWI.fwi.transforms.amplitude`. |
| `tests/test_fwi_data_contract.py` | Use canonical transform-layer normalization import. |

## Preserved Behavior

The implementation remains `ADFWI.fwi.transforms.amplitude.normalize_waveform`.
Only the package-level re-export is removed. FWI drivers still call the same
function in their `_normalize()` fallback path.

## Validation

Focused tests:

```bash
conda run -n adfwi python -m unittest \
  tests/test_fwi_data_contract.py \
  tests/test_fwi_runtime.py \
  tests/test_backend_integration.py \
  tests/test_data_transforms.py
```

Compile check:

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/fwi/data/__init__.py \
  ADFWI/fwi/data/loss.py \
  ADFWI/fwi/acoustic_fwi.py \
  ADFWI/fwi/elastic_fwi.py
```

## Numerical Precision

No numerical formula changed. This is an import-surface cleanup only. Full-case
rerun is not required unless a later commit changes loss preparation or
normalization behavior.

## Next Direction

Continue scanning package `__init__` exports for symbols that belong to a more
specific canonical module, but keep normal package-level public APIs that are
part of the bv1.2 user-facing surface.
