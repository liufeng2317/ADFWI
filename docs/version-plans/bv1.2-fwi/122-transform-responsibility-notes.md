# 122 - Transform Responsibility Notes

## Optimization Path

Focus on `ADFWI.fwi.transforms` after confirming its module split is already
reasonable. This pass does not split modules or move code; it only clarifies the
meaning of the existing transform APIs.

## Change

- Clarified `ADFWI.fwi.transforms` as the public waveform transform API for FWI
  loss preparation.
- Documented that most transforms preserve synthetic/observed tensor shape and
  compose through `DataTransformPipeline`.
- Documented why `select_or_mask_receivers(...)` is exported but remains a
  standalone helper: it can gather active receiver traces and change the receiver
  dimension.
- Clarified `LowPassFilter` as the torch-native FIR path and
  `LegacyLowPassFilter` as the historical multiscale-compatible path.
- Clarified `LegacyOffsetMute` and `LegacyLateWindowMute` as wrappers around
  historical mute utilities rather than redesigned torch-native operators.

## Scientific Contract

- No function body changed.
- No transform order, tensor operation, filter kernel, mute behavior, receiver
  selection behavior, normalization formula, loss input shape, or gradient path
  changed.
- This is a documentation and responsibility-contract pass only.

## Validation

Completed validation:

- `conda run -n adfwi python -m py_compile ADFWI/fwi/transforms/__init__.py ADFWI/fwi/transforms/filters.py ADFWI/fwi/transforms/mutes.py ADFWI/fwi/transforms/receivers.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_data_transforms.py tests/test_import_surface_policy.py`: passed.
- `git diff --check`: passed.

## Next Optimization Direction

Do not restructure `ADFWI.fwi.transforms`. If more work is needed here, prefer
small user-facing documentation or examples that show the recommended transform
pipeline and the legacy-compatible options.
