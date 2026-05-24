# bv1.2 Transform Module Organization

## Goal

Keep the transform pipeline as the single composition interface while avoiding a
large catch-all `transforms/waveform.py` implementation file. The previous file
had started to mix amplitude normalization, filtering, masks, mute operations,
and receiver selection.

## Organization

The package now separates transform responsibilities into smaller modules:

- `base.py`: `DataTransform` and `DataTransformPipeline`;
- `amplitude.py`: amplitude-only transforms such as `TraceNormalize`;
- `filters.py`: differentiable and legacy low-pass filters;
- `masks.py`: same-shape receiver/data mask transforms;
- `mutes.py`: legacy offset and late-window mute transforms;
- `receivers.py`: receiver selection helpers that can match synthetic traces to
  missing-trace observed data;
- `_utils.py`: shared internal helpers for context and mask broadcasting;
- `waveform.py`: backward-compatible re-export module for the historical import
  path.

## Compatibility

The public import path remains stable:

```python
from ADFWI.fwi.transforms import DataTransformPipeline, TraceNormalize
from ADFWI.fwi.transforms.waveform import TraceNormalize
```

Both forms continue to work. New code should prefer `ADFWI.fwi.transforms` as the
public API and use the responsibility modules only when working inside the
package.

## Design Notes

`DataTransformPipeline` remains the unified orchestration layer. This refactor is
only a module-layout change and should not change numerical behavior.

Receiver selection still lives outside the same-shape pipeline during FWI
execution because it can change the receiver dimension before loss calculation.
The helper is placed in `receivers.py` to make that distinction explicit.

## Validation Plan

- Compile all transform modules and FWI callers.
- Re-run receiver selection, mask, mute, low-pass, backend integration, and smoke
  compare unit tests.
- Re-run at least one CPU/NPU trace-missing comparison if transform imports are
  touched by smoke scripts.

## Validation Results

Commands run:

```bash
python -m py_compile ADFWI/fwi/transforms/_utils.py ADFWI/fwi/transforms/amplitude.py ADFWI/fwi/transforms/filters.py ADFWI/fwi/transforms/masks.py ADFWI/fwi/transforms/mutes.py ADFWI/fwi/transforms/receivers.py ADFWI/fwi/transforms/waveform.py ADFWI/fwi/transforms/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_receiver_selection.py tests/test_smoke_compare.py
conda run -n adfwi python -m unittest tests/test_receiver_selection.py tests/test_smoke_compare.py tests/test_mute_transform_comparison.py tests/test_lowpass_transform_comparison.py tests/test_data_transforms.py tests/test_backend_integration.py
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0
```

Results:

- Transform modules and FWI callers compiled successfully.
- Combined transform/backend regression passed: `49 tests OK`.
- Acoustic trace-missing CPU/NPU smoke comparison passed with zero drift for
  `loss`, `vp_grad_norm`, and `vp_update_norm`.
- Elastic trace-missing CPU/NPU smoke comparison passed; maximum relative drift
  was `4.880620563312549e-06` on `loss`, within `rel_tol=1e-05`.

Conclusion: the module split is an organization-only refactor. The public API and
validated numerical behavior remain stable.
