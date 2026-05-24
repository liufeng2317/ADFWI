# bv1.2 Data Transform Pipeline Plan

## Motivation

`AcousticFWI` and `ElasticFWI` currently mix inversion-loop logic with waveform
processing logic. This includes trace normalization, receiver masks, data masks,
offset mute, first-arrival mute, and low-pass filtering. As bv1.2 adds CPU/NPU
backend support and more smoke tests, these processing steps should become
composable, testable units instead of being hidden inside each FWI class.

The goal is to create a small transform pipeline that can eventually be shared
by acoustic and elastic workflows while preserving existing constructor
arguments during migration.

## Phase 1 Scope

Create a low-risk, pure torch transform module without changing the existing FWI
loops yet. The first phase only includes operations that are tensor-local and
straightforward to validate on CPU/NPU:

- `DataTransform` base interface;
- `DataTransformPipeline` composition;
- `TraceNormalize`;
- `ReceiverMask`;
- `DataMask`;
- `LowPassFilter` as an experimental pure torch FIR low-pass transform.

The initial tensor convention is waveform data shaped `[shot, time, receiver]`.
A receiver mask shaped `[shot, receiver]` is expanded to `[shot, 1, receiver]`
for broadcasting.

## Deferred Scope

The following transforms are intentionally deferred because they currently depend
on extra CPU/NumPy utilities or require additional physics-specific validation:

- offset mute;
- first-arrival or late-window mute;
- mute-window transforms and pure torch replacement of legacy low-pass filtering;
- real-case receiver selection where observed and synthetic receiver counts differ.

## Proposed API

```python
from ADFWI.fwi.transforms import (
    DataTransformPipeline,
    TraceNormalize,
    ReceiverMask,
    DataMask,
)

pipeline = DataTransformPipeline([
    ReceiverMask(receiver_mask),
    DataMask(data_mask),
    TraceNormalize(),
])

syn, obs = pipeline(syn, obs)
```

Masks can also be supplied through context when they are produced dynamically by
the FWI loop:

```python
pipeline = DataTransformPipeline([ReceiverMask(), DataMask(), TraceNormalize()])
syn, obs = pipeline(syn, obs, context={
    "receiver_mask": receiver_mask,
    "data_mask": data_mask,
})
```

## Validation

Phase 1 should be accepted when the following tests pass:

```bash
conda run -n adfwi python -m unittest tests/test_data_transforms.py
```

Required behavior:

- empty pipeline returns the original tensor pair;
- shape mismatches fail clearly;
- trace normalization keeps all-zero traces at zero and avoids NaN/Inf;
- receiver masks broadcast from `[shot, receiver]`;
- data masks apply sample-level masking;
- masks can be supplied through `context`;
- CPU and NPU tensors keep device and dtype.

## Migration Path

1. Add transform module and tests.
2. Add optional `data_transform_pipeline=None` to `AcousticFWI` while preserving
   existing `waveform_normalize`, receiver mask, data mask, mute, and low-pass
   arguments.
3. Internally route the low-risk parts of `AcousticFWI.calculate_loss` through
   transforms.
4. Repeat for `ElasticFWI` after acoustic behavior is stable.
5. Move offset mute, first-arrival mute, and low-pass filtering into transforms
   only after separate CPU/NPU validation.

## Implementation Status

Implemented in bv1.2 phase 1:

- `ADFWI/fwi/transforms/base.py`;
- `ADFWI/fwi/transforms/waveform.py`;
- `tests/test_data_transforms.py`.

Phase 1 did not alter `AcousticFWI` or `ElasticFWI` behavior.

Phase 2 started with a conservative optional integration in `AcousticFWI`:

- `AcousticFWI(..., data_transform_pipeline=None, waveform_normalize=True)` now builds an internal `DataTransformPipeline([LegacyLowPassFilter(required=False), DataMask(required=False, apply_to="synthetic"), TraceNormalize()])`.
- The default `DataMask` replaces the old synthetic-waveform `syn_p * data_mask` branch while preserving the existing observed-data masking performed during initialization.
- If a custom pipeline is provided, `AcousticFWI` still prepends the compatibility `DataMask(required=False, apply_to="synthetic")` so existing `obs_data.data_masks` behavior is not accidentally skipped.
- The legacy `_normalize()` branch is bypassed for the default normalization path, but kept as a compatibility fallback for explicit/manual calls.
- When a custom pipeline is provided, it is applied to `(synthetic, observed)` after mute/filter steps and before any remaining legacy normalization branch.
- Existing arguments such as `waveform_normalize`, receiver masks, data masks, offset mute, late-window mute, and `cutoff_freq` low-pass filtering remain supported.
- Users can set `waveform_normalize=False` when `TraceNormalize()` is included in a custom pipeline to avoid double normalization.
- Receiver-mask migration is deferred because the trace-missing branch can change receiver dimensions before loss calculation.

Elastic integration started after the acoustic path was validated:

- `ElasticFWI(..., data_transform_pipeline=None, waveform_normalize=True)` now uses the same default `LegacyLowPassFilter(required=False)`, `DataMask(required=False, apply_to="synthetic")`, and `TraceNormalize()` pipeline.
- `ElasticFWI.real_case_data_selecting()` now handles receiver/trace selection only; sample-level `data_masks` are applied in `calculate_loss()` through the transform pipeline for pressure, vx, and vz components.
- `LegacyLowPassFilter` now preserves exact legacy `multiScaleProcessing.lpass` numerics inside the default AcousticFWI/ElasticFWI transform pipeline. `calculate_loss(..., cutoff_freq=...)` passes cutoff settings through pipeline context instead of using a standalone low-pass branch. Pure torch `LowPassFilter` remains experimental because inversion smoke comparisons show non-equivalent loss/gradient values versus legacy filtering.
- Mute windows and trace-missing receiver selection remain in the legacy path until they have separate CPU/NPU validation.
