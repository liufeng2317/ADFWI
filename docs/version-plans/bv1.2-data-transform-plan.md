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
- `DataMask`.

The initial tensor convention is waveform data shaped `[shot, time, receiver]`.
A receiver mask shaped `[shot, receiver]` is expanded to `[shot, 1, receiver]`
for broadcasting.

## Deferred Scope

The following transforms are intentionally deferred because they currently depend
on extra CPU/NumPy utilities or require additional physics-specific validation:

- offset mute;
- first-arrival or late-window mute;
- low-pass filtering;
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

This phase does not alter `AcousticFWI` or `ElasticFWI` behavior.
