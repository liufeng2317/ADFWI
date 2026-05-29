# bv1.2 FWI User-Facing Interface

## Goal

Keep the public FWI interface simple while the internal data path becomes more
structured. The bv1.2 refactor adds a clear pre-loss data contract, transform
pipeline, receiver selection helper, and elastic component weights, but the
default user workflow should remain close to existing notebooks.

## Recommended Default Usage

Acoustic pressure-only FWI remains unchanged for typical users:

```python
fwi = AcousticFWI(
    propagator=propagator,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    obs_data=obs_data,
)
```

Elastic pressure-only FWI also remains the default:

```python
fwi = ElasticFWI(
    propagator,
    model,
    loss_fn,
    obs_data,
    optimizer=optimizer,
    scheduler=scheduler,
    gradient_processor=gradient_processor,
    inversion_component=["pressure"],
)
```

The default internal component weight is `1.0`, so omitting
`component_weights` preserves earlier behavior.

## Multi-Component Elastic Usage

When using multiple elastic components, name them explicitly and optionally set
relative loss weights:

```python
fwi = ElasticFWI(
    propagator,
    model,
    loss_fn,
    obs_data,
    optimizer=optimizer,
    scheduler=scheduler,
    gradient_processor=gradient_processor,
    inversion_component=["pressure", "vx", "vz"],
    component_weights={"pressure": 1.0, "vx": 0.5, "vz": 0.5},
)
```

Rules:

- Supported elastic components are `pressure`, `vx`, and `vz`.
- Missing active component weights default to `1.0`.
- Unknown component names and negative weights raise `ValueError`.
- Elastic pressure is always defined as `-(txx + tzz)`.

## Transform Pipeline Usage

Most users can continue using constructor flags such as `waveform_normalize`,
`waveform_mute_offset`, `waveform_mute_late_window`, and `cutoff_freq` passed to
`forward(...)`. bv1.2 routes these through the default transform pipeline while
preserving legacy numerical behavior.

Advanced users can provide an extra `DataTransformPipeline`. It is appended after
the compatibility transforms:

```python
from ADFWI.fwi.transforms import DataTransformPipeline, TraceNormalize

custom_pipeline = DataTransformPipeline([TraceNormalize()])
fwi = AcousticFWI(..., data_transform_pipeline=custom_pipeline, waveform_normalize=False)
```

If a custom pipeline already includes normalization, set `waveform_normalize=False`
to avoid double normalization.

## Internal Boundary

`ADFWI.fwi.data` is an internal orchestration layer for the pair that enters the
misfit. It may prepare synthetic/observed pairs, build transform context, select
receivers, and organize elastic components. It should not contain concrete
filter/mute/mask algorithms, misfit formulas, propagator logic, or file IO.

## API Hygiene

Constructor defaults avoid shared mutable lists. Regularization weight defaults
and the elastic default `inversion_component` are created per instance, so one
FWI object cannot accidentally mutate another object's defaults. User-facing
behavior remains the same: omitting these arguments still gives zero
regularization weights and pressure-only elastic inversion.

## Validation

The interface is covered by the same tests used for the data-contract refactor:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_smoke_compare.py
conda run -n adfwi python -m unittest tests/test_backend_integration.py tests/test_fwi_data_contract.py
conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_receiver_selection.py tests/test_smoke_compare.py tests/test_mute_transform_comparison.py tests/test_lowpass_transform_comparison.py tests/test_data_transforms.py tests/test_backend_integration.py
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problem elastic --case weighted-pressure --devices cpu,npu:0
```

The key expectation is default-preserving behavior: pressure-only acoustic and
elastic runs should match previous CPU/NPU smoke baselines, while weighted
elastic cases should scale loss and gradients according to the supplied weights.
