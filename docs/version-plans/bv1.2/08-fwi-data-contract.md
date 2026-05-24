# bv1.2 FWI Data Contract

## Goal

Centralize the data preparation step immediately before misfit evaluation while
keeping acoustic and elastic FWI numerical behavior stable. This step is the
contract between propagator outputs, observed data, receiver selection, waveform
transforms, and the final loss function.

## Data Contract

FWI loss evaluation uses waveform tensors shaped:

```text
[shot, time, receiver]
```

Current component mapping:

- Acoustic: `p`;
- Elastic pressure: `-(txx + tzz)` compared with observed pressure
  `-(obs_txx + obs_tzz)`;
- Elastic particle velocity: `vx`, `vz`.

Before a component reaches the misfit function, the pair must satisfy:

1. `synthetic` and `observed` describe the same shots and component.
2. Trace-missing receiver selection has already matched synthetic receiver traces
   to the observed receiver dimension.
3. Same-shape transforms such as mute, low-pass, data mask, and normalization
   can run without changing the component contract.

## Implementation

Added `ADFWI.fwi.data` with two helpers:

- `build_transform_context(...)`: creates the context consumed by transform
  pipelines.
- `prepare_loss_pair(...)`: applies receiver selection first, then applies the
  same-shape `DataTransformPipeline`.
- `elastic_pressure(...)`, `elastic_synthetic_components(...)`, and
  `elastic_observed_components(...)`: centralize the elastic `pressure/vx/vz`
  component contract.
- `normalize_elastic_component_weights(...)`: validates optional elastic
  component loss weights while defaulting active components to `1.0`.

Updated `AcousticFWI` and `ElasticFWI` to use `_prepare_loss_pair()` inside the
forward path. `ElasticFWI` now also builds observed and synthetic components
through the shared component helpers. This removes duplicated pre-loss data
handling from component loops while preserving `calculate_loss()` compatibility. By default,
`calculate_loss()` still only runs the transform pipeline and misfit on the
provided tensors; receiver selection is not silently applied there.

## Compatibility Notes

- `ElasticFWI` accepts optional `component_weights`; omitting it preserves
  the previous `1.0` weight for every active component.
- Elastic smoke tests expose `--components` and `--component-weights` so weighted
  component behavior can be fixed in CI/smoke baselines.
- `calculate_loss(..., apply_transforms=True)` keeps the historical expectation
  that its input tensors are already receiver-aligned.
- Internal forward paths now call `calculate_loss(..., apply_transforms=False)`
  after `_prepare_loss_pair()` has already performed receiver selection and
  pipeline preprocessing.
- Elastic `real_case_data_selecting()` remains as a compatibility wrapper around
  receiver selection only.

## Validation

Commands run:

```bash
python -m py_compile ADFWI/fwi/data.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py
conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_receiver_selection.py tests/test_smoke_compare.py
conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_receiver_selection.py tests/test_smoke_compare.py tests/test_mute_transform_comparison.py tests/test_lowpass_transform_comparison.py tests/test_data_transforms.py tests/test_backend_integration.py
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problem elastic --case weighted-pressure --devices cpu,npu:0
/liufeng1afs/software/miniconda3/envs/adfwi/bin/python -c "<pressure=2 CPU scaling check>"
```

Results:

- Data contract, elastic component, and component weight helper tests passed.
- Combined transform/backend regression passed: `59 tests OK`.
- Acoustic trace-missing CPU/NPU smoke comparison passed with zero drift for
  `loss`, `vp_grad_norm`, and `vp_update_norm`.
- Elastic trace-missing CPU/NPU smoke comparison passed; maximum relative drift
  was `4.880620563312549e-06` on `loss`, within `rel_tol=1e-05`.
- Elastic weighted-pressure CPU/NPU smoke comparison passed; maximum relative
  drift was `2.2725155155996827e-06` on `loss`, within `rel_tol=1e-05`.
- CPU pressure weight scaling check passed: `pressure=2.0` produced loss ratio
  `2.0`, gradient norm ratio `2.0`, and update norm ratio `1.99999978078`.

The smoke metrics match the previous receiver-selection validation, so this
refactor is considered numerically stable for the tested CPU/NPU paths.

## Next Step

The next low-risk improvement is to add an explicit multi-component elastic
smoke case, for example `pressure,vx,vz`, with documented expectations for loss
scales and gradient behavior.
