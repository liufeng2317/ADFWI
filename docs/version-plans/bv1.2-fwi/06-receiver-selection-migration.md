# bv1.2 Receiver Selection Migration

## Goal

Organize the receiver-mask / trace-missing path without changing FWI numerical
behavior. This is separate from `DataMask`: receiver selection can change the
receiver dimension, while `DataMask` only multiplies samples after synthetic and
observed tensors already have matching shapes.

## Existing Behavior

ADFWI currently has two receiver-mask modes in both acoustic and elastic FWI:

1. **Partial-data mask, same shape**

   Synthetic and observed tensors both keep the full receiver dimension
   `[shot, time, receiver]`. The synthetic tensor is multiplied by
   `receiver_masks_3D[shot_index]`. Observed data may already be masked during
   FWI initialization when `receiver_masks_obs=True`.

2. **Trace-missing selection, different shape**

   Observed tensors only contain active traces. Synthetic tensors still come
   from the propagator with the full receiver dimension. A per-shot receiver mask
   is used to select active receiver traces from synthetic data, producing a
   tensor shaped like the observed data.

## Design Decision

Do not force this operation into `DataTransformPipeline` yet. The current
pipeline validates that synthetic and observed tensors have the same shape before
running transforms. Trace-missing selection is the step that makes the shapes
match, so it must run before the regular transform pipeline.

Instead, add a shared helper under `ADFWI.fwi.transforms.waveform`:

```python
select_or_mask_receivers(synthetic, observed, receiver_mask)
```

The helper preserves the two existing modes:

- if `synthetic.shape == observed.shape`, apply a broadcast receiver mask;
- otherwise, gather active receiver traces per shot and return a synthetic tensor
  shaped like `observed`.

## Validation Plan

- Add unit tests comparing the helper against the legacy acoustic/elastic branch
  logic for same-shape masking and trace-missing selection.
- Replace duplicate acoustic forward/closure selection code with the helper.
- Replace `ElasticFWI.real_case_data_selecting()` internals with the helper while
  preserving the method as a compatibility wrapper.
- Run backend integration and mini smoke tests after migration.

## Implementation

- Added `select_or_mask_receivers()` in `ADFWI.fwi.transforms.waveform`.
- Exported the helper from `ADFWI.fwi.transforms`.
- Updated `AcousticFWI.forward()` and `AcousticFWI.forward_closure()` to use the
  helper before the regular data-transform pipeline.
- Updated `ElasticFWI.real_case_data_selecting()` to delegate to the helper,
  keeping the public method as the existing compatibility entry point.
- Added `--receiver-mask-mode {none,mask,select}` to acoustic and elastic mini
  inversion smoke tests.
- Added `trace-missing` to `scripts/smoke/compare_backend_smoke.py`.
- Added unit coverage in `tests/test_receiver_selection.py` and
  `tests/test_smoke_compare.py`.

The helper deliberately keeps receiver selection outside
`DataTransformPipeline`: selecting traces can change tensor shape, while the
pipeline is still a same-shape waveform preprocessing stage.

## Validation Results

Commands run:

```bash
python -m py_compile ADFWI/fwi/transforms/waveform.py ADFWI/fwi/transforms/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_receiver_selection.py
conda run -n adfwi python -m unittest tests/test_receiver_selection.py tests/test_backend_integration.py
python -m py_compile scripts/smoke/acoustic_mini_inversion_smoke.py scripts/smoke/elastic_mini_inversion_smoke.py scripts/smoke/compare_backend_smoke.py tests/test_smoke_compare.py
conda run -n adfwi python -m unittest tests/test_receiver_selection.py tests/test_smoke_compare.py
conda run -n adfwi python -m unittest tests/test_receiver_selection.py tests/test_smoke_compare.py tests/test_mute_transform_comparison.py tests/test_lowpass_transform_comparison.py tests/test_data_transforms.py tests/test_backend_integration.py
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0
```

Results:

- `tests/test_receiver_selection.py` and backend integration: passed.
- `tests/test_receiver_selection.py` and smoke compare unit tests: passed.
- Combined regression `tests/test_receiver_selection.py`, `tests/test_smoke_compare.py`,
  `tests/test_mute_transform_comparison.py`, `tests/test_lowpass_transform_comparison.py`,
  `tests/test_data_transforms.py`, and `tests/test_backend_integration.py`: `49 tests OK`.
- Acoustic trace-missing CPU/NPU comparison: `status=ok`; `loss`,
  `vp_grad_norm`, and `vp_update_norm` all matched exactly for this smoke case.
- Elastic trace-missing CPU/NPU comparison: `status=ok`; maximum relative drift
  was `4.880620563312549e-06` on `loss`, within the comparison threshold
  `rel_tol=1e-05`.

Conclusion: receiver selection is now centralized and covered by both exact
legacy-behavior unit tests and CPU/NPU forward/backward smoke comparisons.

## Non-goals

- Do not change observed-data initialization behavior.
- Do not move trace-missing selection into `DataTransformPipeline` until the
  pipeline can explicitly support shape-changing preprocessing steps.
- Do not alter receiver order. Active traces must remain in the order implied by
  `receiver_mask`.
