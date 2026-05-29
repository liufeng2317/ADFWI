# 00 - Utils Optimization Outline

## Goal

Clarify the role of `ADFWI/utils` before making code changes. This module is a
high-impact support layer because it feeds examples, validation cases, survey
construction, model demos, transforms, and propagator wrappers.

The target is a clearer and more reliable helper layer, not a larger module
tree.

## Scope

Inspect and plan around:

- `ADFWI/utils/__init__.py`
- `ADFWI/utils/utils.py`
- `ADFWI/utils/wavelets.py`
- `ADFWI/utils/velocityDemo.py`
- `ADFWI/utils/first_arrivel_picking.py`
- `ADFWI/utils/offset_mute.py`
- `ADFWI/utils/frequency_domin_process.py`
- `ADFWI/utils/assessment_metric.py`
- `ADFWI/utils/noise.py`

Direct callers in scope:

- examples and validation scripts;
- `ADFWI/survey` source setup;
- `ADFWI/propagator` tensor conversion;
- `ADFWI/fwi/transforms` legacy mute compatibility;
- tests that compare legacy helper behavior.

## Non-Goals

- Do not rename public helper functions in the first round.
- Do not move dataset helpers out of `utils` until examples and notebooks have a
  migration plan.
- Do not change source wavelet formulas without numerical reference outputs.
- Do not change mute, first-arrival picking, or filtering behavior without
  parity tests.
- Do not replace `velocityDemo.py` wholesale in this stage, even though the name
  is not ideal.

## Current Module Roles

| File | Current role | Risk if changed |
| --- | --- | --- |
| `utils.py` | NumPy/list/tensor conversion helpers | device/dtype regressions in propagator/FWI paths |
| `wavelets.py` | source wavelet generation | source signature and waveform amplitude changes |
| `velocityDemo.py` | benchmark model loaders, resamplers, smoothers, synthetic model builders | example and validation case drift |
| `first_arrivel_picking.py` | legacy first-arrival mute helpers | transform parity and FWI mute behavior drift |
| `offset_mute.py` | legacy offset mute helper | transform parity and receiver selection behavior drift |
| `frequency_domin_process.py` | spectrum, high-pass filtering, and plotting helpers | signal-processing output drift |
| `assessment_metric.py` | model-comparison metrics | reported benchmark metric drift |
| `noise.py` | reproducible Gaussian noise injection | data augmentation and synthetic test drift |
| `__init__.py` | public utility import surface | example import breakage |

## Observed Issues

These are audit findings, not automatic tasks:

1. `utils.py` docstrings are too generic and do not state dtype/device behavior.
2. `wavelets.py` has typo-prone public error text and uses `type` as an
   argument name, but that is part of the current public API.
3. `velocityDemo.py` mixes dataset download, file loading, interpolation,
   smoothing, and synthetic model generation in one large file.
4. `velocityDemo.py` still has a stale file header and duplicate
   `get_linear_vel_model` export in `__init__.py`.
5. `first_arrivel_picking.py` has a misspelled filename, but it is used by
   legacy transform parity tests and should not be renamed casually.
6. `frequency_domin_process.py` includes plotting helpers next to filtering
   helpers.
7. Several helpers have numerical edge cases that are currently undocumented,
   such as zero denominators in metrics or device handling in tensor conversion.

## Optimization Strategy

Use small bounded rounds:

1. **Responsibility and import-surface clarification**
   - Add package/module docstrings.
   - Remove obvious duplicate exports only if tests prove no public behavior
     changes.
   - Validation: `py_compile` and import smoke.

2. **Conversion helper contract tests**
   - Lock `numpy2tensor`, `tensor2numpy`, `gpu2cpu`, `list2numpy`, and
     `numpy2list` behavior before changing documentation or internals.
   - Validation: focused unit tests for dtype, device, gradient detach, and list
     conversion behavior.

3. **Wavelet contract tests**
   - Lock current Ricker/Gaussian/Ramp output shapes, dtype, time axis, and
     selected reference values.
   - Validation: focused numerical tests with absolute/relative tolerances.

4. **Legacy mute helper boundary**
   - Keep behavior consistent with `ADFWI.fwi.transforms`.
   - Validation: existing mute transform comparison tests.

5. **Model helper documentation**
   - Clarify data orientation and units in model loaders/resamplers.
   - Avoid formula or interpolation changes unless real-case validation is
     planned.

## First Implementation Candidate

Start with **responsibility and import-surface clarification**:

```text
Goal: make `ADFWI.utils` ownership readable without changing behavior.
Scope: `ADFWI/utils/__init__.py` docstring and docs only.
Validation: py_compile, import smoke, existing mute/validation tests.
Stop: no public function rename, no formula edit, no dataset-helper move.
```

## Validation Menu

Light checks:

```bash
conda run -n adfwi python -m py_compile ADFWI/utils/*.py
conda run -n adfwi python - <<'PY'
from ADFWI.utils import wavelet, numpy2tensor, load_marmousi_model, resample_marmousi_model
print("utils import ok")
PY
```

Focused tests:

```bash
conda run -n adfwi python -m unittest tests/test_mute_transform_comparison.py tests/test_marmousi2_validation_example.py
```

Real-case validation only when dataset/model/wavelet behavior changes:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py check --output-root examples/validation/marmousi2_acoustic_bv12/outputs/utils_validation_check --device npu:0 --shots 3 --checkpoint-segments 1
```

