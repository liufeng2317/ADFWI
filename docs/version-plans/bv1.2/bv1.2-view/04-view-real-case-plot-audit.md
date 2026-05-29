# 04 - View Real-Case Plot Audit

## Goal

Inspect the actual Marmousi2 validation figures and fix only confirmed plotting
issues that affect result interpretation.

## Scope

- `ADFWI/view/waveform.py`
- `tests/test_view_contracts.py`
- Marmousi2 acoustic validation forward figures
- `docs/version-plans/bv1.2-view/`

No numerical modeling, inversion, backend policy, or Matplotlib global backend
behavior is changed.

## Finding

The actual validation figure
`waveform/obs_wiggle_shot_0.png` contained the previous boundary-condition
image in the upper part of the saved figure, with the wiggle plot compressed at
the bottom.

The cause was `plot_waveform_wiggle()` drawing directly on `plt.gca()` without
creating its own figure. If a previous figure was still current, the wiggle plot
could reuse that axes/figure.

## What Changed

- `plot_waveform_wiggle()` now creates a fresh figure before drawing.
- Added a regression test that opens an existing figure first, calls
  `plot_waveform_wiggle()`, and verifies the existing figure remains open.
- Added a `figsize` argument to `plot_waveform_wiggle()` with a default value.
  This is additive and does not force a Matplotlib backend.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_view_contracts.py
conda run -n adfwi python -m py_compile ADFWI/view/*.py tests/test_view_contracts.py
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/forward_modeling.py forward \
  --output-root examples/validation/marmousi2_acoustic_bv12/outputs/view_plot_audit \
  --device npu:0 --dtype float32 --shots 3 --checkpoint-segments 1
git diff --check
```

## Result

- View contract tests passed: 7 tests.
- `py_compile` passed.
- Marmousi2 validation forward completed successfully.
- Forward numerical summary remained stable:
  - `p` shape: `[3, 3000, 200]`
  - `p` norm: `1.3137102127075195`
  - finite: `true`
- The regenerated `obs_wiggle_shot_0.png` is now an independent wiggle figure
  rather than a reused boundary-condition figure.

## Remaining Plot Notes

- The wiggle plot is visually dense for 200 traces. This is an expected
  parameter/display-density issue, not a plotting bug. Users can thin traces or
  adjust plotting parameters in examples if needed.
- The other validation figures inspected in this round were readable enough for
  validation purposes and do not justify further style-only changes.

## Next Bounded Task

Stop view optimization again after this real-case fix. Move to another module
unless a new user-facing plotting bug appears.
