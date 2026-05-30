# 05 - View Stage Summary

## Goal

Archive the completed `ADFWI/view` optimization stage and record what changed,
how it was validated, and why the stage should stop.

## Optimization Path

1. Designed the view boundary.
   `ADFWI/view` is defined as a plotting/output helper layer. It does not own
   backend selection, numerical transforms, survey mutation, model constraints,
   or FWI state.

2. Clarified backend policy.
   Library code must not call `matplotlib.use(...)`. Tests and validation
   scripts may select a backend locally, before importing plotting helpers.

3. Added plotting contract tests.
   `tests/test_view_contracts.py` covers the public plotting helpers with tiny
   synthetic inputs, saved figures, figure close behavior, and explicit public
   exports.

4. Fixed confirmed plotting bugs.
   - `plot_eps_delta_gamma()` now draws `delta` on the intended panel in the
     no-spacing branch.
   - `plot_waveform_wiggle()` now creates its own figure, preventing real-case
     figure reuse from mixing boundary-condition plots into wiggle output.

5. Made the public namespace explicit.
   `ADFWI/view/__init__.py` now exports only the intended plotting helper API
   through explicit imports and `__all__`.

6. Checked real validation figures.
   Marmousi2 forward validation figures were inspected with a contact sheet.
   The wiggle figure issue was confirmed and fixed; the regenerated figure is
   independent and no longer mixed with the previous boundary-condition figure.

## Changed Files

- `ADFWI/view/__init__.py`
- `ADFWI/view/velocity_model.py`
- `ADFWI/view/waveform.py`
- `tests/test_view_contracts.py`
- `docs/version-plans/bv1.2-view/`

## Validation Summary

Commands used across the stage:

```bash
conda run -n adfwi python -m unittest tests/test_view_contracts.py
conda run -n adfwi python -m py_compile ADFWI/view/*.py tests/test_view_contracts.py
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/forward_modeling.py forward \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/view_plot_audit \
  --device npu:0 --dtype float32 --shots 3 --checkpoint-segments 1
git diff --check
```

Final test state:

- View contract tests passed: 7 tests.
- `py_compile` passed for `ADFWI/view/*.py`.
- Marmousi2 forward validation passed.
- Forward numerical summary remained stable:
  - `p` shape: `[3, 3000, 200]`
  - `p` norm: `1.3137102127075195`
  - finite: `true`

## Public Plotting API

The explicit `ADFWI.view.__all__` surface is:

- `animate_inversion_process`
- `plot_bcx_bcz`
- `plot_damp`
- `plot_eps_delta_gamma`
- `plot_initial_and_inverted`
- `plot_lam_mu`
- `plot_misfit`
- `plot_model`
- `plot_survey`
- `plot_vp_rho`
- `plot_vp_vs_rho`
- `plot_waveform2D`
- `plot_waveform_trace`
- `plot_waveform_wiggle`
- `plot_wavelet`

## Remaining Risk

- `observed_system.py` remains an archived helper file with `add_colorbar()`
  and commented legacy plotting code. It is not part of `ADFWI.view.__all__`.
- `boundary_condition.plot_bc()` remains an unexported empty placeholder.
- Some files still contain legacy headers, unused imports, and broad warning
  filters. These are not blocking behavior and should not trigger another
  cleanup round by themselves.
- `plot_waveform2D(norm=True)` still normalizes NumPy input in place through
  `norm_traces()`. This is a behavior contract risk and should only be changed
  with a specific caller audit.
- Wiggle plots with all 200 receivers are dense. This is a plotting parameter
  issue, not a bug in the helper itself.

## Stop Decision

Stop the `ADFWI/view` stage here.

The module now has:

- a documented ownership boundary;
- explicit public exports;
- focused plotting contract tests;
- validation-case image review;
- fixes for the confirmed plotting bugs found during this stage.

Further work should only happen for a new, concrete user-facing plotting bug or
a planned compatibility/API migration.

## Commit Trail

- `3672bbc` - Plan view module optimization
- `20e198f` - Clarify view backend policy
- `a9291a4` - Add view plotting contract tests
- `8074213` - Clarify view public exports
- `7ca1e9a` - Document view closeout audit
- `be2122d` - Fix waveform wiggle figure isolation
