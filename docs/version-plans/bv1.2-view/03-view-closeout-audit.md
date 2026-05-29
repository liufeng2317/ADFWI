# 03 - View Closeout Audit

## Goal

Close the current `ADFWI/view` cleanup stage by recording validated behavior,
remaining risk, and the stop decision.

## Scope

- `ADFWI/view/`
- `tests/test_view_contracts.py`
- `docs/version-plans/bv1.2-view/`

No implementation code is changed in this closeout round.

## Current State

`ADFWI/view` is now documented and tested as a plotting/output helper layer.

Completed in this stage:

- wrote a view optimization outline and ownership map;
- clarified that `ADFWI/view` library code must not choose a Matplotlib backend;
- added focused plotting smoke tests for public plot helpers;
- fixed the confirmed `plot_eps_delta_gamma()` panel-axis bug;
- replaced the wildcard package import with explicit plotting helper exports;
- locked the public plotting surface with `__all__`.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_view_contracts.py
conda run -n adfwi python -m py_compile ADFWI/view/*.py tests/test_view_contracts.py
git diff --check
```

## Result

- View contract tests passed: 6 tests.
- `py_compile` passed for `ADFWI/view/*.py` and
  `tests/test_view_contracts.py`.
- No new implementation bug was found that justifies code changes in this
  closeout round.
- Only existing local NPU/Ascend and Matplotlib/PyParsing warnings were printed.

## Remaining Risk

1. Several files still contain legacy headers, unused imports, and broad warning
   filters. These are readability issues. They should not trigger another
   cleanup round unless they block tests, packaging, or user workflows.

2. `boundary_condition.plot_bc()` is an empty placeholder. It is not exported
   by `ADFWI.view.__all__`, and no direct caller was found in the view audit.
   Removing it would be a compatibility choice, not a required bug fix.

3. `observed_system.py` contains `add_colorbar()` plus commented legacy plotting
   code. It is not part of the current public package exports. It can remain
   archived unless a caller needs it.

4. `waveform.norm_traces()` mutates its NumPy input when used by
   `plot_waveform2D(norm=True)`. This is a plotting-helper behavior risk and
   should be changed only with a before/after caller audit.

5. Some plotting helpers still use fixed style defaults and repeated layout
   code. This is acceptable for now. Visual style cleanup should not continue
   without a concrete user-facing issue.

## Stop Decision

Stop the `ADFWI/view` optimization stage here.

The module now has a clearer public boundary and smoke coverage for public plot
helpers. Further work would mainly be aesthetic or compatibility-sensitive.

## Recommended Next Step

Move to the next framework module rather than continuing view cleanup. If view
work resumes later, it should be limited to a confirmed user-facing plotting bug
or a planned compatibility break.
