# 00 - View Optimization Outline

## Goal

Make `ADFWI/view` easier to understand and safer to maintain as a plotting
layer, without changing scientific outputs or moving numerical behavior into
the plotting code.

## Scope

First-stage design scope:

- `ADFWI/view/__init__.py`
- `ADFWI/view/velocity_model.py`
- `ADFWI/view/waveform.py`
- `ADFWI/view/survey.py`
- `ADFWI/view/boundary_condition.py`
- `ADFWI/view/inverted_loss_model.py`
- `ADFWI/view/observed_system.py`
- direct callers in validation scripts and examples

No code implementation is changed in this planning round.

## Current Findings

1. The module is mostly a plotting compatibility layer.
   Functions accept NumPy arrays or tensors, convert via `gpu2cpu`, then draw
   Matplotlib figures.

2. Responsibilities are mostly reasonable by file:
   - `velocity_model.py`: model-property images;
   - `waveform.py`: trace, section, and wiggle plots;
   - `survey.py`: acquisition geometry and source wavelet plots;
   - `boundary_condition.py`: boundary/damping images;
   - `inverted_loss_model.py`: inversion history plots and animation;
   - `observed_system.py`: colorbar helper and archived plotting code.

3. The public namespace is broad and partly implicit.
   `__init__.py` imports concrete plotting functions and uses a wildcard import
   from `inverted_loss_model.py`. This keeps old examples working, but it makes
   the exported surface less explicit.

4. Some files contain legacy headers, unused imports, and no module-level
   boundary explanation.
   This is a readability issue, not a numerical issue.

5. There are a few clear bug candidates in plotting-only paths:
   - `velocity_model.plot_eps_delta_gamma()` uses `ax[0].imshow(delta, ...)`
     in the `dx/dz <= 0` branch where `ax[1]` is intended.
   - `boundary_condition.plot_bc()` is an empty placeholder and is not exported.
   - `boundary_condition.plot_bcx_bcz()` stores `bcz` in a local named `bxz`,
     which is confusing but behavior-preserving.
   - `waveform.norm_traces()` mutates the input array in place when
     `plot_waveform2D(norm=True)` is used with a NumPy array.
   - `waveform.wiggle_input_check()` has error messages that say `tt` when
     checking `xx`.

6. Plot helper tests are currently indirect.
   Validation examples exercise several functions by saving figures, but there
   is no focused `ADFWI/view` smoke test that ensures each public plot helper
   can run with the non-interactive `Agg` backend.

## Optimization Strategy

The view module should be optimized by contract and smoke validation, not by
large refactors.

### Phase 1 - Design and Boundary

Status: this document.

Record the ownership boundary, current structure, likely risks, and validation
strategy. Do not edit implementation code.

### Phase 2 - Plotting Contract Smoke Tests

Add focused tests using Matplotlib `Agg`:

- tiny synthetic arrays only;
- call each public plotting helper with `show=False`;
- save to a temporary path where useful;
- assert files are created and no figures are left open;
- avoid visual-style assertions.

This phase should not change plotting behavior unless a clear bug is exposed.

### Phase 3 - Clear Plotting Bugs Only

Fix only behavior-obvious plotting bugs:

- `plot_eps_delta_gamma()` wrong axis in the no-spacing branch;
- misleading local names or unreachable placeholder exports if tests expose
  them.

Validation should be the focused view smoke tests plus `py_compile`.

### Phase 4 - Public Namespace Clarity

Make `ADFWI/view/__init__.py` explicit:

- add a package docstring;
- replace wildcard import from `inverted_loss_model.py` with explicit imports;
- preserve all existing public names.

Validation should include import smoke and any direct validation/example path
that imports from `ADFWI.view`.

### Phase 5 - Optional Closeout

Write a short closeout summary and remaining-risk list. Stop unless a real
plotting bug or caller break appears.

## Validation Plan

Use light validation first:

```bash
conda run -n adfwi python -m py_compile ADFWI/view/*.py
conda run -n adfwi python -m unittest tests/test_view_contracts.py
git diff --check
```

If a changed plotting path is used by the validation case:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/forward_modeling.py forward \
  --output-root examples/validation/marmousi2_acoustic_bv12/outputs/view_validation \
  --device npu:0 --dtype float32 --shots 3 --checkpoint-segments 1
```

Full inversion is not required for pure view changes unless inversion-specific
figure helpers are modified.

## Stop Decision

Stop the view stage when:

- public plotting helpers have smoke coverage;
- obvious plotting-only bugs are fixed;
- namespace exports are explicit;
- validation examples still generate expected figures.

Do not keep improving visual style, color palettes, or layout aesthetics unless
there is a user-facing bug.
