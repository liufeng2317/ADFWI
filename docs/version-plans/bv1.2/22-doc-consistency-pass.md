# Documentation Consistency Pass

## Goal

Keep the bv1.2 planning documents aligned with the implementation after the
recent FWI data-path cleanup. Several earlier documents were intentionally
written before later follow-up steps, so they still described old behavior for
`calculate_loss(...)`, receiver selection, and waveform normalization.

## Current Step

Updated the earlier planning records to match the current code state:

- `03-data-transform-pipeline.md` now notes that `TraceNormalize()` and the FWI
  fallback normalization share `ADFWI.fwi.normalization.normalize_waveform(...)`.
- `03-data-transform-pipeline.md` now states that direct
  `calculate_loss(..., apply_transforms=True)` calls route through
  `_prepare_loss_pair()` when `shot_index` is provided.
- `08-fwi-data-contract.md` now lists the current shared helpers, including
  `build_fwi_data_transform_pipeline(...)`, `build_fwi_transform_context(...)`,
  `prepare_fwi_loss_pair(...)`, `normalize_waveform(...)`, and
  `evaluate_misfit_loss(...)`.
- `08-fwi-data-contract.md` and `17-fwi-loss-pair-builder.md` no longer claim
  that direct `calculate_loss()` calls skip receiver selection.

## Validation

This step changes documentation only. No Python code, tests, notebooks, examples,
or generated outputs were modified. The numerical baseline referenced by these
docs remains the most recent validated state from
`21-trace-normalize-shared-formula.md`:

- transform/FWI related tests passed with 68 tests;
- acoustic CPU/NPU trace-missing smoke drift was zero for loss, gradient norm,
  and update norm;
- elastic CPU/NPU trace-missing smoke stayed within tolerance with loss relative
  drift `4.880620563312549e-06` and gradient relative drift
  `1.2590211728446073e-06`.

## Next Steps

1. Continue with code-level FWI engine cleanup after the documentation catches
   up with the current data path.
2. Prefer one current summary document for future architectural decisions, while
   keeping older numbered records as historical change logs.
3. Keep documenting whether each step is code-changing or documentation-only, so
   numerical validation expectations remain clear.
