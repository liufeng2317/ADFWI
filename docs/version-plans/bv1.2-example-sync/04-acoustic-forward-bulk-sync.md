# Acoustic Forward Bulk Sync

## Boundary

```text
Goal:
Apply the bv1.2 forward import/backend pattern to clearly matching acoustic
forward examples.

Scope:
Acoustic-only forward notebooks under `examples/` that match the standard
`AcousticModel` + `AcousticPropagator` pattern.

Validation:
Notebook JSON parse, code-cell compile, and residual old-pattern scan.

Stop:
Skip elastic, DIP, multi-scale/FWI-coupled, validation-reference, real-case, and
issue-reproduction notebooks. Do not run heavy forward simulations in this
round.
```

## Applied Pattern

For matching acoustic forward notebooks:

- removed notebook-local `sys.path.append(...)`;
- added `import ADFWI`;
- added `backend = ADFWI.set_backend(device, dtype=dtype)` after the dtype
  declaration;
- updated device comments to `CPU/GPU/NPU`;
- removed `device=device` and `dtype=dtype` from `AcousticModel(...)`;
- removed `device=device` from `AcousticPropagator(model, survey, ...)`.

The wildcard imports were not rewritten in this bulk pass. Replacing them with
fully explicit imports requires per-case symbol analysis and should be handled
separately.

## Result

The scan found 106 forward notebooks:

| Category | Count | Action |
| --- | ---: | --- |
| Acoustic-only standard forward pattern | 51 | Updated locally |
| Special / skipped | 55 | Not modified by this pass |

Of the 51 updated acoustic-only notebooks:

- 26 were tracked notebooks updated by this branch pass;
- 25 were ignored or untracked local examples, updated on disk but not added to
  git in this commit;
- the original Marmousi2 forward notebook was already in a manual dirty state
  and was not part of the automated pass.

## Skipped Reasons

Skipped notebooks included:

- elastic or VTI/ISO elastic forward cases;
- DIP-related examples;
- multi-scale / FWI-coupled examples;
- validation reference notebooks;
- real-case study notebooks;
- GitHub issue reproduction notebooks;
- the manually edited Marmousi2 forward notebook.

These require smaller follow-up passes because their constructor contracts,
runtime intent, or result expectations differ from the acoustic-only forward
pattern.

## Validation Result

```text
bad_json 0
bad_compile 0
candidates_with_backend 53
residual_core_old_patterns 1
```

The one residual old pattern is:

```text
examples/validation/marmousi2_acoustic_reduced/notebooks/01_forward_modeling.ipynb
```

That notebook was intentionally skipped as a validation reference.

## Next Direction

Next bounded task:

1. decide whether ignored/untracked local example updates should be force-added
   to git or treated as local-only cleanup;
2. close the manually edited Marmousi2 forward notebook and generated figures;
3. handle elastic forward notebooks in a separate pass after validating
   `ElasticModel`/`ElasticPropagator` backend inheritance.
