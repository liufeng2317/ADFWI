# Forward Notebook Import Dedup

## Scope

This pass used the cleaned Marmousi2 forward notebook import style as the
reference and scanned all forward notebooks under `examples/`.

The cleanup only touched import cells. It did not change forward parameters,
models, survey geometry, propagator calls, save paths, notebook outputs, or
numerical expressions.

## Optimization Path

1. Scan `*forward*.ipynb` for broad template import markers such as
   `AbstractModel`, `GradProcessor`, `build_anomaly_background_model`, and
   `animate_inversion_process`.
2. Parse each matched notebook with AST and collect the symbols actually used
   outside import statements.
3. Filter `from ADFWI... import ...` lines per notebook, keeping only symbols
   referenced by that notebook.
4. Run a second pass to remove duplicated ADFWI symbols when a notebook already
   had a smaller, explicit import block before the template block.

## Result

- Cleaned 99 forward notebooks.
- Removed 3577 unused ADFWI import aliases from broad template blocks.
- Removed 199 duplicated ADFWI import aliases in the follow-up dedup pass.
- Remaining broad-template marker hits in changed notebooks: 0.
- Remaining duplicated ADFWI import symbols in changed notebooks: 0.

## Validation

```text
JSON + AST parse for changed notebooks
checked=99
failed=0
```

Representative import-cell smoke test:

```text
conda run -n adfwi python /tmp/adfwi_forward_import_smoke.py
ok examples/acoustic/01-model-test/02-FootHill/01_forward.ipynb
ok examples/elastic/Iso-elastic-Anomaly/01_forward.ipynb
ok examples/DR-FWI/multi-parameters/ISO_acoustic/Marmousi2/00_forward.ipynb
ok examples/validation/marmousi2_acoustic_reduced/notebooks/01_forward_modeling.ipynb
```

The full forward simulations were not rerun because this pass only removed
unused imports and duplicate import aliases.

## Remaining Risk

Some notebooks may still import modules such as `matplotlib.pyplot` even when
they are not heavily used. This pass intentionally focused on ADFWI template
imports, which were the confusing part of the public API examples.

## Next Direction

Use the same per-file symbol analysis for inversion notebooks only where the
template block is clearly present. Avoid replacing notebook-specific imports
with a single global import template.
