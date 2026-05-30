# Inversion Notebook Import Audit

## Scope

This pass applied the same per-notebook import analysis used for forward
notebooks to inversion notebooks.

Target pattern:

```text
examples/**/*inversion*.ipynb
```

The audit only inspected imports. It did not change inversion parameters,
optimizers, loss functions, regularization, DIP models, paths, outputs, or
notebook execution results.

## Method

1. Parse each tracked inversion notebook as JSON.
2. Parse every code cell with AST.
3. Collect symbols actually referenced outside import statements.
4. Compare referenced symbols against `from ADFWI... import ...` aliases.
5. Report removable aliases only when an imported ADFWI symbol is not used by
   that notebook.

## Result

Tracked inversion notebooks:

```text
tracked_inversion_notebooks=25
parse_skipped=0
tracked_star_imports=0
tracked_multiline_adfwi_imports=3
tracked_removable_adfwi_aliases=0
```

No tracked inversion notebook needed import changes. Markers such as
`GradProcessor`, `Misfit_global_correlation`, `regularization_TV_2order`, and
`DIP_AcousticFWI` remain because they are real dependencies in inversion
examples, not unused template imports.

## Ignored Legacy Backup

One ignored backup notebook still contains wildcard imports:

```text
examples/elastic/Iso-elastic-Anomaly/backup/02_inversion.ipynb
```

It is ignored by `.gitignore` through `examples/elastic/` and is not part of
the tracked example set. It was intentionally not promoted or force-added in
this pass.

## Validation

```text
tracked_inversion_notebooks=25
parse_skipped=0
tracked_star_imports=0
tracked_removable_adfwi_aliases=0
```

## Next Direction

Do not run broad inversion-notebook import cleanup again unless new notebooks
are added or a concrete notebook shows a duplicated template block. The next
useful step is a small smoke run of representative synchronized examples,
rather than more static import cleanup.
