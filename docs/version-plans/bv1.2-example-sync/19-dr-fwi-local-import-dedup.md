# DR-FWI Local Import Dedup

## Scope

This pass only cleaned duplicated local imports in DR-FWI Python scripts.
It did not change model setup, inversion parameters, file paths, backend
selection, plotting calls, or numerical behavior.

## Optimization Path

1. Scan `examples/DR-FWI/**/*.py` with AST.
2. Build each file's top-level ADFWI import symbol set.
3. Remove a local `from ADFWI... import ...` only when every imported symbol
   was already available from the file's top-level imports.
4. Treat `ADFWI.view` and `ADFWI.view.inverted_loss_model` as equivalent for
   the already-exported plotting helpers.

## Result

- Removed 2750 duplicated local ADFWI import statements.
- Git-tracked diff touches 563 DR-FWI Python scripts.
- Remaining duplicated local ADFWI imports detected by the same rule: 0.

## Validation

```text
conda run -n adfwi python /tmp/adfwi_py_compile_changed.py
checked=563
failed=0
```

This is a syntax/contract cleanup only. Full inversion runs are not required
because no runtime configuration or numerical expression was modified.

## Remaining Risk

- Some scripts may still contain non-duplicated local imports that are used as
  local dependencies. They were intentionally preserved.
- Further import cleanup should be file-aware and should not replace analysis
  with broad import blocks.

## Next Direction

Continue with example sync by auditing remaining DR-FWI `.py` scripts for
absolute output paths and stale wrapper patterns. Avoid another broad pass
unless the scan identifies a repeated mechanical issue.
