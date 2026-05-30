# Tracked Notebook Path Cleanup

## Scope

This pass cleaned executable path residuals in tracked notebooks under
`examples/`.

The cleanup targeted code-cell paths only, then removed one stale notebook
error output that still contained an absolute local machine path.

## Optimization Path

Path types handled:

- dataset paths written as repo-root strings such as
  `./examples/datasets/...`;
- case-local output paths written as `./examples/<case>/data`;
- absolute local paths under `/ailab/user/liufeng1/...`;
- validation notebook absolute fallback path under `/liufeng1afs/...`.

Edits were made relative to each notebook's own directory:

- dataset paths now point to `../.../datasets/<dataset>` from the notebook;
- case output paths now use `./data` or the corresponding local folder;
- gradient-checking generated data paths now use `./data/...`;
- validation notebook fallback now raises a clear runtime error instead of
  silently switching to a machine-specific path.

Notebook outputs were preserved except for one stale Jupyter kernel-start error
output containing an absolute local path.

## Validation

```text
changed notebooks: 27
code-cell path residual hits: 0
full-text path residual hits: 0
changed notebook JSON/AST parse: passed
```

The AST parse emitted existing Python `SyntaxWarning` messages in unrelated
DR-FWI article figure notebooks because of invalid escape sequences in plot
labels. They are warnings, not parse failures, and were not modified in this
path cleanup pass.

## Remaining Risk

This pass does not prove every referenced data file exists. It only removes
machine-specific and repo-root-style path assumptions from tracked notebooks.
Heavy notebooks should still be validated by category before being presented as
release examples.

## Next Direction

The tracked example wrappers are now clean for:

```text
sys.path.append/sys.path.insert: 0
from ADFWI... import *: 0
matplotlib.use("agg"/"Agg"): 0
tracked notebook executable path residuals: 0
```

Next, run a final global residual scan and one representative smoke validation
before closing the example-sync branch.
