# 14 Analysis Notebook Path Sync

## Scope

This round used the analysis notebook below as the template:

```text
examples/DR-FWI/reparameterization-strategy/Analysis/Article-Figure10-FBC.ipynb
```

The same cleanup was applied locally to both notebooks in that ignored
`Analysis` folder:

- remove old `sys.path.append(...)` repository injection;
- replace `from ADFWI.utils.assessment_metric import *` with the metrics that
  are actually used;
- convert repository absolute paths to paths relative to the notebook folder.

Because `examples/DR-FWI/reparameterization-strategy/Analysis` is ignored by
`.gitignore` and the notebooks contain embedded outputs, these local notebook
edits are intentionally not force-added in this round.

## Tracked Path Cleanup

The same absolute-path rule was applied to tracked `Article_Figure` notebooks
that still pointed to old machine-specific repository roots:

```text
/home/.../ADFWI-github/examples/...
/liufeng1afs/.../ADFWI-github/examples/...
/ailab/.../ADFWI-github/examples/...
```

Those references were converted to relative paths such as:

```text
../reparameterization-strategy/Marmousi2-nowater/data
../multi-parameters/ISO_elastic/Marmousi2-nowater-vp_vs_rho/data
```

No figure logic, data selection, metric computation, or plotting parameters
were changed.

## Validation

- JSON validation passed for the two local `Analysis` notebooks.
- Residual scan found no old repository absolute paths, `sys.path.append`, or
  `assessment_metric import *` in `Analysis` or `Article_Figure` notebooks.

## Next Direction

Continue with the remaining DR-FWI analysis/report notebooks only when they are
tracked or when there is a clear decision to unignore and strip outputs. For
normal examples, continue using the established wrapper rules: minimal imports,
editable install, explicit backend setup, and paths relative to the file.
