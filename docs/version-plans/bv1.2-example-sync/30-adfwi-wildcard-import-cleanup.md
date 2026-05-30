# ADFWI Wildcard Import Cleanup

## Scope

This pass removed all remaining tracked `from ADFWI... import *` usage under
`examples/`.

After the `sys.path` cleanup, the remaining tracked wildcard imports were all
assessment-metric imports in plotting notebooks:

```text
examples/acoustic/01-model-test/08-Marmousi2-vp-rho/03_plot_inverted_res.ipynb
examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/03_plot_misfit.ipynb
examples/acoustic/03-optimizer-test/01-Marmousi2-Test/03_plot_misfit.ipynb
examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/03_plot_misfit.ipynb
examples/elastic/Iso-elastic-Marmousi2-shotTop-recTop/03_plot_inverted_res.ipynb
examples/multi-scale/Iso-elastic-Marmousi2-multifreq/03_plot_inverted_res.ipynb
```

## Optimization Path

- Replaced `from ADFWI.utils.assessment_metric import *` with the metric
  functions actually used by each notebook.
- Removed one unused assessment-metric import entirely.
- Updated one commented wildcard import so static scans do not report it as an
  actionable import residual.
- Did not change plotting logic, loaded files, numerical arrays, or notebook
  outputs.

Explicit imports used:

```text
MAPE, MSE, SNR, SSIM
MSE, SSIM
MSE
```

## Validation

```text
tracked ADFWI wildcard import hits: 0
notebook JSON/AST parse for touched notebooks: passed
```

## Next Direction

The tracked examples are now clean for two wrapper markers:

```text
sys.path.append/sys.path.insert: 0
from ADFWI... import *: 0
```

The next separate cleanup should be path residuals in tracked notebooks. Review
those per file because some hits are comments, saved outputs, or intentionally
repo-root execution paths.
