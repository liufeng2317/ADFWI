# Ricker Misfit Sibling Import Sync

## Scope

This pass completed the same import cleanup for the remaining Ricker-Test
notebooks with legacy ADFWI wrappers:

```text
examples/acoustic/02-misfit-functions-test/00-Ricker-Test/00_test_new_misfit.ipynb
examples/acoustic/02-misfit-functions-test/00-Ricker-Test/02_misfit_wavelets-shift_and_Amplitude.ipynb
examples/acoustic/02-misfit-functions-test/00-Ricker-Test/03_misfit_wavelets-shift_and_f0.ipynb
examples/acoustic/02-misfit-functions-test/00-Ricker-Test/04_misfit_wavelets-shift_and_Gaussian_noise.ipynb
```

## Optimization Path

For `02`, `03`, and `04`:

- removed notebook-local `sys.path.append("../../../../")`;
- removed broad wildcard imports from unused ADFWI namespaces;
- moved repeated cell-local misfit imports into the setup cell;
- kept only the misfit classes used by the notebook;
- kept `wavelet` and `numpy2tensor` from `ADFWI.utils`;
- moved `matplotlib.gridspec` to the setup cell;
- preserved outputs and all numerical/plotting cells.

For `00_test_new_misfit.ipynb`:

- removed unused `ADFWI.view` wildcard import;
- replaced `ADFWI.utils` wildcard import with explicit `numpy2tensor`;
- preserved the notebook-local `wavelet(...)` helper because this notebook
  defines and uses its own wavelet implementation.

CUDA-specific cells such as `.to("cuda:0")` were not changed in this pass.
That should be handled separately if these notebooks need backend-independent
execution.

## Validation

```text
JSON parse: passed
Python AST parse for all code cells: passed
legacy wrapper markers in the four notebooks: 0
setup/import cell execution in conda env adfwi: passed
```

The setup/import execution produced only the known local `torch_npu` Ascend
toolkit owner warnings.

## Next Direction

Do not continue broad notebook cleanup one file at a time. The next bounded
category should be either:

- gradient-checking notebooks after the forward step; or
- waveform-checking notebooks.

Use the same per-file symbol analysis and avoid introducing a shared large
import template.
