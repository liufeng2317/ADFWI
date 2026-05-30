# Ricker Misfit Notebook Import Sync

## Scope

This pass handled a legacy import wrapper found outside the previous
forward/inversion notebook scope:

```text
examples/acoustic/02-misfit-functions-test/00-Ricker-Test/01_misfit_wavelets-shift.ipynb
```

The earlier example-sync passes focused on files named `*forward*.ipynb` and
`*inversion*.ipynb`. This Ricker misfit notebook is a teaching/demo notebook,
so it was not included in those scans even though it still had the old wrapper.

## Optimization Path

The notebook was cleaned per file, based on the symbols actually used by its
cells:

- removed `sys.path.append("../../../../")`;
- removed broad wildcard imports from `ADFWI.propagator`, `ADFWI.model`,
  `ADFWI.view`, `ADFWI.utils`, `ADFWI.survey`, and `ADFWI.fwi`;
- added explicit imports for the used misfit classes;
- kept only `wavelet` and `numpy2tensor` from `ADFWI.utils`;
- moved repeated cell-local misfit imports into the setup cell;
- kept notebook outputs and numerical/plotting cells unchanged.

The notebook still contains CUDA-specific cells:

```python
numpy2tensor(...).to("cuda:0")
```

Those lines were intentionally left unchanged in this pass because changing
device placement would alter execution behavior and should be handled as a
separate compatibility update.

## Validation

Notebook checks:

```text
JSON parse: passed
Python AST parse for code cells: passed
legacy wrapper markers in this notebook: 0
import/setup cell execution in conda env adfwi: passed
```

The import/setup cell execution emitted only the known `torch_npu` environment
warnings from the local Ascend toolkit installation.

## Broader Scan Result

After this cleanup, a repository-wide notebook scan still found 89 notebooks
under `examples/` with one or more legacy wrapper markers such as
`sys.path.append(...)` or `from ADFWI... import *`.

The remaining files are mostly outside the previously targeted forward and
inversion workflow names:

- DR-FWI comparison and analysis notebooks under `cmp/`;
- acoustic misfit design/test notebooks;
- DIP result/comparison notebooks;
- gradient-checking notebooks after the forward step;
- waveform-checking notebooks;
- ignored backup notebooks.

This means the remaining work should be treated as a separate notebook-category
sync phase, not as more forward/inversion cleanup.

## Next Direction

Clean the sibling Ricker misfit notebooks first because they share the same
structure and are small:

```text
examples/acoustic/02-misfit-functions-test/00-Ricker-Test/02_misfit_wavelets-shift_and_Amplitude.ipynb
examples/acoustic/02-misfit-functions-test/00-Ricker-Test/03_misfit_wavelets-shift_and_f0.ipynb
examples/acoustic/02-misfit-functions-test/00-Ricker-Test/04_misfit_wavelets-shift_and_Gaussian_noise.ipynb
```

Then handle gradient-checking and waveform-checking notebooks by category,
still using per-file symbol analysis rather than a shared large import
template.
