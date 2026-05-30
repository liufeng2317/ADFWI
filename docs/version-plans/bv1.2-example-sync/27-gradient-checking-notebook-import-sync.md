# Gradient Checking Notebook Import Sync

## Scope

This pass cleaned legacy import wrappers in `examples/gradient_checking`.

Forward notebooks in this category were already synchronized. The remaining
legacy wrappers were in:

```text
examples/gradient_checking/Acoustic-Marmousi2/02_gradient_check_AD.ipynb
examples/gradient_checking/Acoustic-Marmousi2/02_gradient_check_FD.ipynb
examples/gradient_checking/Acoustic-Marmousi2/03_1_compare_gradient.ipynb
examples/gradient_checking/Elastic-Marmousi2/02_gradient_check_AD.ipynb
examples/gradient_checking/Elastic-Marmousi2/02_gradient_check_FD.ipynb
examples/gradient_checking/Elastic-Marmousi2/03_1_compare_gradient.ipynb
```

## Optimization Path

Each notebook was parsed independently and imports were reduced to the symbols
actually used by that notebook:

- removed notebook-local `sys.path.append(...)`;
- removed wildcard imports from `ADFWI.propagator`, `ADFWI.model`,
  `ADFWI.view`, `ADFWI.utils`, `ADFWI.survey`, and `ADFWI.fwi`;
- added explicit imports for the used model, propagator, survey, utility, and
  view symbols;
- moved repeated `tqdm` imports in finite-difference notebooks into the setup
  cell;
- left gradient-checking logic, perturbation loops, device strings, data paths,
  outputs, and numerical cells unchanged.

This pass intentionally did not change hard-coded CUDA/device choices or
absolute data paths that appear later in some finite-difference cells. Those
should be handled as a separate path/backend compatibility pass.

## Validation

```text
notebooks checked: 8
JSON parse: passed
Python AST parse for all code cells: passed
legacy wrapper markers under examples/gradient_checking: 0
ADFWI import availability in conda env adfwi: passed
```

The import availability check executed only import statements, not full AD/FD
gradient computations. Full execution is expensive and depends on existing
generated waveform and gradient data.

Known local `torch_npu` Ascend toolkit owner warnings were emitted during
import validation.

## Remaining Risk

Some gradient-checking finite-difference cells still contain hard-coded
absolute paths and CUDA-specific execution assumptions. They are outside the
scope of this import cleanup and should be reviewed only if these notebooks are
promoted to release-quality runnable examples.

## Next Direction

Apply the same category-level import cleanup to `examples/waveform_checking`.
Keep the scope narrow: first remove wrapper/import ambiguity, then decide
separately whether waveform checking paths and backend assumptions need a
behavioral compatibility pass.
