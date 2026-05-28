# 126 - Marmousi2 Validation Local Case Definition

## Optimization Path

The validation notebooks should not import case definitions from another case or
from a backend-check script. Forward modeling and inversion notebooks need their
own local case-definition helpers so the validation example is self-contained.

## Change

- Added `examples/validation/marmousi2_acoustic_bv12/scripts/case_definition.py`
  with local helpers for:
  - loading required `.npz` files;
  - building the Marmousi2 survey;
  - constructing the integrated source wavelet.
- Updated validation notebooks to import from local `case_definition.py`.
- Updated README to document the local case-definition helper.
- Added a regression test that prevents notebooks from importing
  `marmousi2_acoustic_backend_check`.

## Scientific Contract

- No FWI core code changed.
- No original Marmousi2 notebooks changed.
- The local helper mirrors the same case metadata interpretation used by the
  validation runner, but the notebooks no longer import another case/check
  script for definitions.

## Validation

Completed validation:

- `conda run -n adfwi python -m py_compile examples/validation/marmousi2_acoustic_bv12/scripts/case_definition.py examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py tests/test_marmousi2_validation_example.py`: passed.
- `python -m json.tool examples/validation/marmousi2_acoustic_bv12/notebooks/01_forward_modeling.ipynb`: passed.
- `python -m json.tool examples/validation/marmousi2_acoustic_bv12/notebooks/02_inversion.ipynb`: passed.
- `conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py all --device cpu --dry-run`: passed.
- `conda run -n adfwi python -m unittest tests/test_marmousi2_validation_example.py`: passed.
- `git diff --check`: passed.

## Next Direction

Keep validation examples self-contained. If more notebooks are added, each
validation example should own its case-definition helpers instead of importing
setup functions from another example.
