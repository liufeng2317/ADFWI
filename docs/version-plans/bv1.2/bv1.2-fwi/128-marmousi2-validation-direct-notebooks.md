# 128 - Marmousi2 Validation Direct Notebooks

## Optimization Path

The validation notebooks should behave like notebooks, not command-line
wrappers. The Python runner remains useful for shell automation, but Jupyter
workflows should define each step in-place and call ADFWI APIs directly.

## Change

- Removed CLI execution from the validation notebooks:
  - no `subprocess`;
  - no `run_validation.py`;
  - no dry-run command wrapper cells.
- Updated `01_forward_modeling.ipynb` to directly define and build:
  - backend configuration;
  - model;
  - survey/observation system;
  - source wavelet;
  - acoustic propagator;
  - optional forward modeling execution.
- Updated `02_inversion.ipynb` to directly define and build:
  - backend configuration;
  - true and initial models;
  - survey subset;
  - synthetic-true observed data;
  - optimizer, scheduler, misfit, gradient processor, and `AcousticFWI`;
  - optional 10-iteration and 100-iteration inversion runs.
- Updated tests to prevent CLI imports/calls from being reintroduced into the
  notebooks.

## Scientific Contract

- No FWI core code changed.
- No original Marmousi2 notebook changed.
- The command-line validation script remains available for shell automation.
- Jupyter validation now uses direct ADFWI APIs, matching the style of the
  original example notebooks.

## Validation

Completed validation:

- `python -m json.tool examples/validation/marmousi2_acoustic_reduced/notebooks/01_forward_modeling.ipynb`: passed.
- `python -m json.tool examples/validation/marmousi2_acoustic_reduced/notebooks/02_inversion.ipynb`: passed.
- `conda run -n adfwi python -m py_compile examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py tests/test_marmousi2_validation_example.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_marmousi2_validation_example.py`: passed.
- `git diff --check`: passed.

## Next Direction

Use the notebooks for interactive validation and the Python script only for
repeatable shell runs. Keep the two paths separate.
