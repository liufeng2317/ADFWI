# 127 - Marmousi2 Validation Inline Notebook Definitions

## Optimization Path

The validation notebooks should be readable like the original Marmousi2 example:
parameters, model setup, survey setup, and source-wavelet construction should be
visible directly in the notebook. A separate helper file made the validation
workflow harder to inspect and diverged from the original example style.

## Change

- Removed `examples/validation/marmousi2_acoustic_reduced/scripts/case_definition.py`.
- Inlined the lightweight case-definition helpers in both validation notebooks:
  - `load_npz`;
  - `cumulative_trapezoid`;
  - `build_source_wavelet`;
  - `build_survey`.
- Updated README to state that notebook case definitions live directly inside
  the notebooks.
- Updated tests to guard against reintroducing `case_definition.py` or importing
  setup definitions from `marmousi2_acoustic_backend_check`.

## Scientific Contract

- No FWI core code changed.
- No original Marmousi2 notebook changed.
- The validation runner command behavior is unchanged.
- This is a notebook readability and experiment-definition clarity change.

## Validation

Completed validation:

- `conda run -n adfwi python -m py_compile examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py tests/test_marmousi2_validation_example.py`: passed.
- `python -m json.tool examples/validation/marmousi2_acoustic_reduced/notebooks/01_forward_modeling.ipynb`: passed.
- `python -m json.tool examples/validation/marmousi2_acoustic_reduced/notebooks/02_inversion.ipynb`: passed.
- `conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py all --device cpu --dry-run`: passed.
- `conda run -n adfwi python -m unittest tests/test_marmousi2_validation_example.py`: passed.
- `git diff --check`: passed.

## Next Direction

Keep validation notebooks self-contained unless helper code becomes genuinely
large or shared by multiple validation cases. For this Marmousi2 example, inline
definitions are clearer.
