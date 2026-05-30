# 124 - Marmousi2 Validation Notebook Split

## Optimization Path

Clarify the example validation workflow by separating forward modeling and
inversion at the notebook level. The Python runner already exposes separate
stages; this pass aligns the Jupyter entry points with that experimental
boundary.

## Change

- Replaced the combined validation notebook with:
  - `notebooks/01_forward_modeling.ipynb`;
  - `notebooks/02_inversion.ipynb`.
- Updated the validation README to describe the two notebook workflows.
- Updated the dry-run test to assert that forward and inversion notebooks remain
  separate.

## Scientific Contract

- No FWI code changed.
- No validation command behavior changed.
- The original Marmousi2 example notebooks remain untouched.
- The new notebooks only reorganize how users launch `check`/`forward` versus
  `inversion10`/`inversion100`.

## Validation

Completed validation:

- `conda run -n adfwi python -m py_compile examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py tests/test_marmousi2_validation_example.py`: passed.
- `python -m json.tool examples/validation/marmousi2_acoustic_reduced/notebooks/01_forward_modeling.ipynb`: passed.
- `python -m json.tool examples/validation/marmousi2_acoustic_reduced/notebooks/02_inversion.ipynb`: passed.
- `conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py all --device cpu --dry-run`: passed.
- `conda run -n adfwi python -m unittest tests/test_marmousi2_validation_example.py`: passed.
- `git diff --check`: passed.

## Next Direction

Run the forward-modeling notebook first, inspect the true-model forward output,
then run the inversion notebook for 10 iterations before considering the
100-iteration stage.
