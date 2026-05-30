# 123 - Marmousi2 Validation Example

## Optimization Path

Move from framework optimization to instance validation. The original Marmousi2
example directory is kept unchanged. A separate validation folder is added under
`examples/validation/` so Python and Jupyter workflows can run the same staged
checks without modifying the historical notebooks.

## Change

- Added `examples/validation/marmousi2_acoustic_reduced/README.md`.
- Added `examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py` with
  staged validation commands:
  - `check`;
  - `forward`;
  - `inversion10`;
  - `inversion100`;
  - `all` for `check + forward + inversion10`.
- Added separate Jupyter validation notebooks:
  - `examples/validation/marmousi2_acoustic_reduced/notebooks/01_forward_modeling.ipynb`;
  - `examples/validation/marmousi2_acoustic_reduced/notebooks/02_inversion.ipynb`.
- Added `tests/test_marmousi2_validation_example.py` to verify dry-run command
  construction without running heavy forward/inversion jobs.

## Scientific Contract

- The original Marmousi2 example notebooks are untouched.
- The validation wrapper reuses existing script-style Marmousi2 entry points
  instead of duplicating case construction logic.
- Dry-run and unit tests do not run heavy numerical workloads.
- Actual numerical validation remains explicit through the `check`, `forward`,
  `inversion10`, and `inversion100` stages.
- Forward modeling and inversion are separate notebook workflows even though
  they share the same script runner.

## Validation

Completed validation:

- `conda run -n adfwi python -m py_compile examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py tests/test_marmousi2_validation_example.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_marmousi2_validation_example.py`: passed.
- `conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py all --device cpu --dry-run`: passed.
- `git diff --check`: passed.

## Next Direction

Run staged validation manually on NPU:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py check --overwrite
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py forward --overwrite
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py inversion10 --overwrite
```

Only run `inversion100` after inspecting the 10-iteration loss and model-update
outputs.
