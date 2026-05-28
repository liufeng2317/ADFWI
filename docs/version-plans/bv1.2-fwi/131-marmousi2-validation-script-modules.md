# 131 - Marmousi2 Validation Script Modules

## Optimization Path

The validation scripts should mirror the notebook split: forward modeling and
inversion are separate workflow modules. A combined runner can remain as a
dispatcher, but it should not own both workflows.

## Change

- Added `scripts/forward_modeling.py` for true-model forward simulation and
  observed-data generation.
- Added `scripts/inversion.py` for initial-model FWI using the forward-generated
  `obs_data.npz`.
- Reduced `scripts/run_validation.py` to orchestration and dry-run planning.
- Updated tests to require separate forward and inversion script files.
- Updated validation README with direct module commands.

## Scientific Contract

- No FWI core code changed.
- No original Marmousi2 example notebook changed.
- Script defaults remain aligned with the minimal notebooks:
  - `npu:0`;
  - 3 shots;
  - `checkpoint_segments=1`;
  - 10/100 inversion iterations;
  - shared `outputs/minimal_notebook` root.

## Validation

Completed validation:

- `python -m py_compile examples/validation/marmousi2_acoustic_bv12/scripts/forward_modeling.py examples/validation/marmousi2_acoustic_bv12/scripts/inversion.py examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py tests/test_marmousi2_validation_example.py`: passed.
- `python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py all --device cpu --dry-run`: passed.
- `python -m unittest tests/test_marmousi2_validation_example.py`: passed.
- `conda run -n adfwi python -m py_compile ...`: passed.
- `conda run -n adfwi python -m unittest tests/test_marmousi2_validation_example.py`: passed.
- `git diff --check`: passed.

## Next Direction

Use `forward_modeling.py` and `inversion.py` when validating one workflow at a
time. Use `run_validation.py` only when a staged dry-run or combined execution
is useful.
