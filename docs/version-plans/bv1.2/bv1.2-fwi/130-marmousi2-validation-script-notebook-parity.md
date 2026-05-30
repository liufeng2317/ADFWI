# 130 - Marmousi2 Validation Script Notebook Parity

## Optimization Path

The validation script should match the minimal validation notebooks instead of
wrapping older script-style smoke tests. This keeps the automated path and the
interactive Jupyter path comparable.

## Change

- Replaced the wrapper-based `run_validation.py` implementation with a direct
  ADFWI API workflow.
- Aligned script defaults with the notebooks:
  - output root: `outputs/minimal_notebook`;
  - device: `npu:0`;
  - shots: `3`;
  - inversion iterations: `10` or `100`;
  - checkpoint segments: `1`;
  - Adam learning rate: `10`;
  - `StepLR(step_size=200, gamma=0.75)`;
  - waveform L2 misfit and gradient top mute.
- Made `forward` generate `waveform/obs_data.npz` from the true Marmousi2 model.
- Made `inversion10` and `inversion100` read the forward-generated observed
  data, matching the notebook workflow.
- Kept `--dry-run` lightweight so it reports the plan without importing
  torch/ADFWI.

## Scientific Contract

- No FWI core code changed.
- No original Marmousi2 example notebook changed.
- Script and notebook validation now share the same output root and observed
  data handoff.

## Validation

Completed validation:

- `python -m py_compile examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py tests/test_marmousi2_validation_example.py`: passed.
- `python -m unittest tests/test_marmousi2_validation_example.py`: passed.
- `python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py all --device cpu --dry-run`: passed.
- `git diff --check`: passed.

## Next Direction

Use the notebooks for visual/manual validation and the script for the same
case flow in automation. Avoid reintroducing separate backend-check wrappers
unless there is a distinct CI need.
