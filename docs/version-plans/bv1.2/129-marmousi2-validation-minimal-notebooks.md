# 129 - Marmousi2 Validation Minimal Notebooks

## Optimization Path

The validation notebooks should be easy to compare with the original
Marmousi2 example. The previous direct-API notebooks were correct, but they
introduced too much helper structure and summary output for interactive
instance validation.

## Change

- Reworked the validation notebooks into minimal variants of the original
  Marmousi2 notebooks.
- Preserved the original top-to-bottom order:
  - imports;
  - basic parameters;
  - true or initial model;
  - source/receiver survey;
  - wavelet and survey plots;
  - propagator and damping plot;
  - forward modeling or inversion;
  - waveform/model/loss visualization.
- Removed the centralized helper-style notebook layout.
- Kept only validation-specific changes:
  - separate validation output folder;
  - `npu:0` backend setup;
  - reduced `validation_shots`;
  - 10-iteration inversion default;
  - `checkpoint_segments=1`.

## Scientific Contract

- No FWI core code changed.
- No original Marmousi2 example notebook changed.
- The notebooks still avoid CLI/subprocess execution.
- Forward validation generates observed data from the true Marmousi2 model;
  inversion validation loads that generated observed data.

## Validation

Completed validation:

- Notebook JSON parsing passed for both validation notebooks.
- `conda run -n adfwi python -m py_compile tests/test_marmousi2_validation_example.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_marmousi2_validation_example.py`: passed.
- `git diff --check`: passed.

## Next Direction

Use the minimal notebooks for visual validation. Run
`01_forward_modeling.ipynb` first, then `02_inversion.ipynb`; increase
`validation_shots` or `iteration` only after the small case visually matches
expectations.
