# 125 - Marmousi2 Validation Parameter Notebooks

## Optimization Path

The first validation notebooks were too wrapper-oriented: they launched staged
commands but did not expose the experiment definitions clearly. For a scientific
example, the notebook should show the parameters, model definition, observation
system, and source wavelet before running forward modeling or inversion.

## Change

- Updated `notebooks/01_forward_modeling.ipynb` to explicitly define:
  - validation paths;
  - forward-modeling parameters;
  - true-model file and model geometry summary;
  - source/receiver observation-system metadata;
  - source wavelet construction and summary;
  - dry-run and optional execution commands.
- Updated `notebooks/02_inversion.ipynb` to explicitly define:
  - inversion parameters;
  - true model versus initial model roles;
  - survey subset and wavelet settings;
  - 10-iteration and optional 100-iteration run controls.
- Extended `run_validation.py` so notebook-defined case/model/wavelet
  parameters are passed through to the maintained script entry points.
- Updated README and dry-run tests for the explicit parameter path.

## Scientific Contract

- No FWI core code changed.
- No original Marmousi2 notebook changed.
- Numerical behavior remains delegated to the existing script-style validation
  commands.
- The notebook now makes the experiment definition visible before execution.

## Validation

Completed validation:

- `conda run -n adfwi python -m py_compile examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py tests/test_marmousi2_validation_example.py`: passed.
- `python -m json.tool examples/validation/marmousi2_acoustic_reduced/notebooks/01_forward_modeling.ipynb`: passed.
- `python -m json.tool examples/validation/marmousi2_acoustic_reduced/notebooks/02_inversion.ipynb`: passed.
- `conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py all --device cpu --dry-run`: passed.
- `conda run -n adfwi python -m unittest tests/test_marmousi2_validation_example.py`: passed.
- `git diff --check`: passed.

## Next Direction

Run the forward-modeling notebook first. After inspecting the model, survey,
wavelet, and true-model forward output, run the inversion notebook for 10
iterations and inspect the saved loss/model-update artifacts before attempting
100 iterations.
