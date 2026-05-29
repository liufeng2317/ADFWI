# All Forward Wrapper Sync

## Boundary

```text
Goal:
Apply the shared forward-notebook wrapper rules to every remaining forward case.

Scope:
All `examples/**/**/*forward*.ipynb` notebooks discovered under `examples/`.

Validation:
Notebook JSON parse, code-cell compile, residual ADFWI old-pattern scan, output
cell scan, and representative import/setup-cell execution in the `adfwi` conda
environment.

Stop:
Do not run full forward simulations. Do not change numerical parameters,
source/receiver geometry, model geometry, DIP/DR-FWI algorithms, waveform
comparison logic, or real-case data processing logic.
```

## Applied Pattern

- Removed ADFWI `sys.path` setup and ADFWI wildcard imports.
- Added explicit ADFWI imports.
- Added `ADFWI.set_backend(device, dtype=dtype)` where missing.
- Let model and propagator constructors inherit the active backend.
- Converted repo-root or absolute output paths to notebook-local paths.
- Converted `examples/datasets/...` input paths to notebook-local relative paths.
- Replaced repeated output-directory creation with `exist_ok=True` loops where
  present.
- Cleared notebook outputs and execution counts.

The Devito comparison notebooks that import `ADSWIT` were left with their
external `ADSWIT` wildcard imports, because those are not ADFWI API imports and
belong to the comparison dependency.

## Validation Result

```text
forward_count 105
json_bad 0
compile_bad 0
outputs 0
residual 0
```

Representative setup-cell checks passed in the `adfwi` conda environment for:

```text
examples/acoustic/01-model-test/01-Marmousi2/01_forward.ipynb
examples/elastic/Iso-elastic-Anomaly/01_forward.ipynb
examples/multi-scale/Iso-acoustic-Marmousi2-multifreq/01_forward.ipynb
examples/gradient_checking/Elastic-Marmousi2/01_forward.ipynb
examples/pip_usage/acoustic_forward_test.ipynb
examples/DR-FWI/multi-parameters/ISO_elastic/Marmousi2-vp_vs/00_forward.ipynb
examples/dip/DIP-ADFWI/01_Multi-CNN/01_forward.ipynb
examples/real-case-study/Mongolia/03_forward_inverted_model.ipynb
```

## Remaining Risk

This pass validates wrapper consistency and notebook syntax, not full forward
waveform equivalence. Any case promoted to a benchmark still needs a focused
short forward run and output comparison.

## Next Direction

Start inversion notebook synchronization with the same rule: pick one
representative case first, validate the wrapper and a short run, then broaden by
category.
