# Gradient Checking Forward Sync

## Boundary

```text
Goal:
Make gradient-checking forward notebooks follow the same example wrapper and
path rules without changing gradient-checking behavior.

Scope:
examples/gradient_checking/Acoustic-Marmousi2/01_forward.ipynb
examples/gradient_checking/Elastic-Marmousi2/01_forward.ipynb

Validation:
Notebook JSON parse, code-cell compile, import/setup cell execution in the
adfwi conda environment, and residual old-pattern scan.

Stop:
Do not run full forward simulations or change gradient-checking parameters,
source/receiver geometry, model geometry, or numerical formulas.
```

## Applied Pattern

- Use notebook-local output paths:

```python
project_path = "./data"
for subdir in ("model", "waveform", "survey"):
    os.makedirs(os.path.join(project_path, subdir), exist_ok=True)
```

- Use notebook-local dataset paths:

```python
load_marmousi_model(in_dir="../../datasets/marmousi2_source")
```

- Use explicit imports.
- Set the active backend once with `ADFWI.set_backend(device, dtype=dtype)`.
- Let model and propagator constructors inherit the active backend.
- Clear outputs and execution counts before committing.

## Validation Result

```text
Elastic-Marmousi2/01_forward.ipynb:
  compile_bad []
  old_patterns []
  outputs 0
  exec_counts 0
  setup_ok ['ADFWI', 'ElasticPropagator', 'IsotropicElasticModel', 'Receiver', 'SeismicData', 'Source', 'Survey', 'wavelet']
```

## Next Direction

Use the same wrapper/path rules for `examples/pip_usage` forward notebooks,
then leave DIP, real-case, issue-reproduction, and waveform-checking examples
for separate targeted passes.
