# Marmousi2 Inversion Wrapper Sync

## Boundary

```text
Goal:
Start inversion notebook synchronization with the original acoustic Marmousi2
inversion notebook.

Scope:
examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb

Validation:
Notebook JSON parse, code-cell compile, residual old-pattern scan, output-cell
scan, and import/setup-cell execution in the `adfwi` conda environment.

Stop:
Do not change inversion parameters, optimizer, scheduler, loss function,
gradient mask, batch size, checkpointing, or iteration count.
```

## Change

- Removed notebook-local `sys.path`.
- Replaced ADFWI wildcard imports with explicit imports.
- Kept notebook-local output path:

```python
project_path = "./data"
for subdir in ("model", "waveform", "survey", "inversion"):
    os.makedirs(os.path.join(project_path, subdir), exist_ok=True)
```

- Added standard backend setup:

```python
device = "npu:0"
dtype = torch.float32
backend = ADFWI.set_backend(device, dtype=dtype)
```

- Updated Marmousi input path relative to the notebook directory:

```python
load_marmousi_model(in_dir="../../../../examples/datasets/marmousi2_source")
```

- Let `AcousticModel` and `AcousticPropagator` inherit the active backend.
- Cleared outputs and execution counts.

## Validation Result

```text
compile_bad []
old_patterns []
outputs 0
exec_counts 0
setup_ok ['ADFWI', 'AcousticFWI', 'AcousticModel', 'AcousticPropagator', 'GradProcessor', 'Receiver', 'SeismicData', 'Source', 'Survey']
```

## Next Direction

Use this notebook as the initial inversion wrapper template. Before broad
inversion synchronization, run one short inversion validation with reduced
iterations to ensure the wrapper changes preserve behavior.
