# Multiscale Forward Template

## Boundary

```text
Goal:
Use the acoustic Marmousi2 multifrequency forward notebook as the representative
special forward case after the standard acoustic/elastic forward sync.

Scope:
examples/multi-scale/Iso-acoustic-Marmousi2-multifreq/01_forward.ipynb

Validation:
Notebook JSON parse, code-cell compile, import/setup cell execution in the
adfwi conda environment, and residual old-pattern scan.

Stop:
Do not change the multifrequency filtering logic, model geometry, source and
receiver geometry, or full forward runtime behavior.
```

## Change

- Use notebook-local relative paths for outputs and input datasets:

```python
project_path = "./data"
marmousi_model = load_marmousi_model(in_dir="../../datasets/marmousi2_source")
```

- Replaced repeated output-directory creation with:

```python
for subdir in ("model", "waveform", "survey"):
    os.makedirs(os.path.join(project_path, subdir), exist_ok=True)
```

- Kept explicit ADFWI imports for the forward and multiscale display workflow.
- Added the standard backend setup pattern.
- Let `AcousticModel` and `AcousticPropagator` inherit the active backend.
- Cleared notebook outputs and execution counts before versioning.

## Sync Template

For similar special forward notebooks:

```text
1. Make imports explicit.
2. Set backend once with ADFWI.set_backend(device, dtype=dtype).
3. Let model and propagator constructors inherit the active backend.
4. Set project_path relative to the notebook directory, usually ./data or a
   case-specific local data folder.
5. Set dataset paths relative to the notebook directory, not repo-root strings
   such as ./examples/datasets/....
6. Keep source, receiver, model geometry, filtering, and forward physics
   unchanged.
7. Clear outputs before committing.
```

## Follow-Up Sync

The same wrapper/path template was applied to:

```text
examples/multi-scale/Iso-elastic-Marmousi2-multifreq/01_forward.ipynb
examples/new_features/source_encoding/01_forward.ipynb
examples/acoustic/Article-Adding-Test/source_encoding/01_forward.ipynb
```

Path corrections:

```text
multi-scale elastic:
  project_path = "./data"
  in_dir = "../../datasets/marmousi2_source"

new_features/source_encoding:
  project_path = "./data-8source"
  in_dir = "../../datasets/marmousi2_source"

Article-Adding-Test/source_encoding:
  project_path = "./data-8source"
  in_dir = "../../../datasets/marmousi2_source"
```

## Validation Result

```text
python -m json.tool examples/multi-scale/Iso-acoustic-Marmousi2-multifreq/01_forward.ipynb
compile_bad []
old_patterns []
non_null_execution_counts 0
output_cells 0
setup_ok ['ADFWI', 'AcousticModel', 'AcousticPropagator', 'Receiver', 'SeismicData', 'Source', 'Survey', 'numpy2tensor', 'wavelet']
```

## Next Direction

Use this template for the remaining special forward notebooks only in small
groups by category, starting with gradient-checking or pip-usage examples.
