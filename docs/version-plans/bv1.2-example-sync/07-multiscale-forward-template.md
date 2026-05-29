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

- Replaced repeated output-directory creation with:

```python
for subdir in ("model", "waveform", "survey"):
    os.makedirs(os.path.join(project_path, subdir), exist_ok=True)
```

- Kept explicit ADFWI imports for the forward and multiscale display workflow.
- Added the standard backend setup pattern.
- Let `AcousticModel` and `AcousticPropagator` inherit the active backend.
- Cleared notebook outputs and execution counts before versioning.

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

Use this notebook as the multiscale forward template, then handle the elastic
multifrequency forward notebook with the same wrapper-only rule.
