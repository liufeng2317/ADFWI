# Article Figure Notebook Sync

## Boundary

```text
Scope:
examples/DR-FWI/Article_Figure/*.ipynb

Goal:
Apply the same wrapper cleanup used for forward/inversion examples to article
figure notebooks that still used old ADFWI imports.

Stop:
Do not change article figure plotting logic, selected datasets, model
parameters, or displayed figure composition.
```

## Applied Pattern

- Removed repo-root `sys.path.append(...)`.
- Replaced ADFWI wildcard imports with per-notebook minimal explicit imports.
- Switched CUDA literal device setup to the current example backend pattern:

```python
device = "npu:0"
dtype = torch.float32
backend = ADFWI.set_backend(device, dtype=dtype)
```

- Removed repeated `device=device` / `dtype=dtype` constructor arguments where
  the active backend can supply them.
- Kept output paths local to the article figure folder.
- Cleared notebook outputs and execution counts.

## Validation

```text
article_figure_notebooks 15
json_bad 0
ast_bad 0
outputs 0
old_sys_path_append 0
old_adfwi_wildcard_imports 0
cuda_device_left 0
manual_constructor_device_dtype_left 0
repo_root_project_path_left 0
repo_root_dataset_path_left 0
```

## Next Direction

Apply the same notebook-specific cleanup to the remaining analysis and checking
groups, starting with `examples/gradient_checking/` and then
`examples/waveform_checking/`.
