# Inversion Wrapper Sync

## Boundary

```text
Goal:
Apply the shared bv1.2 notebook wrapper pattern to inversion notebooks.

Scope:
Tracked inversion notebooks under `examples/`, plus a local scan of ignored
research notebooks to find common wrapper risks.

Stop:
Do not change inversion physics, model geometry, source/receiver geometry,
misfit definitions, optimizer choice, learning rate, iteration count,
checkpoint settings, DIP/DR-FWI network definitions, or propagator kernels.
```

## Jupyter Update Pattern

- Restore test-generated notebook outputs and image products before committing.
- Remove old ADFWI repo-root `sys.path` setup.
- Replace ADFWI wildcard imports with explicit imports from public bv1.2
  namespaces.
- Generate explicit imports per notebook from the names actually used in that
  notebook. Do not use a large shared fallback import block; it is less readable
  than the wildcard form it replaces.
- Keep local dependency `sys.path` entries when they point to a notebook-local
  helper package, for example pretrained DIP helper code.
- Add explicit backend setup:

```python
device = "npu:0"         # Specify the CPU/GPU/NPU device
dtype = torch.float32     # Set data type to 32-bit floating point
backend = ADFWI.set_backend(device, dtype=dtype)
```

- Let `AcousticModel`, `IsotropicElasticModel`, `AnisotropicElasticModel`,
  `AcousticPropagator`, and `ElasticPropagator` inherit the active backend
  instead of repeating `device=device` or `dtype=dtype`.
- Convert `project_path` and dataset paths to notebook-local relative paths.
- Preserve notebook-local output folder names such as `data-explosion` or
  experiment-specific inversion folders.
- Clear notebook execution counts and outputs.

## Validation Result

```text
inversion_notebook_scan 205
json_bad 0
compile_bad 0
outputs 0
old_adfwi_wildcard_imports 0
repo_root_sys_path 0
missing_backend_import 0
repo_root_project_path 0
repo_root_dataset_path 0
large_shared_import_blocks 0
```

Representative import/setup-cell checks passed in the `adfwi` conda
environment for:

```text
examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb
examples/acoustic/01-model-test/02-FootHill/02_inversion.ipynb
examples/elastic/Iso-elastic-Anomaly/02_inversion.ipynb
examples/multi-scale/Iso-elastic-Marmousi2-multifreq/02_inversion.ipynb
examples/pip_usage/acoustic_inversion_test.ipynb
```

## Git Boundary

Only notebooks already tracked by git are committed in this pass. Many DR-FWI
and DIP research notebooks are under ignored example folders; they were scanned
and locally normalized for consistency, but they should not be force-added
without a separate decision because that would change the repository surface.

The external InversionNet pretrained notebook is not an ADFWI wrapper workflow
and should remain outside this sync rule.

## Remaining Risk

This pass validates wrapper consistency and notebook syntax. It does not prove
full inversion numerical equivalence for every case. Any case promoted as a
public benchmark still needs a short inversion run and result comparison.

## Next Direction

Run a small representative inversion validation set by category: acoustic,
elastic, multiscale, and pip-usage. Keep the run count low and compare loss
trend plus key saved figures before moving to full example release cleanup.
