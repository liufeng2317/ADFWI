# 16 FootHill Inversion Script Path Sync

## Scope

Target folder:

```text
examples/DR-FWI/reparameterization-strategy/FootHill/
```

This round checked the FootHill forward/inversion case after the forward
notebook import cleanup. It only changed Python script wrapper behavior, not
inversion parameters or algorithms.

## Findings

- `00_forward.ipynb` had already been cleaned and uses repository-root relative
  paths.
- All 49 FootHill inversion scripts used `SCRIPT_DIR` for output folders, but
  still loaded the Foothill velocity model through:

```python
model_path = "../../../datasets/foothill_source"
```

That path depends on the current working directory and can fail when the script
is launched from the repository root or from a scheduler.

- The same scripts imported `matplotlib.pyplot` and then called
  `matplotlib.use("agg")`. Setting the backend after importing pyplot is not a
  clean contract and unnecessarily hard-codes the plotting backend.

## Changes

- Replaced the Foothill dataset path with a script-location based path:

```python
model_path = str(SCRIPT_DIR.parents[2] / "datasets" / "foothill_source")
```

- Removed the explicit `matplotlib.use("agg")` and the now-unused
  `import matplotlib`.

No optimizer, iteration count, source/receiver geometry, misfit, regularization,
or output filenames were changed.

## Validation

```text
conda run -n adfwi python -m py_compile examples/DR-FWI/reparameterization-strategy/FootHill/*.py
```

Result: passed for all 49 scripts.

Residual scan found no old Foothill dataset path, old repository absolute path,
`sys.path`, or forced Agg backend in the FootHill scripts.

## Remaining Risk

Some scripts still contain duplicate local imports for misfit, regularization,
and plotting functions that are already imported at module scope. Those are
readability issues only and should be cleaned in a separate pass if FootHill is
chosen as the next active example family.
