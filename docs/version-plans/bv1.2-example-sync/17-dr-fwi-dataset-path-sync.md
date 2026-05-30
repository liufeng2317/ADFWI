# 17 DR-FWI Dataset Path Sync

## Scope

Target:

```text
examples/DR-FWI/**/*.py
```

This round only fixed dataset path construction in Python scripts. It did not
change backend setup, inversion parameters, model/survey geometry, optimizers,
loss functions, regularization, or output file names.

## Problem

Many DR-FWI scripts still referenced datasets through launch-directory
dependent strings such as:

```python
model_path = "../../../datasets/overthrust_source"
load_marmousi_model(in_dir=str(SCRIPT_DIR / "../../../datasets/marmousi2_source"))
```

Those paths only work from specific working directories and are fragile when
scripts are launched from the repository root or a scheduler.

## Change

Dataset paths now locate the `examples` directory from `SCRIPT_DIR` and then
enter `examples/datasets`:

```python
next(parent for parent in SCRIPT_DIR.parents if parent.name == "examples") / "datasets" / "marmousi2_source"
```

This keeps every script self-contained and independent of the current working
directory, regardless of how deeply nested the script is under `examples/`.

## Validation

- Residual scan found no old quoted dataset paths under `examples/DR-FWI/**/*.py`.
- `py_compile` passed for all DR-FWI scripts touched by the dataset-path sync.

## Remaining Risk

The scripts still contain unrelated readability issues, especially repeated
local imports and forced Matplotlib backend setup in many files. Those should be
handled as separate, scoped passes so this path-only change remains easy to
review.
