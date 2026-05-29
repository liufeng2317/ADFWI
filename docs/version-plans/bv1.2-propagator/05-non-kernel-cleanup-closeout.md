# 05 - Non Kernel Cleanup Closeout

## Goal

Close the current non-kernel propagator cleanup round in one bounded pass.

This round keeps the earlier caution around propagator numerical code, but does
not split pure readability work into one task per file.

## Scope

- `ADFWI/propagator/gradient_process.py`
- `ADFWI/propagator/__init__.py`
- `docs/version-plans/bv1.2-propagator/`

The active kernel files remain out of scope:

- `ADFWI/propagator/acoustic_kernels.py`
- `ADFWI/propagator/acoustic_kernels_bs.py`
- `ADFWI/propagator/elastic_kernels.py`

## Optimization Path

1. Clarify `gradient_process.py` as gradient post-processing, not forward
   propagation.
2. Replace stale `gradient_process.py` file header with a module docstring.
3. Add concise docstrings for helper functions and `GradProcessor`.
4. Add a concise public API docstring in `ADFWI/propagator/__init__.py`.
5. Do not change any gradient taper, smoothing, mask, illumination, or
   normalization logic.

## Validation

This round does not change kernels, boundary formulas, wrapper tensor
extraction, checkpoint behavior, or gradient numerical logic:

- `conda run -n adfwi python -m unittest tests/test_boundary_conditions.py`
- `conda run -n adfwi python -m unittest tests/test_torch_grad_processor.py`
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py`
- `conda run -n adfwi python -m py_compile ADFWI/propagator/*.py`
- `git diff --check`

## Result

- `conda run -n adfwi python -m unittest tests/test_boundary_conditions.py` 通过，4 tests OK。
- `conda run -n adfwi python -m unittest tests/test_torch_grad_processor.py` 通过，7 tests OK。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，24 tests OK。
- `conda run -n adfwi python -m py_compile ADFWI/propagator/*.py` 通过。
- `git diff --check` 通过。

本轮结论：

- `gradient_process.py` 已明确为 FWI gradient post-processing utilities。
- `ADFWI/propagator/__init__.py` 已增加 public API docstring 并整理导入格式。
- 没有修改 legacy `GradProcessor`、`TorchGradProcessor`、taper/smoothing/mask/illumination/normalization 数值逻辑。
- 没有修改 active kernel files、boundary formulas、wrapper tensor extraction、checkpoint behavior 或 forward output contracts。

## Remaining Risks

These are known risks, not active cleanup tasks:

| Risk | Current decision | When to revisit |
| --- | --- | --- |
| `gradient_process.py` is exported from `ADFWI.propagator` but semantically belongs to FWI/runtime post-processing | keep export for compatibility | revisit only with a broader public API migration plan |
| legacy NumPy `GradProcessor` mutates input gradients in places | preserve behavior | revisit only with exact NumPy/Torch comparison and FWI trajectory validation |
| active kernels still contain old comments and large formula blocks | do not touch during non-kernel cleanup | revisit only with forward and backward numerical references |
| acoustic boundary wrapper forces `free_surface=False` for acoustic damping builders | documented as current behavior | revisit only as a numerical boundary-condition task |

## Stop Rule

Stop non-kernel cleanup after this round. Future propagator work should be
driven by a concrete validation need, performance target, or clear bug.
