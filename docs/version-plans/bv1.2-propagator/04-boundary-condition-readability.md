# 04 - Boundary Condition Readability

## Goal

Clean `boundary_condition.py` readability after locking current boundary profile
behavior with tests.

## Scope

- `ADFWI/propagator/boundary_condition.py`
- `docs/version-plans/bv1.2-propagator/`

## Optimization Path

1. Replace stale file header with a module docstring.
2. Clarify that the module builds NumPy boundary damping profiles for
   propagator wrappers.
3. Replace terse function comments with concise docstrings.
4. Remove the unused `torch` import.

## Validation

This round does not change boundary formulas, array construction, free-surface
semantics, propagator wrappers, kernels, or checkpoint behavior:

- `conda run -n adfwi python -m unittest tests/test_boundary_conditions.py`
- `conda run -n adfwi python -m py_compile ADFWI/propagator/*.py`
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py`
- `git diff --check`

## Result

- `conda run -n adfwi python -m unittest tests/test_boundary_conditions.py` 通过，4 tests OK。
- `conda run -n adfwi python -m py_compile ADFWI/propagator/*.py` 通过。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，24 tests OK。
- `git diff --check` 通过。

本轮结论：

- `boundary_condition.py` 的 stale file header 已替换为模块 docstring。
- 四个 boundary profile builder 的函数说明已收敛为简短 docstrings。
- 删除了未使用的 `torch` import。
- 没有修改任何 damping formula、array indexing、free-surface 语义、propagator wrapper dispatch 或 kernel。

## Stop Rule

No formula edits, no array indexing edits, no wrapper dispatch edits, no kernel
edits, and no forward numerical run.
