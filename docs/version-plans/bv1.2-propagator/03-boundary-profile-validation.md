# 03 - Boundary Profile Validation

## Goal

Lock the current `boundary_condition.py` behavior with focused tests before any
readability cleanup or refactor. This creates a baseline for boundary profile
shape and basic value contracts without changing formulas.

## Scope

- `tests/test_boundary_conditions.py`
- `docs/version-plans/bv1.2-propagator/`

## Optimization Path

1. Add tests for `bc_pml()`:
   - free-surface and non-free-surface padded shapes;
   - finite, non-negative damping;
   - current top/bottom damping semantics.
2. Add tests for `bc_sincos()`:
   - padded shapes;
   - finite values in `[0, 1]`;
   - current top/bottom free-surface behavior.
3. Add tests for `bc_gerjan()`:
   - padded shapes;
   - finite values in `(0, 1]`;
   - current top/bottom free-surface behavior.
4. Add tests for `bc_pml_xz()`:
   - `BCx/BCz` padded shapes;
   - finite, non-negative values;
   - current top/bottom free-surface behavior.

## Validation

This round adds tests only. It does not change boundary formulas or propagator
wrappers:

- `conda run -n adfwi python -m unittest tests/test_boundary_conditions.py`
- `conda run -n adfwi python -m py_compile tests/test_boundary_conditions.py`
- `git diff --check`

## Result

- `conda run -n adfwi python -m unittest tests/test_boundary_conditions.py` 通过，4 tests OK。
- `conda run -n adfwi python -m py_compile tests/test_boundary_conditions.py` 通过。
- `git diff --check` 通过。

本轮结论：

- `bc_pml()` 当前 free-surface / non-free-surface padded shape 和 top/bottom damping 语义已被测试锁定。
- `bc_sincos()` 当前 shape、`[0, 1]` range 和 free-surface top/bottom 语义已被测试锁定。
- `bc_gerjan()` 当前 shape、`(0, 1]` range 和 free-surface top/bottom 语义已被测试锁定。
- `bc_pml_xz()` 当前 `BCx/BCz` shape、非负 finite 值和 free-surface top/bottom 语义已被测试锁定。
- 没有修改 `boundary_condition.py`、propagator wrapper、kernel 或 forward 数值路径。

## Stop Rule

No edits to `ADFWI/propagator/boundary_condition.py`, no propagator wrapper
changes, no kernel edits, and no forward numerical run.
