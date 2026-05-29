# 05 - Staggered Grid Validation

## Goal

验证 `parameter_staggered_grid` 的当前数值行为和 shape 契约，确认它与弹性模型 `forward()` 后提供给传播器的输入形状一致。

## Scope

- `tests/test_model_parameters.py`
- `docs/version-plans/bv1.2-model/`

本轮不修改 `ADFWI/model/parameters.py` 的公式表达式。

## Validation Path

1. 用小张量显式复现当前 `parameter_staggered_grid` stencil，锁定 `bx/bz/muxz/C44/C55/C66` 的数值行为。
2. 用常量场验证交错平均不会改变常量物理量。
3. 构造小型 `IsotropicElasticModel` 和 `AnisotropicElasticModel`，调用 `forward()` 后检查：
   - `bx.shape == (nz, nx - 1)`
   - `bz.shape == (nz - 1, nx)`
   - `muxz.shape == (nz - 2, nx - 2)`
   - `CC[18]` 作为传播器使用的 `C55`，shape 为 `(nz - 2, nx - 2)`。

## Result

- `conda run -n adfwi python -m unittest tests/test_model_parameters.py` 通过，6 tests OK。
- `conda run -n adfwi python -m py_compile ADFWI/model/*.py tests/test_model_parameters.py` 通过。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，22 tests OK。
- `git diff --check` 通过。

本轮结论：

- 当前 `parameter_staggered_grid` 的数值 stencil 已被小张量测试锁定。
- 常量场经过当前交错平均后保持常量值。
- 小型各向同性/各向异性弹性模型 `forward()` 后的 `bx/bz/muxz/C55` shape 与传播器入口需求一致。
- 本轮没有修改交错网格公式。

## Stop Rule

只验证当前交错网格行为。若后续要调整 stencil 或 padding 关系，需要单独开启公式修正轮，并做前后数值对比。
