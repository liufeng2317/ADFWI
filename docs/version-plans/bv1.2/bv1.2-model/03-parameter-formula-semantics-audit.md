# 03 - Parameter Formula Semantics Audit

## Goal

审计 `ADFWI/model/parameters.py` 中的公式语义边界，使读代码的人能清楚区分：

- 输入模型量：`vp`、`vs`、`rho`、Thomsen 参数；
- 派生物理量：`mu`、`lambda`、`lambda + 2mu`、buoyancy；
- 弹性刚度分量：`C11` 到 `C66`；
- 交错网格量：`bx`、`bz`、`muxz`、`C44/C55/C66`。

本轮只做语义审计和说明补全，不改公式实现。

## Scope

- `ADFWI/model/parameters.py`
- `docs/version-plans/bv1.2-model/`

## Audit Findings

1. `parameters.py` 的定位是合理的：它只应负责物理参数变换和交错网格准备，不应包含模型对象生命周期、plot 或 FWI 逻辑。
2. `vs_vp_to_Lame` 旧说明有歧义：`lamu` 不是 `lambda`，而是 `rho * vp**2 = lambda + 2mu`。本轮已在 docstring 中明确。
3. `elastic_moduli_init` 的旧矩阵说明最后两行存在语义混乱，容易把 `C56`/`C66` 顺序看错。本轮已改为明确的 21 分量上三角列表顺序。
4. `thomsen_to_elastic_moduli` 和 `elastic_moduli_to_thomsen` 是公式敏感区域，后续任何公式改动都必须做数值回归。
5. `parameter_staggered_grid` 的输出形状原来没有说明。本轮已补充 `bx/bz/muxz/C44/C55/C66` 的 shape 约定。
6. 暂不修改的问题：
   - `elastic_moduli_for_TI` 对非法 `anisotropic_type` 没有显式报错；
   - `parameter_staggered_grid` 的五点平均实现需要单独数值审计后再决定是否调整。

## Validation

本轮没有修改公式表达式。验证重点是确认说明变更没有破坏导入和现有公式行为：

- `py_compile` 检查 `ADFWI/model/*.py`。
- 小张量公式 smoke test：
  - `vs_vp_to_Lame` 与手算公式一致；
  - Thomsen -> stiffness -> Thomsen round trip 在浮点容差内一致；
  - `parameter_staggered_grid` 输出 shape 符合说明。

## Result

- `conda run -n adfwi python -m py_compile ADFWI/model/*.py` 通过。
- 公式 smoke test 通过：
  - `roundtrip_vp_max_abs = 0.0`
  - `roundtrip_vs_max_abs = 0.0`
  - `roundtrip_eps_max_abs = 4.470348358154297e-08`
  - `roundtrip_delta_max_abs = 5.587935447692871e-08`
  - `roundtrip_gamma_max_abs = 2.9802322387695312e-08`
  - staggered shapes: `bx=(4, 4)`, `bz=(3, 5)`, `muxz/C44/C55/C66=(2, 3)`
- `git diff --check` 通过。

## Stop Rule

停止在语义说明和审计记录。不要在本轮修改公式、重命名公开函数或调整交错网格平均方式。
