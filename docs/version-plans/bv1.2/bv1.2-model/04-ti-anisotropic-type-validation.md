# 04 - TI Anisotropic Type Validation

## Goal

处理 `elastic_moduli_for_TI` 对非法 `anisotropic_type` 静默返回未定义结果的问题。

## Scope

- `ADFWI/model/parameters.py`
- `tests/test_model_parameters.py`
- `docs/version-plans/bv1.2-model/`

## Optimization Path

1. 保持合法类型 `vti` 和 `hti` 的 stiffness 分量排列不变。
2. 对其他类型显式抛出 `ValueError`，避免调用者传入错误类型后继续产生不明确结果。
3. 增加小张量测试：
   - `vti` 输出分量来源稳定；
   - `hti` 输出分量来源稳定；
   - 非法类型报错。

## Validation

本轮是输入校验修正，不改公式。需要验证合法路径无数值漂移，非法路径明确失败。

## Result

- `conda run -n adfwi python -m unittest tests/test_model_parameters.py` 通过，3 tests OK。
- `conda run -n adfwi python -m py_compile ADFWI/model/*.py tests/test_model_parameters.py` 通过。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，22 tests OK。
- `git diff --check` 通过。

## Stop Rule

只处理 `anisotropic_type` 非法输入，不调整 TI/HTI 公式、不重命名函数、不改 propagator 调用。
