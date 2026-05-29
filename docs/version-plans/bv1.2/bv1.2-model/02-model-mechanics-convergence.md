# 02 - Model Mechanics Convergence

## Goal

收敛 `ADFWI/model` 内部重复的模型参数生命周期逻辑，不改变声学、各向同性弹性、各向异性弹性的物理公式和数值路径。

## Scope

- `ADFWI/model/base.py`
- `ADFWI/model/acoustic_model.py`
- `ADFWI/model/elastic_model.py`

本轮不拆分 `runtime`、不修改传播器、不迁移 examples。

## Optimization Path

1. 在 `AbstractModel` 中统一薄层公共逻辑：
   - 参数注册：`_register_model_parameter`
   - 参数边界：`_set_parameter_bounds`
   - 梯度开关元数据：`_set_requires_grad_flags`
   - 水层 mask：`_prepare_water_layer_mask`
   - 有边界参数裁剪：`clip_params`
2. `AcousticModel`、`IsotropicElasticModel`、`AnisotropicElasticModel` 只保留模型自身定义、经验关系和弹性参数刷新。
3. 删除三个子类中重复的参数注册、bounds、requires_grad、水层 mask 和 `clip_params` 实现。

## Validation

本轮需要进行轻量但直接的数值回归：

- 小模型声学/弹性/各向异性模型刷新前后结果逐项对比。
- `py_compile` 检查 `ADFWI/model` 语法。
- backend integration 测试确认后端选择路径未受影响。

若所有数值数组 `max_abs = 0`，则说明本轮是结构收敛，不引入数值漂移。

## Result

- 小模型声学/弹性/各向异性模型刷新对比通过，所有数组 `max_abs=0.000000e+00`。
- `conda run -n adfwi python -m py_compile ADFWI/model/*.py` 通过。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，22 tests OK。
- `git diff --check` 通过。

## Stop Rule

到公共生命周期逻辑收敛为止。不要继续扩展到参数公式重写、模型文件拆分、传播器接口或 examples 调用方式。
