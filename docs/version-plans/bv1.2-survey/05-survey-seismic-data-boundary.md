# 05 - Survey SeismicData Boundary

## Goal

继续收敛 `Survey` 与 `SeismicData` 的职责边界，明确哪些内容只是保存状态，哪些内容是兼容辅助处理路径，避免后续优化再次把 acquisition、waveform、plot、FWI 混在一起。

## Scope

- `ADFWI/survey/survey.py`
- `ADFWI/survey/data.py`
- `tests/test_survey_contracts.py`
- `docs/version-plans/bv1.2-survey/`

## Optimization Path

1. `Survey` 明确为 acquisition-state object：
   - 组合 `Source` 与 `Receiver`；
   - 保存 receiver mask 状态；
   - 不保存 waveform；
   - 不执行 propagator 或 FWI loss。
2. `SeismicData` 明确为 waveform snapshot object：
   - 保存 survey metadata snapshot；
   - 保存 recorded waveform dictionary；
   - 维护当前 `.npz` save/load 格式；
   - `parse_*` 与 plot 方法只是围绕已保存 waveform 的兼容辅助路径。
3. 为 `parse_acoustic_data()` 和 `parse_elastic_data()` 增加显式 recorded-state/component 检查：
   - 没有 record/load 的数据时给出清楚错误；
   - 缺少必要 component 时给出清楚错误；
   - 不改变已有合法数据的数值结果。
4. 保持边界收敛，不拆出新模块，不迁移 examples。

## Validation

本轮不改变 acoustic/elastic 解析公式和 `.npz` 格式：

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py`
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py`
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py`
- `git diff --check`

## Result

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py` 通过，9 tests OK。
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py` 通过。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，22 tests OK。
- `git diff --check` 通过。

本轮结论：

- `Survey` 已明确为 acquisition-state object，只保存 source/receiver 组合和 receiver mask 状态。
- `SeismicData` 已明确为 waveform snapshot object，`parse_*` 与 plot 方法作为围绕已保存数据的兼容辅助路径保留。
- `parse_acoustic_data()` 与 `parse_elastic_data()` 在解析前会检查是否已有 record/load 数据，以及是否存在所需 component。
- 已有合法 acoustic/elastic 解析数值保持不变，测试继续覆盖 acoustic exact match、elastic `pressure = -(txx + tzz)` 和 zero-trace normalization。

## Stop Rule

不拆 `SeismicData`，不迁移 plot API，不改 save/load `.npz` 格式，不改变 `pressure = -(txx + tzz)` 或 acoustic component 直通语义。
