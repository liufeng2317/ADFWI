# 02 - SeismicData Contracts

## Goal

验证并收敛 `SeismicData` 的 `record_data/save/load/parse` 契约，确保 waveform 内容、shape、dtype 和保存/加载行为清晰。

## Scope

- `ADFWI/survey/data.py`
- `tests/test_survey_contracts.py`
- `docs/version-plans/bv1.2-survey/`

## Optimization Path

1. 明确 `record_data()` 的所有权：`SeismicData` 保存转换后的 numpy 副本，不再原地修改调用者传入的 waveform dict。
2. 保持当前 `.npz` 保存格式不变，仍通过 `allow_pickle=True` 加载 waveform dict。
3. 增加 acoustic 契约测试：
   - 输入 torch tensor；
   - `SeismicData.data` 内部为 numpy array；
   - `parse_acoustic_data()` 与输入数值 exact match；
   - `save()`/`load()` round trip exact match。
4. 增加 elastic/normalization 契约测试：
   - `pressure = -(txx + tzz)`；
   - `txz/vx/vz` 直通；
   - zero trace normalization 保持 zero trace，非零 trace 按最大绝对值归一化。

## Validation

本轮涉及 `record_data()` 副作用收敛，需要验证保存内容和 parse 数值不漂移：

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py`
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py`
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py`
- `git diff --check`

## Result

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py` 通过，4 tests OK。
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py` 通过。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，22 tests OK。
- `git diff --check` 通过。

本轮结论：

- `record_data()` 现在将 waveform dict 转换为 `SeismicData.data` 内部 numpy 副本，不再原地修改调用者传入的 dict。
- acoustic `parse_acoustic_data()` 与输入 waveform 数值 exact match。
- `save()`/`load()` round trip 后 `p/u/w`、source/receiver metadata、`nt/dt` exact match。
- elastic `parse_elastic_data()` 的 `pressure = -(txx + tzz)`、`txz/vx/vz` 直通行为已被测试锁定。
- zero trace normalization 行为已被测试锁定。

## Stop Rule

不改变 `.npz` 格式、不改变 waveform key、不改变 acoustic/elastic parse 输出定义、不迁移 examples。
