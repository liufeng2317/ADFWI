# 06 - Survey Plot Helper Contract

## Goal

审计 `Survey` 的 plotting helper，收敛 `plot_single_shot()` 中 receiver mask 索引后的坐标 shape，避免辅助可视化路径传出二维 receiver 坐标。

## Scope

- `ADFWI/survey/survey.py`
- `tests/test_survey_contracts.py`
- `docs/version-plans/bv1.2-survey/`

## Optimization Path

1. 保持 `Survey.plot()` 和 `Survey.plot_single_shot()` 仍只是 acquisition plotting helper。
2. `plot_single_shot()` 使用 `np.flatnonzero(receiver_mask)` 取得 active receiver index。
3. 保证传给 `plot_survey()` 的 active receiver `rcv_x/rcv_z` 仍是一维坐标数组，而不是 `np.argwhere()` 产生的 `(active, 1)` 形状。
4. 增加 mock 测试锁定 masked single-shot plotting helper 的 receiver coordinate shape。

## Validation

本轮只影响 plotting/helper 输入 shape，不改变 receiver mask 保存语义、propagator 输入或 FWI 数值路径：

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py`
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py`
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py`
- `git diff --check`

## Result

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py` 通过，10 tests OK。
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py` 通过。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，22 tests OK。
- `git diff --check` 通过。

本轮结论：

- `plot_single_shot()` 现在用 `np.flatnonzero(receiver_mask)` 取得 active receiver index。
- masked plotting helper 传给 `plot_survey()` 的 `rcv_x/rcv_z` 保持为 1D 坐标数组。
- 该修改只影响可视化辅助路径的输入形状，不改变 receiver mask 保存、propagator 输入或 FWI 数值路径。

## Stop Rule

不改 `plot_survey()`，不迁移 examples/notebooks，不改变 receiver mask 的 propagator/FWI 应用语义。
