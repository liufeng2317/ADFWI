# 03 - Receiver Mask Contract

## Goal

验证并收敛 `Survey.receiver_masks` 的输入契约，明确它是 2D `[source, receiver]` active-receiver mask，并确认 `receiver_masks_obs` 只作为下游 observed-data mask 标志保存。

## Scope

- `ADFWI/survey/survey.py`
- `tests/test_survey_contracts.py`
- `docs/version-plans/bv1.2-survey/`

## Optimization Path

1. `Survey.set_receiver_masks()` 将输入转换为 numpy array。
2. 显式校验 mask 必须是 2D。
3. 显式校验 shape 必须等于 `(source.num, receiver.num)`。
4. 保持下游语义不变：
   - `Survey` 只保存 mask；
   - propagator 透传 mask；
   - FWI/receiver-selection 负责应用 mask；
   - `receiver_masks_obs` 仍只表示 observed waveform 是否已按 mask 处理。
5. 增加测试覆盖 array-like mask、shape 错误、维度错误、`receiver_masks_obs` 保存。

## Validation

本轮改动是输入契约收敛，不改变 receiver selection 或 FWI mask 应用逻辑：

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py`
- `conda run -n adfwi python -m unittest tests/test_receiver_selection.py`
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py`
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py`
- `git diff --check`

## Result

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py` 通过，5 tests OK。
- `conda run -n adfwi python -m unittest tests/test_receiver_selection.py` 通过，6 tests OK。
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py` 通过。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，22 tests OK。
- `git diff --check` 通过。

本轮结论：

- `Survey.set_receiver_masks()` 现在接受 array-like 输入并保存为 numpy array。
- 非 2D mask 会明确报错。
- shape 不等于 `(source.num, receiver.num)` 会明确报错。
- `receiver_masks_obs` 的保存行为已被测试覆盖。
- 没有修改 `select_or_mask_receivers()`、propagator 透传行为或 FWI mask 应用顺序。

## Stop Rule

不修改 `select_or_mask_receivers()`、不修改 FWI 中 receiver mask 应用顺序、不改变 propagator 输入和 observed-data mask 语义。
