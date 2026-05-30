# 04 - Source Receiver Contracts

## Goal

收敛 `Source` 和 `Receiver` 的输入契约，明确 normal source/receiver 的位置输入是一维 array-like，encoded source 保留多维编码语义，同时不改变已有正演/反演数值路径。

## Scope

- `ADFWI/survey/source.py`
- `ADFWI/survey/receiver.py`
- `tests/test_survey_contracts.py`
- `docs/version-plans/bv1.2-survey/`

## Optimization Path

1. 在 `Source.add_source()`、`Source.add_sources()`、`Source.add_encoded_sources()` 入口统一将 array-like 输入转换为 numpy array。
2. 明确 normal `add_sources()` 的位置输入必须是 1D，避免二维输入导致 `num`、location 和 wavelet 数量隐式错位。
3. 明确 `add_source()` 和 `add_sources()` 的 source wavelet 必须是长度为 `nt` 的 1D array。
4. 明确 encoded source wavelet 的最后一维必须是 `nt`，并保留已有 encoded leading axes 语义。
5. 在 `Receiver.add_receivers()` 入口接受 array-like 输入，并明确 receiver 位置输入必须是 1D。
6. 将过去可能暴露为 `AttributeError` 的非法输入收敛为清楚的 `ValueError`。

## Validation

本轮是 survey 输入契约收敛，不修改 propagator、FWI iteration 或数据解析数值行为：

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py`
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py`
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py`
- `conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py check --output-root examples/validation/marmousi2_acoustic_reduced/outputs/source_receiver_contract_check --device npu:0 --shots 3 --checkpoint-segments 1`
- `git diff --check`

## Result

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py` 通过，8 tests OK。
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py` 通过。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，22 tests OK。
- `conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py check --output-root examples/validation/marmousi2_acoustic_reduced/outputs/source_receiver_contract_check --device npu:0 --shots 3 --checkpoint-segments 1` 通过，status OK。
- `git diff --check` 通过。

本轮结论：

- `Source` 的 normal source 添加接口现在接受 array-like wavelet/location 输入，并对非法 shape 给出明确 `ValueError`。
- `Source.add_encoded_sources()` 保留 encoded leading axes，仅明确最后一维为 `nt`。
- `Receiver.add_receivers()` 现在接受 array-like 位置输入，并明确 normal receiver location 必须是 1D。
- 没有修改 source/receiver 的正常输出 shape：normal source location `(src_num, 2)`、wavelet `(src_num, nt)`，receiver location `(rcv_num, 2)`。
- 轻量 Marmousi2 validation check 在 NPU 上通过，survey 为 3 shots、200 receivers、3000 time samples。

## Stop Rule

不修改 source/receiver 存储结构，不迁移 examples，不调整 `get_type(unique=True)` 顺序语义，不进入 propagator 或 FWI 核心路径。
