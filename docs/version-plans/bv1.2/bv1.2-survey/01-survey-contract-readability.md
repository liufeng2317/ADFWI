# 01 - Survey Contract Readability

## Goal

清晰化 `ADFWI/survey` 的职责和契约，不改变数值行为、receiver mask 语义或 `.npz` 保存格式。

## Scope

- `ADFWI/survey/source.py`
- `ADFWI/survey/receiver.py`
- `ADFWI/survey/survey.py`
- `ADFWI/survey/data.py`
- `tests/test_survey_contracts.py`
- `docs/version-plans/bv1.2-survey/`

## Optimization Path

1. 将旧文件头替换为模块职责 docstring。
2. 明确 `Source`、`Receiver`、`Survey`、`SeismicData` 各自应拥有的内容：
   - source geometry/wavelet/moment tensor；
   - receiver geometry/type；
   - survey acquisition composition 和 receiver masks；
   - waveform data snapshot、save/load、parse/plot。
3. 修正明显误导的 docstring，例如 receiver 方法中写成 source 的说明。
4. 移除重复或未使用 import。
5. 增加聚焦契约测试，锁定当前 source/receiver shape、receiver mask shape、SeismicData record/save/load round trip。

## Validation

本轮不改数值公式和数据格式，验证目标是确认现有契约未被破坏：

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py`
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py`
- `conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py check --output-root examples/validation/marmousi2_acoustic_bv12/outputs/survey_contract_check --device npu:0 --shots 3 --checkpoint-segments 1`
- `git diff --check`

## Result

- `conda run -n adfwi python -m unittest tests/test_survey_contracts.py` 通过，3 tests OK。
- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py tests/test_survey_contracts.py` 通过。
- `conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py check --output-root examples/validation/marmousi2_acoustic_bv12/outputs/survey_contract_check --device npu:0 --shots 3 --checkpoint-segments 1` 通过。
- validation `check` 结果：`shots=3`、`receivers=200`、`nt=3000`、backend `npu:0/float32`。
- `git diff --check` 通过。

本轮没有修改 waveform 字典内容、receiver mask 下游行为、`.npz` 保存格式或 propagator 输入。

## Stop Rule

到职责说明和契约测试通过为止。不要在本轮调整 waveform 字典内容、receiver mask 下游行为、`.npz` 格式或 propagator 输入。
