# 01 - Propagator Wrapper Readability

## Goal

Clarify the public acoustic/elastic propagator wrappers without changing
numerical behavior.

## Scope

- `ADFWI/propagator/acoustic_propagator.py`
- `ADFWI/propagator/elastic_propagator.py`
- `docs/version-plans/bv1.2-propagator/`

## Optimization Path

1. Replace stale file headers with concise module docstrings.
2. Clarify that wrapper classes own model/survey/backend adaptation and kernel
   dispatch.
3. Clarify that finite-difference formulas remain in kernel modules.
4. Remove clearly unused wrapper imports only:
   - `numpy`;
   - `matplotlib.pyplot`;
   - unused `torch.jit` alias in elastic wrapper.

## Validation

This round does not change kernel code, boundary formulas, checkpoint behavior,
or forward dispatch arguments:

- `conda run -n adfwi python -m py_compile ADFWI/propagator/*.py`
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py`
- `git diff --check`

## Result

- `conda run -n adfwi python -m py_compile ADFWI/propagator/*.py` 通过。
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，22 tests OK。
- `git diff --check` 通过。

本轮结论：

- `AcousticPropagator` 和 `ElasticPropagator` 的 stale file headers 已替换为模块 docstrings。
- wrapper class docstrings 已明确其职责是 model/survey/backend adaptation 和 kernel dispatch。
- 删除了 wrapper 内明显未使用的 import。
- 没有修改 kernel、boundary-condition formula、checkpoint behavior、forward dispatch arguments 或 tensor extraction logic。

## Stop Rule

No kernel edits, no boundary-condition formula edits, no checkpoint edits, no
shape/device extraction refactor.
