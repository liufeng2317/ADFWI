# 80. Full-Case Test Suite

## Purpose

Add a dedicated location for heavy end-to-end case tests that should not run as
part of the default lightweight unit-test suite. These tests validate the full
scientific workflow rather than isolated helpers.

## Change

- Added `tests/full_cases/README.md`.
- Added `tests/full_cases/test_marmousi2_acoustic_full_flow.py`.
- The Marmousi2 full-flow test has two parts:
  - a default command-construction test that runs quickly;
  - an opt-in full-case test guarded by `ADFWI_RUN_FULL_CASES=1`.

The opt-in test runs:

1. current-code forward modeling on `true_model.npz` to generate observed data
   in memory;
2. short notebook-like acoustic inversion on the initial model;
3. validation of finite loss history, loss decrease, gradient norm, and model
   update norm.

## Default Full-Flow Parameters

| Setting | Value |
| --- | --- |
| device | `npu:0` |
| observed source | `synthetic-true` |
| shot_count | 1 |
| nt_samples | 3000 |
| iterations | 2 |
| optimizer | Adam |
| lr | 10 |
| scheduler | `StepLR(step_size=200, gamma=0.75)` |
| misfit | legacy L2 |
| waveform_normalize | enabled |
| auto_update_rho | enabled |
| checkpoint_segments | 10 |

## Commands

Lightweight default check:

```bash
conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

Full case:

```bash
ADFWI_RUN_FULL_CASES=1 conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

Useful overrides:

```bash
ADFWI_RUN_FULL_CASES=1 ADFWI_FULL_CASE_DEVICE=npu:0 conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
ADFWI_RUN_FULL_CASES=1 ADFWI_FULL_CASE_ITERATIONS=5 conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

## Validation

Run in the `adfwi` conda environment:

```bash
conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
ADFWI_RUN_FULL_CASES=1 conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
conda run -n adfwi python -m py_compile tests/full_cases/test_marmousi2_acoustic_full_flow.py
git diff --check
```

Results:

- default check: 2 tests, 1 skipped, passed;
- full case: 2 tests, passed in 127.148 seconds.

## Next Step

Use this full-case test before larger real-case optimization changes. For long
100-iteration baselines, add timestamped JSON output capture rather than relying
only on unittest pass/fail.
