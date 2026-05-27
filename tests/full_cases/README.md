# Full-Case Integration Tests

This directory contains heavier end-to-end tests that exercise real example
cases. They are not part of the default lightweight unit-test path.

Run explicitly from the repository root:

```bash
ADFWI_RUN_FULL_CASES=1 conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

The Marmousi2 acoustic full-flow test generates observed data from the saved
true model with the current code, then runs a short inversion using notebook-like
settings. It validates forward generation, inversion loss history, gradient
processing, and model update in one command.

Useful overrides:

```bash
ADFWI_RUN_FULL_CASES=1 ADFWI_FULL_CASE_DEVICE=npu:0 conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
ADFWI_RUN_FULL_CASES=1 ADFWI_FULL_CASE_ITERATIONS=5 conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
ADFWI_RUN_FULL_CASES=1 ADFWI_FULL_CASE_OUTPUT_DIR=tests/full_cases/outputs/marmousi2_latest conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

Default parameters are intentionally close to the Marmousi2 notebook while still
small enough for a gate:

- device: `npu:0`
- shot count: 1
- time samples: 3000
- iterations: 2
- observed data: generated from `true_model.npz` in memory
- optimizer: Adam, `lr=10`
- misfit: legacy L2
- waveform normalization: enabled
- checkpoint segments: 10

When `ADFWI_FULL_CASE_OUTPUT_DIR` is set, the underlying script writes:

- `summary.json`
- `loss_history.csv`
- `loss_curve.png`
- `vp_initial_final_delta.png`

Saved full-case outputs can be compared without manual JSON parsing:

```bash
conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py \
  tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10 \
  --labels shot3,shot5
```

The fixed Marmousi2 NPU baselines can be run through presets instead of a long
environment-variable command:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 --dry-run
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 --overwrite
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot5 --overwrite
```

The latest local visual check can be generated with:

```bash
rm -rf tests/full_cases/outputs/marmousi2_latest
ADFWI_RUN_FULL_CASES=1 ADFWI_FULL_CASE_OUTPUT_DIR=tests/full_cases/outputs/marmousi2_latest conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```
