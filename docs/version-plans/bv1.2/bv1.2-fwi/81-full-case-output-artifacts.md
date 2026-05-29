# 81. Full-Case Output Artifacts

## Purpose

The opt-in Marmousi2 full-case test previously returned only pass/fail status.
That is not enough for checking whether the current result visually matches the
older notebook behavior. This pass adds optional result artifacts for manual
inspection.

## Change

- Added `--output-dir` to
  `scripts/examples/marmousi2_acoustic_reduced_inversion.py`.
- Added `ADFWI_FULL_CASE_OUTPUT_DIR` support to
  `tests/full_cases/test_marmousi2_acoustic_full_flow.py`.
- Added `tests/full_cases/outputs/` to `.gitignore`.
- The output directory now contains:
  - `summary.json`;
  - `loss_history.csv`;
  - `loss_curve.png`;
  - `vp_initial_final_delta.png`.

## Command

```bash
rm -rf tests/full_cases/outputs/marmousi2_latest
ADFWI_RUN_FULL_CASES=1 ADFWI_FULL_CASE_OUTPUT_DIR=tests/full_cases/outputs/marmousi2_latest conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

## Current Local Artifacts

The command completed with `status: ok` and wrote:

| File | Size |
| --- | ---: |
| `summary.json` | 1793 bytes |
| `loss_history.csv` | 53 bytes |
| `loss_curve.png` | 53771 bytes |
| `vp_initial_final_delta.png` | 273301 bytes |

PNG dimensions:

| File | Dimensions |
| --- | --- |
| `loss_curve.png` | 1200 x 750 |
| `vp_initial_final_delta.png` | 2100 x 675 |

Key metrics from `summary.json`:

| Metric | Value |
| --- | ---: |
| observed_source | synthetic-true |
| synthetic_true_forward_seconds | 6.356838984414935 |
| iterations | 2 |
| initial_loss | 2197.3671875 |
| final_loss | 2058.807861328125 |
| loss_delta | -138.559326171875 |
| loss_relative_delta | -0.0630569742554122 |
| vp_grad_norm | 0.2128792554140091 |
| vp_update_norm | 2300.30810546875 |
| inversion_seconds | 77.52065790817142 |

## Validation

```bash
conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
ADFWI_RUN_FULL_CASES=1 ADFWI_FULL_CASE_OUTPUT_DIR=tests/full_cases/outputs/marmousi2_latest conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
conda run -n adfwi python -m py_compile scripts/examples/marmousi2_acoustic_reduced_inversion.py tests/full_cases/test_marmousi2_acoustic_full_flow.py
git diff --check
```

## Next Step

Use the generated PNGs to compare current behavior against the historical
notebook outputs. If the visual trend is acceptable, add timestamped output
directories for longer 10/100-iteration full-case baselines.
