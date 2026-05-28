# 86. Marmousi2 NPU 5-Shot Checkpoint-10 Gate

## Purpose

Increase the full-length Marmousi2 NPU benchmark from 3 shots to 5 shots while
keeping the current faster checkpoint setting: 10 inversion iterations and
`checkpoint_segments=10`.

This probes whether NPU throughput continues improving with a larger shot batch
or begins to reach a runtime and memory knee.

## Command

```bash
rm -rf tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10
ADFWI_RUN_FULL_CASES=1 \
ADFWI_FULL_CASE_SHOT_COUNT=5 \
ADFWI_FULL_CASE_ITERATIONS=10 \
ADFWI_FULL_CASE_CHECKPOINT_SEGMENTS=10 \
ADFWI_FULL_CASE_OUTPUT_DIR=tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10 \
conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

## Reproducibility State

| Item | Value |
| --- | --- |
| conda environment | adfwi |
| git branch | bv1.2 |
| git baseline before run | `b5aa615` |
| dirty worktree during run | `examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb` metadata change |
| output directory | `tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10` |

## Configuration

| Setting | Value |
| --- | --- |
| device | npu:0 |
| observed source | synthetic-true |
| shot_count | 5 |
| nt_samples | 3000 |
| iterations | 10 |
| optimizer | Adam |
| lr | 10 |
| scheduler | `StepLR(step_size=200, gamma=0.75)` |
| misfit | legacy L2 |
| waveform_normalize | true |
| auto_update_rho | true |
| checkpoint_segments | 10 |

## Result

The full-case unittest passed:

```text
Ran 2 tests in 367.343s
OK
```

Saved local artifacts:

| File | Size |
| --- | ---: |
| `tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10/summary.json` | 1958 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10/loss_history.csv` | 191 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10/loss_curve.png` | 48311 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10/vp_initial_final_delta.png` | 348670 bytes |

PNG dimensions:

| File | Dimensions |
| --- | --- |
| `loss_curve.png` | 1200 x 750 |
| `vp_initial_final_delta.png` | 2100 x 675 |

Backend diagnostics:

| Metric | Value |
| --- | ---: |
| backend_available | true |
| backend_device | npu:0 |
| backend_dtype | float32 |
| backend_memory_allocated | 25880576 |

Key metrics:

| Metric | Value |
| --- | ---: |
| synthetic_true_forward_seconds | 6.304843237623572 |
| inversion_seconds | 318.6111433841288 |
| seconds_per_iteration | 31.861114338412882 |
| initial_loss | 10287.890625 |
| final_loss | 7882.56396484375 |
| loss_delta | -2405.32666015625 |
| loss_relative_delta | -0.2338017333029578 |
| loss_min | 7882.56396484375 |
| loss_max | 10287.890625 |
| vp_grad_norm | 0.4460679590702057 |
| vp_update_norm | 8201.2109375 |

Loss history:

```text
[
  10287.890625,
  9721.419921875,
  9284.3994140625,
  8961.8427734375,
  8733.25390625,
  8538.12890625,
  8350.2978515625,
  8176.640625,
  8023.484375,
  7882.56396484375
]
```

## Comparison With 3-Shot Baseline

| Metric | 3 shots, ckpt=10 | 5 shots, ckpt=10 | Difference |
| --- | ---: | ---: | ---: |
| seconds_per_iteration | 31.358286083862186 | 31.861114338412882 | 0.5028282545506961 |
| backend_memory_allocated | 15091200 | 25880576 | 10789376 |
| loss_relative_delta | -0.2477983357153393 | -0.2338017333029578 | 0.013996602412381504 |

Loss scales differ because the shot count changes, so absolute loss values
should not be compared directly.

## Interpretation

- The 5-shot full-case path is stable on `npu:0`.
- Loss decreases monotonically over all 10 iterations.
- Average runtime is only `0.5028282545506961s/iteration` slower than the
  3-shot checkpoint-10 baseline, while processing more shots per iteration.
- NPU memory increases from `15091200` to `25880576` allocated bytes in the
  backend diagnostics, but this run does not show a memory failure or major
  runtime cliff.
- The 5-shot setup is a better throughput stress test than 3-shot, while
  3-shot remains the slightly faster per-iteration benchmark.

## Next Step

Use 5-shot, 10-iteration, `checkpoint_segments=10` as the next realistic
full-case throughput baseline. The next useful test is a 7-shot run with the
same settings to locate the memory/runtime knee, or a longer 5-shot run if the
goal shifts from throughput probing to convergence behavior.
