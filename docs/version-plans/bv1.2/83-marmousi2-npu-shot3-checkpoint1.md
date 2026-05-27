# 83. Marmousi2 NPU 3-Shot Checkpoint-1 Test

## Purpose

Extend the full-length synthetic-true Marmousi2 NPU gate from one shot to three
shots while keeping `checkpoint_segments=1`. This checks whether the larger
batch improves NPU utilization and whether memory remains sufficient before
changing checkpoint defaults.

## Command

```bash
rm -rf tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter2
ADFWI_RUN_FULL_CASES=1 \
ADFWI_FULL_CASE_SHOT_COUNT=3 \
ADFWI_FULL_CASE_ITERATIONS=2 \
ADFWI_FULL_CASE_CHECKPOINT_SEGMENTS=1 \
ADFWI_FULL_CASE_OUTPUT_DIR=tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter2 \
conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

## Configuration

| Setting | Value |
| --- | --- |
| device | npu:0 |
| observed source | synthetic-true |
| shot_count | 3 |
| nt_samples | 3000 |
| iterations | 2 |
| optimizer | Adam |
| lr | 10 |
| scheduler | `StepLR(step_size=200, gamma=0.75)` |
| misfit | legacy L2 |
| waveform_normalize | true |
| auto_update_rho | true |
| checkpoint_segments | 1 |

## Result

The full-case unittest passed:

```text
Ran 2 tests in 125.852s
OK
```

Saved local artifacts:

| File | Size |
| --- | ---: |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter2/summary.json` | 1793 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter2/loss_history.csv` | 53 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter2/loss_curve.png` | 54530 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter2/vp_initial_final_delta.png` | 265380 bytes |

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
| backend_memory_allocated | 15091200 |

Key metrics:

| Metric | Value |
| --- | ---: |
| synthetic_true_forward_seconds | 8.034145724028349 |
| inversion_seconds | 68.46886807493865 |
| seconds_per_iteration | 34.23443403746933 |
| initial_loss | 6375.7919921875 |
| final_loss | 6006.283203125 |
| loss_delta | -369.5087890625 |
| loss_relative_delta | -0.057954963009344274 |
| vp_grad_norm | 0.542148768901825 |
| vp_update_norm | 2295.790283203125 |

Loss history:

```text
[
  6375.7919921875,
  6006.283203125
]
```

## Interpretation

- The 3-shot full-case path is stable on `npu:0` with
  `checkpoint_segments=1`.
- The NPU memory headroom is sufficient for this case.
- Average inversion time improved from the previous 1-shot
  `checkpoint_segments=1` run, which averaged `38.60068005267531s/iteration`,
  to `34.23443403746933s/iteration`.
- The larger shot count changes the loss scale, so loss values should not be
  compared directly against the 1-shot run. The relevant signal is that the loss
  decreases over the two iterations.

## Next Step

Run a medium 3-shot gate with 10 iterations to verify that the faster
per-iteration behavior persists and that the loss remains monotonic beyond the
first two updates. If stable, use that result as the baseline before testing
larger shot counts or deeper propagator-level performance changes.
