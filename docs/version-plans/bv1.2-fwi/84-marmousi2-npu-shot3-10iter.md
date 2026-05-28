# 84. Marmousi2 NPU 3-Shot 10-Iteration Gate

## Purpose

Validate that the faster 3-shot full-length Marmousi2 NPU behavior observed in
the 2-iteration gate persists over a medium 10-iteration inversion. This run is
the current baseline before testing larger shot counts or changing checkpoint
defaults.

## Command

```bash
rm -rf tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter10
ADFWI_RUN_FULL_CASES=1 \
ADFWI_FULL_CASE_SHOT_COUNT=3 \
ADFWI_FULL_CASE_ITERATIONS=10 \
ADFWI_FULL_CASE_CHECKPOINT_SEGMENTS=1 \
ADFWI_FULL_CASE_OUTPUT_DIR=tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter10 \
conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

## Reproducibility State

| Item | Value |
| --- | --- |
| conda environment | adfwi |
| git branch | bv1.2 |
| git baseline before run | `def80f8` |
| dirty worktree during run | `examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb` metadata change |
| output directory | `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter10` |

## Configuration

| Setting | Value |
| --- | --- |
| device | npu:0 |
| observed source | synthetic-true |
| shot_count | 3 |
| nt_samples | 3000 |
| iterations | 10 |
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
Ran 2 tests in 401.811s
OK
```

Saved local artifacts:

| File | Size |
| --- | ---: |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter10/summary.json` | 1965 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter10/loss_history.csv` | 190 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter10/loss_curve.png` | 55402 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt1_iter10/vp_initial_final_delta.png` | 348861 bytes |

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
| synthetic_true_forward_seconds | 9.180721519514918 |
| inversion_seconds | 342.9237162936479 |
| seconds_per_iteration | 34.29237162936479 |
| initial_loss | 6375.7919921875 |
| final_loss | 4795.88134765625 |
| loss_delta | -1579.91064453125 |
| loss_relative_delta | -0.2477983357153393 |
| loss_min | 4795.88134765625 |
| loss_max | 6375.7919921875 |
| vp_grad_norm | 0.3065323829650879 |
| vp_update_norm | 8384.810546875 |

Loss history:

```text
[
  6375.7919921875,
  6006.283203125,
  5718.0625,
  5498.4248046875,
  5339.8671875,
  5215.75390625,
  5099.3203125,
  4991.9140625,
  4891.92919921875,
  4795.88134765625
]
```

## Interpretation

- The 3-shot 10-iteration full-case path is stable on `npu:0` with
  `checkpoint_segments=1`.
- Loss decreases monotonically over all 10 iterations.
- Average inversion time is `34.29237162936479s/iteration`, matching the
  previous 3-shot 2-iteration gate (`34.23443403746933s/iteration`) and staying
  faster than the 1-shot 10-iteration gate (`38.60068005267531s/iteration`).
- This confirms that the 3-shot setup is a better current NPU benchmark than the
  1-shot setup for this full-length Marmousi2 path.

## Next Step

Use this 3-shot 10-iteration result as the current performance and numerical
baseline. The next useful test is either:

1. increase `shot_count` again, such as to 5 shots, to find the NPU memory and
   throughput knee; or
2. compare `checkpoint_segments=1` against a segmented setting such as 5 or 10
   at the same 3-shot, 10-iteration configuration.
