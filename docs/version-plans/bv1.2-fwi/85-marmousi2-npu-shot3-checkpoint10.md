# 85. Marmousi2 NPU 3-Shot Checkpoint-10 Comparison

## Purpose

Compare `checkpoint_segments=10` against the previous
`checkpoint_segments=1` baseline using the same full-length Marmousi2 NPU
configuration: 3 shots, 10 inversion iterations, synthetic-true observations,
and legacy L2 waveform loss.

This tests whether disabling checkpoint segmentation is actually beneficial
when NPU memory is sufficient.

## Command

```bash
rm -rf tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10
ADFWI_RUN_FULL_CASES=1 \
ADFWI_FULL_CASE_SHOT_COUNT=3 \
ADFWI_FULL_CASE_ITERATIONS=10 \
ADFWI_FULL_CASE_CHECKPOINT_SEGMENTS=10 \
ADFWI_FULL_CASE_OUTPUT_DIR=tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

## Reproducibility State

| Item | Value |
| --- | --- |
| conda environment | adfwi |
| git branch | bv1.2 |
| git baseline before run | `6b95a8c` |
| dirty worktree during run | `examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb` metadata change |
| output directory | `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10` |

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
| checkpoint_segments | 10 |

## Result

The full-case unittest passed:

```text
Ran 2 tests in 364.245s
OK
```

Saved local artifacts:

| File | Size |
| --- | ---: |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10/summary.json` | 1981 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10/loss_history.csv` | 202 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10/loss_curve.png` | 55402 bytes |
| `tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10/vp_initial_final_delta.png` | 348854 bytes |

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
| synthetic_true_forward_seconds | 6.3282619789242744 |
| inversion_seconds | 313.58286083862185 |
| seconds_per_iteration | 31.358286083862186 |
| initial_loss | 6375.7919921875 |
| final_loss | 4795.88134765625 |
| loss_delta | -1579.91064453125 |
| loss_relative_delta | -0.2477983357153393 |
| loss_min | 4795.88134765625 |
| loss_max | 6375.7919921875 |
| vp_grad_norm | 0.30653244256973267 |
| vp_update_norm | 8384.8095703125 |

Loss history:

```text
[
  6375.7919921875,
  6006.283203125,
  5718.06201171875,
  5498.42431640625,
  5339.8671875,
  5215.75390625,
  5099.3203125,
  4991.91357421875,
  4891.92919921875,
  4795.88134765625
]
```

## Checkpoint Comparison

Compared with the same 3-shot, 10-iteration run using
`checkpoint_segments=1`:

| Metric | checkpoint_segments=1 | checkpoint_segments=10 | Difference |
| --- | ---: | ---: | ---: |
| seconds_per_iteration | 34.29237162936479 | 31.358286083862186 | -2.934085545502601 |
| final_loss | 4795.88134765625 | 4795.88134765625 | 0.0 |
| max_abs_loss_history_diff | 0.0 | 0.00048828125 | 0.00048828125 |

## Interpretation

- `checkpoint_segments=10` is faster than `checkpoint_segments=1` for this
  current 3-shot full-case NPU benchmark.
- The loss trend remains monotonic and numerically equivalent for the purpose
  of this benchmark. The maximum loss-history drift is `4.8828125e-4`, and the
  final loss is identical at the recorded precision.
- Disabling checkpoint segmentation is not justified by the current evidence,
  even when NPU memory is sufficient.
- The current practical full-case NPU baseline should be 3 shots, 10 iterations,
  and `checkpoint_segments=10`.

## Next Step

Keep `checkpoint_segments=10` as the benchmark default and increase `shot_count`
to 5 for the next test. That will probe whether the NPU throughput keeps
improving with a larger shot batch or reaches a memory/runtime knee.
