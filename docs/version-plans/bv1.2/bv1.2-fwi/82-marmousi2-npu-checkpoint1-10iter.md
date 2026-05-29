# 82. Marmousi2 NPU Checkpoint-1 10-Iteration Test

## Purpose

Test whether disabling time checkpoint segmentation improves the full-case
Marmousi2 NPU inversion when NPU memory is sufficient.

## Command

```bash
rm -rf tests/full_cases/outputs/marmousi2_npu_ckpt1_iter10
ADFWI_RUN_FULL_CASES=1 ADFWI_FULL_CASE_ITERATIONS=10 ADFWI_FULL_CASE_CHECKPOINT_SEGMENTS=1 ADFWI_FULL_CASE_OUTPUT_DIR=tests/full_cases/outputs/marmousi2_npu_ckpt1_iter10 conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

## Configuration

| Setting | Value |
| --- | --- |
| device | npu:0 |
| observed source | synthetic-true |
| shot_count | 1 |
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
Ran 2 tests in 436.690s
OK
```

Saved local artifacts:

| File | Size |
| --- | ---: |
| `tests/full_cases/outputs/marmousi2_npu_ckpt1_iter10/summary.json` | 1972 bytes |
| `tests/full_cases/outputs/marmousi2_npu_ckpt1_iter10/loss_history.csv` | 210 bytes |
| `tests/full_cases/outputs/marmousi2_npu_ckpt1_iter10/loss_curve.png` | 50351 bytes |
| `tests/full_cases/outputs/marmousi2_npu_ckpt1_iter10/vp_initial_final_delta.png` | 355223 bytes |

PNG dimensions:

| File | Dimensions |
| --- | --- |
| `loss_curve.png` | 1200 x 750 |
| `vp_initial_final_delta.png` | 2100 x 675 |

Key metrics:

| Metric | Value |
| --- | ---: |
| synthetic_true_forward_seconds | 8.018649771809578 |
| inversion_seconds | 386.00680052675307 |
| seconds_per_iteration | 38.60068005267531 |
| initial_loss | 2197.3671875 |
| final_loss | 1570.6259765625 |
| loss_delta | -626.7412109375 |
| loss_relative_delta | -0.28522370521540336 |
| vp_grad_norm | 0.14157885313034058 |
| vp_update_norm | 8648.435546875 |
| memory_allocated | 5467136 |

Loss history:

```text
[
  2197.3671875,
  2058.8076171875,
  1943.8106689453125,
  1854.23779296875,
  1787.810546875,
  1736.64697265625,
  1689.11328125,
  1645.5975341796875,
  1607.621337890625,
  1570.6259765625
]
```

## Interpretation

- The 10-iteration full-case path is numerically stable on `npu:0`.
- Loss decreases monotonically over all 10 iterations.
- `checkpoint_segments=1` does not materially improve per-iteration runtime for
  this 1-shot, 3000-sample full-case gate. The previous
  `checkpoint_segments=10`, 2-iteration run averaged about `38.76s/iteration`;
  this run averaged about `38.60s/iteration`.
- Keeping `checkpoint_segments` configurable remains useful for larger shot
  counts, but this test does not justify changing the full-case default from 10
  to 1 yet.

## Next Step

Run the same full-length synthetic-true case with more shots, such as
`shot_count=3`, before drawing conclusions about checkpointing and NPU memory
behavior for production-like workloads.
