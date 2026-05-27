# 76. Marmousi2 NPU 10-Iteration Gate

## Test Path

Continue from the configurable reduced Marmousi2 iteration smoke by running the
first real-case 10-iteration gate on `npu:0`. This checks that the real model,
real observed data subset, backend transfer path, FWI loop, gradient processing,
optimizer update, cache bookkeeping, and loss-history reporting remain stable
beyond one or two iterations.

## Command

Run in the `adfwi` conda environment:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 300 --iterations 10
```

## Result

The run completed with `status: ok`.

| Field | Value |
| --- | ---: |
| device | npu:0 |
| shot_count | 1 |
| nt_samples | 300 |
| receivers | 200 |
| iterations | 10 |
| seconds | 39.64039015583694 |
| memory_allocated | 924160 |
| initial_loss | 4.9887585191754624e-06 |
| final_loss | 4.9887585191754624e-06 |
| loss_min | 4.9887585191754624e-06 |
| loss_max | 4.9887585191754624e-06 |
| vp_grad_norm | 4.9039257084357996e-14 |
| vp_update_norm | 0.4906421899795532 |

Loss history:

```text
[
  4.9887585191754624e-06,
  4.9887585191754624e-06,
  4.9887585191754624e-06,
  4.9887585191754624e-06,
  4.9887585191754624e-06,
  4.9887585191754624e-06,
  4.9887585191754624e-06,
  4.9887585191754624e-06,
  4.9887585191754624e-06,
  4.9887585191754624e-06
]
```

## Interpretation

- The 10-iteration reduced real-case path is stable on `npu:0`.
- The loss remains finite and unchanged for this tiny safe-squared-L2 setup,
  while the model update accumulates across iterations.
- The previous CPU 2-iteration run took about 109 seconds; this NPU 10-iteration
  run took about 39.6 seconds, so future multi-iteration real-case gates should
  use NPU by default.
- This is a validation run only. No code changed in this record.

## Next Step

Prepare a 100-iteration baseline as a scheduled or explicitly requested run.
Start with one shot and 300 samples on `npu:0`; if runtime is acceptable, add
JSON-output capture to a timestamped benchmark directory so the long-run result
is not only stored in the planning notes.
