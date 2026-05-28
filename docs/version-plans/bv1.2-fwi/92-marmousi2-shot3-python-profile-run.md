# 92. Marmousi2 Shot3 Python Profile Run

## Optimization Path

Run the fixed `shot3` full-case preset under Python `cProfile` and compare the
profiled output against the saved `shot3` baseline:

```text
shot3 preset -> cProfile -> full-case output -> baseline compare -> pstats analysis
```

This checks whether the next performance work should target Python-side FWI
bookkeeping or deeper propagator/autograd/NPU paths.

## Command

```bash
rm -rf tests/full_cases/outputs/marmousi2_profiled_shot3_latest
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 \
  --overwrite \
  --profile \
  --output-dir tests/full_cases/outputs/marmousi2_profiled_shot3_latest \
  --compare-to tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  --compare-labels baseline,profiled
```

## Result

The profiled full-case run completed successfully and wrote:

| File | Size |
| --- | ---: |
| `summary.json` | 2010 bytes |
| `loss_history.csv` | 202 bytes |
| `loss_curve.png` | 55402 bytes |
| `vp_initial_final_delta.png` | 348854 bytes |
| `python_profile.prof` | 4382855 bytes |

PNG dimensions:

| File | Dimensions |
| --- | --- |
| `loss_curve.png` | 1200 x 750 |
| `vp_initial_final_delta.png` | 2100 x 675 |

## Baseline Comparison

| Metric | Baseline shot3 | Profiled shot3 | Difference |
| --- | ---: | ---: | ---: |
| inversion_seconds | 313.58286083862185 | 332.0979613419622 | 18.515100503340363 |
| seconds_per_iteration | 31.358286083862186 | 33.209796134196225 | 1.8515100503340385 |
| final_loss | 4795.88134765625 | 4795.88134765625 | 0.0 |
| loss_history_max_abs_diff | 0.0 | 0.0 | 0.0 |
| vp_grad_norm | 0.30653244256973267 | 0.30653244256973267 | 0.0 |
| vp_update_norm | 8384.8095703125 | 8384.8095703125 | 0.0 |

The cProfile wrapper adds about `5.575%` overhead relative to the saved
unprofiled inversion time. The numerical trajectory is identical to the saved
baseline at the recorded precision.

## Profile Summary

The profile recorded:

```text
11442893 function calls (11145876 primitive calls) in 377.985 seconds
```

Top cumulative-time entries:

| Function | Calls | Cumulative seconds |
| --- | ---: | ---: |
| `marmousi2_acoustic_reduced_inversion.py:237(run_smoke)` | 1 | 352.589 |
| `acoustic_fwi.py:305(forward)` | 1 | 332.098 |
| `loss.py:85(apply_acoustic_batch_loss_step)` | 10 | 331.330 |
| `loss.py:59(apply_batch_loss_step)` | 10 | 277.171 |
| `torch._C._EngineBase.run_backward` | 10 | 277.157 |
| `acoustic_propagator.py:125(forward)` | 11 | 60.676 |
| `acoustic_kernels.py:194(forward_kernel)` | 11 | 60.636 |
| `checkpoint.py:224(forward)` | 110 | 60.385 |

Top internal-time entries:

| Function | Calls | Internal seconds |
| --- | ---: | ---: |
| `torch._C._EngineBase.run_backward` | 10 | 277.157 |
| `checkpoint.py:224(forward)` | 110 | 60.258 |
| `io.open_code` | 7093 | 7.002 |
| `torch_npu._C._npu_init` | 1 | 4.461 |
| `builtins.compile` | 2255 | 4.069 |

## Interpretation

- Python-side FWI loop bookkeeping is not the dominant cost in the fixed
  `shot3` run.
- Most measured time sits inside PyTorch autograd backward and checkpointed
  forward execution.
- `cProfile` attributes NPU and C++ extension blocking time to Python call
  sites, so this result is enough to rule out large Python bookkeeping overhead
  but not enough to identify individual NPU kernels.
- The next performance optimization should use a torch/NPU operator-level
  profile around forward/backward, not another broad Python profile.

## Numerical Precision

No code changed in this record. The profiled run exactly matched the saved
baseline for final loss, loss history, gradient norm, and model update norm at
the recorded precision.

## Next Direction

Add or run an operator-level profile for the fixed `shot3` preset, scoped around
`AcousticPropagator.forward` and `loss.backward()`. Use that result to decide
whether the next code change should target checkpointing, propagator kernels, or
gradient-processing migration.
