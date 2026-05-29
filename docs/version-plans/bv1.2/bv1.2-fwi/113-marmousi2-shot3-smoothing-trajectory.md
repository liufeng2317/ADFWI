# 113. Marmousi2 Shot3 Smoothing Trajectory

## Goal

Continue convergence-focused optimization by moving from one-iteration stress
checks to a short multi-iteration real-case trajectory comparison. This validates
whether the NPU `TorchGradProcessor` smoothing drift seen in micro-benchmarks
affects a fixed Marmousi2 inversion trajectory.

## Optimization Path

1. Use the existing fixed Marmousi2 `shot3` preset:
   - `shot_count=3`
   - `nt_samples=3000`
   - `iterations=10`
   - `checkpoint_segments=10`
   - `optimizer=Adam`
   - `lr=10`
   - `misfit=legacy-l2`
   - waveform normalization enabled
2. Enable real-case smoothing stress with `--grad-smooth 2`.
3. Run the legacy gradient processor path.
4. Run the torch gradient processor path with the same settings.
5. Compare saved full-case summaries with `compare_full_case_outputs.py`.

## Numerical Contract

No source code changed in this step. The run compares two explicit opt-in
settings:

| Field | Legacy run | Torch run |
| --- | --- | --- |
| `gradient_processor` | `legacy` | `torch` |
| `grad_smooth` | `2` | `2` |
| `forw_illumination` | `False` | `False` |
| `norm_grad` | `False` | `False` |
| `marine_or_land` | `land` | `land` |

The default Marmousi2 full-case behavior remains the legacy gradient processor
without smoothing.

## Validation Result

Legacy smoothing trajectory:

```bash
rm -rf tests/full_cases/outputs/marmousi2_npu_shot3_grad_smooth_legacy_iter10 tests/full_cases/outputs/marmousi2_npu_shot3_grad_smooth_torch_iter10

conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 --gradient-processor legacy --grad-smooth 2 --output-dir tests/full_cases/outputs/marmousi2_npu_shot3_grad_smooth_legacy_iter10
# status: ok
```

Torch smoothing trajectory:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 --gradient-processor torch --grad-smooth 2 --output-dir tests/full_cases/outputs/marmousi2_npu_shot3_grad_smooth_torch_iter10
# status: ok
```

Comparison:

```bash
conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py tests/full_cases/outputs/marmousi2_npu_shot3_grad_smooth_legacy_iter10 tests/full_cases/outputs/marmousi2_npu_shot3_grad_smooth_torch_iter10 --labels legacy_smooth,torch_smooth
# status: ok
```

### Summary

| Metric | Legacy | Torch | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| initial loss | `6375.7919921875` | `6375.7919921875` | `0.0` | `0.0` |
| final loss | `4844.1640625` | `4844.1787109375` | `0.0146484375` | `3.023925906557867e-06` |
| loss delta | `-1531.6279296875` | `-1531.61328125` | `0.0146484375` | `9.563966036443812e-06` |
| loss history max abs diff | - | - | `0.0146484375` | `3.023925906557867e-06` |
| vp grad norm | `0.2081359177827835` | `0.20834840834140778` | `0.00021249055862426758` | `0.00021249055862426758` |
| vp update norm | `8298.955078125` | `8299.0361328125` | `0.0810546875` | `9.766759199845896e-06` |
| seconds / iteration | `32.233595019578935` | `32.088499035686255` | `0.14509598389268064` | `0.004501390049870274` |

### Loss Histories

Legacy:

```text
6375.7919921875
6032.6943359375
5759.794921875
5553.2353515625
5400.484375
5277.474609375
5158.7294921875
5044.484375
4939.85791015625
4844.1640625
```

Torch:

```text
6375.7919921875
6032.68896484375
5759.7919921875
5553.240234375
5400.49609375
5277.484375
5158.73486328125
5044.4853515625
4939.85546875
4844.1787109375
```

## Interpretation

The 3-shot, 10-iteration smoothing stress trajectory is stable on NPU. Both
legacy and torch paths are monotonic and reach nearly identical final loss. The
largest loss-history difference is `0.0146484375`, or about `3.0e-6` relative to
the final loss scale. The final model-update norm differs by about `9.8e-6`
relative scale.

Runtime is effectively equivalent for this case. The torch run is slightly
faster in this single measurement, but the difference is only about `0.45%` per
iteration and should not be treated as a performance conclusion.

## Next Direction

Run the same 3-shot, 10-iteration trajectory comparison for
`--forw-illumination` before considering any broader `TorchGradProcessor`
default-path discussion. The smoothing path now has both one-step and short
trajectory evidence.
