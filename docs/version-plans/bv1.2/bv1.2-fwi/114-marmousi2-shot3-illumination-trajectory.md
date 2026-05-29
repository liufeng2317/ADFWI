# 114. Marmousi2 Shot3 Illumination Trajectory

## Goal

Continue convergence-focused optimization by validating the NPU
`TorchGradProcessor` forward-illumination path on a short multi-iteration
Marmousi2 trajectory. This complements the smoothing trajectory gate in record
113.

## Optimization Path

1. Use the fixed Marmousi2 `shot3` preset:
   - `shot_count=3`
   - `nt_samples=3000`
   - `iterations=10`
   - `checkpoint_segments=10`
   - `optimizer=Adam`
   - `lr=10`
   - `misfit=legacy-l2`
   - waveform normalization enabled
2. Enable illumination stress with `--forw-illumination`.
3. Run the legacy gradient processor path.
4. Run the torch gradient processor path with the same settings.
5. Compare saved full-case summaries with `compare_full_case_outputs.py`.

## Numerical Contract

No source code changed in this step. The run compares two explicit opt-in
settings:

| Field | Legacy run | Torch run |
| --- | --- | --- |
| `gradient_processor` | `legacy` | `torch` |
| `forw_illumination` | `True` | `True` |
| `grad_smooth` | `0` | `0` |
| `norm_grad` | `False` | `False` |
| `marine_or_land` | `land` | `land` |

The default Marmousi2 full-case behavior remains unchanged.

## Validation Result

Legacy illumination trajectory:

```bash
rm -rf tests/full_cases/outputs/marmousi2_npu_shot3_grad_illum_legacy_iter10 tests/full_cases/outputs/marmousi2_npu_shot3_grad_illum_torch_iter10

conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 --gradient-processor legacy --forw-illumination --output-dir tests/full_cases/outputs/marmousi2_npu_shot3_grad_illum_legacy_iter10
# status: ok
```

Torch illumination trajectory:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 --gradient-processor torch --forw-illumination --output-dir tests/full_cases/outputs/marmousi2_npu_shot3_grad_illum_torch_iter10
# status: ok
```

Comparison:

```bash
conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py tests/full_cases/outputs/marmousi2_npu_shot3_grad_illum_legacy_iter10 tests/full_cases/outputs/marmousi2_npu_shot3_grad_illum_torch_iter10 --labels legacy_illum,torch_illum
# status: ok
```

### Summary

| Metric | Legacy | Torch | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| initial loss | `6375.7919921875` | `6375.7919921875` | `0.0` | `0.0` |
| final loss | `4807.99755859375` | `4807.99755859375` | `0.0` | `0.0` |
| loss delta | `-1567.79443359375` | `-1567.79443359375` | `0.0` | `0.0` |
| loss history max abs diff | - | - | `0.00048828125` | `9.564205082571608e-08` |
| vp grad norm | `10498.9755859375` | `10498.93359375` | `0.0419921875` | `3.9996461708364215e-06` |
| vp update norm | `8354.4775390625` | `8354.478515625` | `0.0009765625` | `1.1689089847721552e-07` |
| seconds / iteration | `32.127970695123075` | `31.741032216511666` | `0.38693847861140895` | `0.012043663830599328` |

### Loss Histories

Legacy:

```text
6375.7919921875
6006.2822265625
5718.28662109375
5499.2646484375
5341.861328125
5219.76904296875
5105.29833984375
4999.59033203125
4901.5234375
4807.99755859375
```

Torch:

```text
6375.7919921875
6006.2822265625
5718.28662109375
5499.2646484375
5341.861328125
5219.76904296875
5105.298828125
4999.59033203125
4901.5234375
4807.99755859375
```

## Interpretation

The 3-shot, 10-iteration illumination stress trajectory is stable on NPU. Both
legacy and torch paths are monotonic and reach identical final loss. The only
loss-history difference is a single `0.00048828125` step difference, about
`9.6e-8` relative scale. The final update norm differs by about `1.2e-7`
relative scale.

As with the smoothing trajectory, the runtime difference is too small and too
single-sample to treat as a performance conclusion.

## Next Direction

The two main gradient post-processing stress paths now have short trajectory
evidence. The next convergence step should be a decision point rather than more
single-feature tests:

1. keep `TorchGradProcessor` opt-in and document it as NPU-validated for
   mask-only, smoothing, and illumination Marmousi2 gates; or
2. run one combined smoothing-plus-illumination trajectory before discussing
   default-path migration.
