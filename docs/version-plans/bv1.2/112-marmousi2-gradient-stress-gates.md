# 112. Marmousi2 Gradient Stress Gates

## Goal

Continue convergence-focused optimization by adding real-case gradient
post-processing stress controls to the Marmousi2 full-case entry points, then
validate `TorchGradProcessor` on NPU against the legacy processor for smoothing
and illumination paths separately.

## Optimization Path

1. Extend `scripts/examples/marmousi2_acoustic_reduced_inversion.py` with:
   - `--grad-smooth`
   - `--grad-mute`
   - `--marine-or-land`
2. Record gradient processor settings in the inversion JSON summary:
   - `gradient_processor`
   - `norm_grad`
   - `forw_illumination`
   - `grad_mute`
   - `grad_smooth`
   - `grad_mute_top`
   - `marine_or_land`
3. Extend `scripts/benchmark/run_marmousi2_full_case.py` so preset dry-runs and
   benchmark runs can pass those stress settings explicitly.
4. Extend `compare_full_case_outputs.py` run summaries with the same gradient
   processor settings so saved reports show what was actually compared.
5. Run two full-length, one-iteration NPU real-case gates:
   - smoothing stress: `--grad-smooth 2`
   - illumination stress: `--forw-illumination`

## Numerical Contract

The default full-case behavior is unchanged. All new settings default to the
previous values:

| Setting | Default |
| --- | --- |
| `gradient_processor` | `legacy` |
| `grad_smooth` | `0` |
| `grad_mute` | `0` |
| `marine_or_land` | `land` |
| `norm_grad` | `False` |
| `forw_illumination` | `False` |

This step does not change FWI core numerics. It only exposes the existing
gradient processor options through the real-case workflow and records them in
the comparison metadata.

## Validation Result

Interface and syntax checks passed:

```bash
conda run -n adfwi python -m unittest tests/test_marmousi2_full_case_presets.py tests/full_cases/test_marmousi2_acoustic_full_flow.py tests/test_full_case_output_compare.py
# Ran 13 tests in 1.646s, OK (skipped=1)

conda run -n adfwi python -m py_compile scripts/examples/marmousi2_acoustic_reduced_inversion.py scripts/benchmark/run_marmousi2_full_case.py scripts/benchmark/compare_full_case_outputs.py tests/test_marmousi2_full_case_presets.py tests/full_cases/test_marmousi2_acoustic_full_flow.py tests/test_full_case_output_compare.py
# OK
```

Preset dry-run with smoothing/illumination stress options passed:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 --dry-run --gradient-processor torch --grad-smooth 2 --forw-illumination --output-dir tests/full_cases/outputs/marmousi2_torch_stress_dry_run
# status: ok
# command includes: --gradient-processor torch --grad-smooth 2 --forw-illumination
```

### Smoothing Stress

Commands:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 3000 --observed-source synthetic-true --iterations 1 --optimizer adam --lr 10.0 --scheduler-step-size 200 --scheduler-gamma 0.75 --misfit legacy-l2 --waveform-normalize --auto-update-rho --checkpoint-segments 10 --gradient-processor legacy --grad-smooth 2 --output-dir tests/full_cases/outputs/marmousi2_npu_grad_smooth_legacy_nt3000_iter1

conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 3000 --observed-source synthetic-true --iterations 1 --optimizer adam --lr 10.0 --scheduler-step-size 200 --scheduler-gamma 0.75 --misfit legacy-l2 --waveform-normalize --auto-update-rho --checkpoint-segments 10 --gradient-processor torch --grad-smooth 2 --output-dir tests/full_cases/outputs/marmousi2_npu_grad_smooth_torch_nt3000_iter1

conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py tests/full_cases/outputs/marmousi2_npu_grad_smooth_legacy_nt3000_iter1 tests/full_cases/outputs/marmousi2_npu_grad_smooth_torch_nt3000_iter1 --labels legacy_smooth,torch_smooth
# status: ok
```

Comparison:

| Metric | Legacy | Torch | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| final loss | `2197.3671875` | `2197.3671875` | `0.0` | `0.0` |
| loss history max abs diff | - | - | `0.0` | `0.0` |
| vp grad norm | `0.1928115040063858` | `0.19281473755836487` | `3.2335519790649414e-06` | `3.2335519790649414e-06` |
| vp update norm | `1236.3282470703125` | `1236.3294677734375` | `0.001220703125` | `9.873606969818654e-07` |
| seconds / iteration | `42.20517492480576` | `40.41564973257482` | `1.7895251922309399` | `0.04240061071703225` |

### Illumination Stress

Commands:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 3000 --observed-source synthetic-true --iterations 1 --optimizer adam --lr 10.0 --scheduler-step-size 200 --scheduler-gamma 0.75 --misfit legacy-l2 --waveform-normalize --auto-update-rho --checkpoint-segments 10 --gradient-processor legacy --forw-illumination --output-dir tests/full_cases/outputs/marmousi2_npu_grad_illum_legacy_nt3000_iter1

conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 3000 --observed-source synthetic-true --iterations 1 --optimizer adam --lr 10.0 --scheduler-step-size 200 --scheduler-gamma 0.75 --misfit legacy-l2 --waveform-normalize --auto-update-rho --checkpoint-segments 10 --gradient-processor torch --forw-illumination --output-dir tests/full_cases/outputs/marmousi2_npu_grad_illum_torch_nt3000_iter1

conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py tests/full_cases/outputs/marmousi2_npu_grad_illum_legacy_nt3000_iter1 tests/full_cases/outputs/marmousi2_npu_grad_illum_torch_nt3000_iter1 --labels legacy_illum,torch_illum
# status: ok
```

Comparison:

| Metric | Legacy | Torch | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: |
| final loss | `2197.3671875` | `2197.3671875` | `0.0` | `0.0` |
| loss history max abs diff | - | - | `0.0` | `0.0` |
| vp grad norm | `79241.4140625` | `79241.65625` | `0.2421875` | `3.056315471699899e-06` |
| vp update norm | `1232.8828125` | `1232.8828125` | `0.0` | `0.0` |
| seconds / iteration | `41.00909478031099` | `40.57726370729506` | `0.43183107301592827` | `0.01053012936104252` |

## Interpretation

The real Marmousi2 one-iteration NPU stress gates show that the previously
measured `TorchGradProcessor` NPU float32 drift does not produce visible loss or
model-update drift for these isolated smoothing and illumination settings. The
gradient norm differs at about `1e-6` relative scale in both real-case gates.

This is still a one-iteration gate. It validates wiring and first-step numerical
behavior, not long-horizon inversion trajectory equivalence.

## Next Direction

Use the stress settings in a short multi-iteration comparison before considering
any default-path change. A reasonable next gate is 3 shots, 10 iterations,
`checkpoint_segments=10`, comparing legacy and torch with either smoothing or
illumination enabled, but only if the runtime cost is acceptable.
