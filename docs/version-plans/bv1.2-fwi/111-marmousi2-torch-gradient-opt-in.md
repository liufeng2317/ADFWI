# 111. Marmousi2 Torch Gradient Opt-In

## Goal

Continue convergence-focused optimization by wiring the opt-in
`TorchGradProcessor` into the fixed Marmousi2 real-case workflow without
changing the default legacy gradient processor.

## Optimization Path

1. Add `--gradient-processor legacy|torch` to
   `scripts/examples/marmousi2_acoustic_reduced_inversion.py`.
2. Keep the default on `legacy`.
3. Record the selected processor in the inversion JSON summary.
4. Let `scripts/benchmark/run_marmousi2_full_case.py` pass the selected
   processor through to the inversion script.
5. Let `tests/full_cases/test_marmousi2_acoustic_full_flow.py` accept
   `ADFWI_FULL_CASE_GRADIENT_PROCESSOR`.
6. Include `gradient_processor` in `compare_full_case_outputs.py` run summaries.
7. Document the full-case and preset usage in `tests/full_cases/README.md`.

## Numerical Contract

This step does not change FWI core numerics. It only exposes the already
available `TorchGradProcessor` in the Marmousi2 full-case entry points. The
default remains `legacy`, so existing saved baselines and preset commands keep
their historical behavior unless `--gradient-processor torch` is requested.

The current Marmousi2 full-case settings use a gradient mask without gradient
normalization or illumination preconditioning. Therefore the NPU smoothing
tolerance profile from record 110 is not exercised by the default full-case
settings yet.

## Validation Result

Command-construction and syntax checks passed:

```bash
conda run -n adfwi python -m unittest tests/test_marmousi2_full_case_presets.py tests/full_cases/test_marmousi2_acoustic_full_flow.py tests/test_full_case_output_compare.py
# Ran 13 tests in 1.537s, OK (skipped=1)

conda run -n adfwi python -m unittest tests/test_marmousi2_full_case_presets.py tests/full_cases/test_marmousi2_acoustic_full_flow.py
# Ran 10 tests in 1.103s, OK (skipped=1)

conda run -n adfwi python -m py_compile scripts/examples/marmousi2_acoustic_reduced_inversion.py scripts/benchmark/run_marmousi2_full_case.py scripts/benchmark/compare_full_case_outputs.py tests/test_marmousi2_full_case_presets.py tests/full_cases/test_marmousi2_acoustic_full_flow.py tests/test_full_case_output_compare.py
# OK
```

Dry-run preset command includes the explicit torch processor:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 --dry-run --gradient-processor torch --output-dir tests/full_cases/outputs/marmousi2_torch_dry_run
# command includes: --gradient-processor torch
```

A too-short `nt_samples=300` real-case smoke was attempted for both legacy and
torch processors. Both failed with `vp_grad_norm` as `nan`, so this configuration
is not a valid gradient-processor comparison gate.

Full-length one-iteration NPU smoke passed for both processors:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 3000 --observed-source synthetic-true --iterations 1 --optimizer adam --lr 10.0 --scheduler-step-size 200 --scheduler-gamma 0.75 --misfit legacy-l2 --waveform-normalize --auto-update-rho --checkpoint-segments 10 --gradient-processor legacy --output-dir tests/full_cases/outputs/marmousi2_npu_grad_legacy_nt3000_iter1_smoke
# status: ok

conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 3000 --observed-source synthetic-true --iterations 1 --optimizer adam --lr 10.0 --scheduler-step-size 200 --scheduler-gamma 0.75 --misfit legacy-l2 --waveform-normalize --auto-update-rho --checkpoint-segments 10 --gradient-processor torch --output-dir tests/full_cases/outputs/marmousi2_npu_grad_torch_nt3000_iter1_smoke
# status: ok
```

Comparison result:

```bash
conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py tests/full_cases/outputs/marmousi2_npu_grad_legacy_nt3000_iter1_smoke tests/full_cases/outputs/marmousi2_npu_grad_torch_nt3000_iter1_smoke --labels legacy,torch
# status: ok
```

| Metric | Legacy | Torch | Abs diff |
| --- | ---: | ---: | ---: |
| gradient processor | `legacy` | `torch` | - |
| final loss | `2197.3671875` | `2197.3671875` | `0.0` |
| loss history max abs diff | - | - | `0.0` |
| vp grad norm | `0.25939640402793884` | `0.25939640402793884` | `0.0` |
| vp update norm | `1232.4454345703125` | `1232.4454345703125` | `0.0` |
| seconds / iteration | `40.05040619149804` | `40.88129739277065` | `0.8308912012726068` |

## Next Direction

Run a longer fixed 3-shot Marmousi2 opt-in comparison only after deciding
whether to keep the current full-case gradient processor settings or add a
separate smoothing/illumination stress preset. The current one-iteration gate
proves that the real-case torch processor switch is wired correctly, but it does
not test the NPU smoothing drift path from records 109-110.
