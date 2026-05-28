# 70 - Acoustic Torch Gradient Smoke

## Optimization Path

Continue from the opt-in TorchGradProcessor by wiring it into the acoustic
mini-inversion smoke script. The goal is to benchmark and validate the
accelerator-native gradient post-processing path before exposing it in broader
examples.

## Change

- Added --gradient-processor legacy|torch to scripts/smoke/acoustic_mini_inversion_smoke.py.
- The default remains legacy, so existing smoke commands and historical behavior
  are unchanged.
- The torch option instantiates TorchGradProcessor with the same default smoke
  settings as the legacy GradProcessor: norm_grad=False and forw_illumination=False.
- The JSON inversion report now records the selected gradient_processor.
- Added a CPU smoke comparison test that runs the acoustic mini-inversion once
  with legacy and once with torch gradient processing, then compares loss,
  vp_grad_norm, and vp_update_norm.

## Scientific Contract

- No propagator formula, misfit formula, optimizer behavior, transform order, or
  default gradient processor behavior changed.
- The new path is opt-in through --gradient-processor torch.
- Because this touches FWI gradient post-processing, the validation compares the
  one-step acoustic FWI numerical metrics against the legacy path. The CPU smoke
  comparison requires absolute drift <= 1e-12 and relative drift <= 1e-6 for
  loss, vp_grad_norm, and vp_update_norm.

## Validation

Completed validation in the adfwi conda environment:

- conda run -n adfwi python -m unittest tests/test_acoustic_gradient_processor_smoke.py: 1 test passed.
- conda run -n adfwi python -m py_compile scripts/smoke/acoustic_mini_inversion_smoke.py tests/test_acoustic_gradient_processor_smoke.py: passed.
- conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device cpu --gradient-processor torch --nx 8 --nz 6 --nabc 2 --nt 8: printed status ok with finite loss, vp_grad_norm, and vp_update_norm.

Manual torch smoke metrics for the small CPU case:

| Metric | Value |
| --- | ---: |
| loss | 1.3075439397880473e-08 |
| vp_grad_norm | 3.8785075151537285e-10 |
| vp_update_norm | 0.03873920068144798 |

## Next Optimization Direction

Run the same gradient-processor comparison on the target NPU and add the option
to minimal acoustic/elastic backend examples once CPU/NPU drift is documented.
