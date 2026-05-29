# 71 - Example Torch Gradient Option

## Optimization Path

Continue from the acoustic TorchGradProcessor smoke by exposing the opt-in torch
native gradient processor in the minimal backend examples and smoke-suite example
runner. The goal is to let users benchmark legacy versus torch gradient
post-processing without editing scripts.

## Change

- Added --gradient-processor legacy|torch to scripts/examples/minimal_acoustic_fwi_backend.py.
- Added --gradient-processor legacy|torch to scripts/examples/minimal_elastic_fwi_backend.py.
- The default remains legacy in both examples, preserving historical behavior.
- The examples record inversion.gradient_processor in their JSON output.
- Added --example-gradient-processor legacy|torch to scripts/smoke/run_backend_smoke_suite.py so the examples suite can run either path.
- Added a focused command-construction test for the smoke suite option.
- Updated scripts/examples/README.md with gradient-processor usage guidance.

## Scientific Contract

- No propagator formula, misfit formula, optimizer behavior, transform order, or
  default gradient processor behavior changed.
- Torch gradient processing remains opt-in.
- Because this exposes a gradient post-processing alternative to user-facing FWI
  examples, numerical validation includes CPU/NPU comparison for the acoustic
  torch-gradient mini-inversion and CPU smoke runs for both minimal examples.

## Validation

Completed validation in the adfwi conda environment:

- conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problem acoustic --case baseline --devices cpu,npu:0 --rel-tol 1e-5 --abs-tol 1e-10 -- --gradient-processor torch --nx 8 --nz 6 --nabc 2 --nt 8: passed.
- conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device cpu --gradient-processor torch --nx 8 --nz 6 --nabc 2 --nt 8: printed status ok.
- conda run -n adfwi python scripts/examples/minimal_elastic_fwi_backend.py --device cpu --gradient-processor torch --nx 8 --nz 8 --nabc 2 --nt 8: printed status ok.
- conda run -n adfwi python -m unittest tests/test_backend_smoke_suite.py: 1 test passed.
- conda run -n adfwi python -m py_compile scripts/examples/minimal_acoustic_fwi_backend.py scripts/examples/minimal_elastic_fwi_backend.py scripts/smoke/acoustic_mini_inversion_smoke.py scripts/smoke/run_backend_smoke_suite.py tests/test_backend_smoke_suite.py: passed.

CPU/NPU acoustic torch-gradient drift on the small smoke case:

| Metric | CPU | NPU | Abs diff | Rel diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| loss | 1.3075439397880473e-08 | 1.3075440286058893e-08 | 8.881784197001252e-16 | 6.792722847330089e-08 | pass |
| vp_grad_norm | 3.8785075151537285e-10 | 3.8785075151537285e-10 | 0.0 | 0.0 | pass |
| vp_update_norm | 0.03873920068144798 | 0.03873920068144798 | 0.0 | 0.0 | pass |

Minimal example torch-mode CPU smoke metrics:

| Example | loss | vp_grad_norm | vp_update_norm |
| --- | ---: | ---: | ---: |
| acoustic | 1.3075439397880473e-08 | 3.8785075151537285e-10 | 0.03873920068144798 |
| elastic | 1.4761366401216947e-05 | 5.113194561090495e-07 | 0.05108551308512688 |

## Next Optimization Direction

Run the examples suite with --example-gradient-processor torch on CPU/NPU for the
standard example dimensions. If drift remains controlled, consider making the
benchmark scaffold record gradient-processor mode and compare legacy versus torch
runtime/memory for larger acoustic grids.
