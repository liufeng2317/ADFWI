# 107. Torch Gradient Parity Convergence

## Goal

Move from broad framework cleanup toward convergence-focused optimization by
strengthening the torch-native gradient processor validation. The goal is not to
change the default FWI path yet, but to make the opt-in torch path measurable
and safe before any future performance migration.

## Optimization Path

1. Audit `GradProcessor` and `TorchGradProcessor` branch coverage.
2. Fix the legacy land taper path for current SciPy versions:
   - old code used `scipy.signal.hamming(...)`;
   - current SciPy exposes this under `scipy.signal.windows.hamming(...)`;
   - the fallback now uses `scipy.signal.windows.hamming`, then
     `scipy.signal.hamming`, then `np.hamming`.
3. Add focused legacy-vs-torch parity tests for:
   - marine smoothing after mute;
   - land mute plus smoothing;
   - forward illumination preconditioning;
   - existing norm-only and marine mute/mask paths.
4. Keep tolerances explicit:
   - strict `1e-6` for simple norm and marine mute/mask paths;
   - `rtol=1e-5`, `atol=2e-3` for smoothing/illumination paths where SciPy and
     torch convolution implementations differ at float32 rounding level.

## Numerical Contract

The legacy `GradProcessor` behavior is preserved, except that the land taper
branch now runs on current SciPy. Torch parity checks on a `6 x 6` deterministic
gradient gave these drift levels before adding tests:

| Case | Max abs diff | Max rel diff |
| --- | ---: | ---: |
| marine smoothing | `4.8828125e-04` | `1.953125e-07` |
| land mute + smoothing | `4.2933866e-04` | `1.7173546e-07` |
| forward illumination | `1.0964354e-03` | `4.3857415e-07` |

These values are far below the chosen tolerances and are consistent with
float32 rounding differences.

## Validation Result

Focused torch gradient tests passed:

```bash
conda run -n adfwi python -m unittest tests/test_torch_grad_processor.py
# Ran 7 tests in 0.340s, OK
```

Runtime/import policy regression tests passed:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_import_surface_policy.py
# Ran 26 tests in 3.466s, OK
```

Syntax checks passed:

```bash
conda run -n adfwi python -m py_compile ADFWI/propagator/gradient_process.py tests/test_torch_grad_processor.py
# OK
```

## Next Direction

Keep the optimization focused on convergence gates:

1. Run the torch gradient processor parity tests on NPU tensors where supported.
2. Add a small timing comparison for legacy vs torch gradient processing on a
   medium acoustic gradient array.
3. Only after parity and timing are recorded, consider using
   `TorchGradProcessor` in a fixed Marmousi2 3-shot baseline comparison.
