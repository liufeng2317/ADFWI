# Marmousi2 Reduced Inversion Smoke

## Purpose

The Marmousi2 case checks now cover object reconstruction and optional single-shot forward modeling. This pass adds `scripts/examples/marmousi2_acoustic_reduced_inversion.py` to exercise the real-case backward/gradient path without running the full 40-shot, 3000-sample, 300-iteration notebook inversion.

The script reads the existing Marmousi2 model and observed data, selects a small shot/time subset, runs one `AcousticFWI` iteration, and prints a JSON summary. It writes no notebooks, figures, wavefields, inversion outputs, or data files.

## Reduced Contract

Default subset:

- `shot_count=1`;
- `nt_samples=300`;
- all 200 receivers;
- `checkpoint_segments=1`;
- one `AcousticFWI` iteration;
- SGD optimizer;
- `safe-squared-l2` misfit.

The smoke checks that loss, `vp` gradient norm, and `vp` update norm are finite and positive.

## Misfit Decision

An initial test with the legacy `Misfit_waveform_L2` produced NaN `vp` gradients on both CPU and NPU, even for `nt_samples=100`. This points to the reduced-window misfit formula rather than an NPU-specific backend issue. The legacy L2 implementation uses a square root of squared residual energy; exactly zero residual samples can make the gradient singular at `sqrt(0)`.

For this smoke, the default misfit is therefore a local `safe-squared-l2` class that uses mean squared residuals and avoids the square-root singularity. The legacy L2 path remains available with `--misfit legacy-l2` for diagnostics.

## Validation Commands

```bash
python -m py_compile scripts/examples/marmousi2_acoustic_reduced_inversion.py
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 300
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device cpu --shot-count 1 --nt-samples 300
```

## Validation Results

| Device | Loss | `vp_grad_norm` | `vp_update_norm` | Seconds |
| --- | ---: | ---: | ---: | ---: |
| NPU | `4.9887585191754624e-06` | `4.905012959926895e-14` | `0.04906421899795532` | `5.012942746281624` |
| CPU | `4.9887580644281115e-06` | `4.905012621113716e-14` | `0.04906421899795532` | `61.402819622308016` |

The gradient norm is very small, so the smoke uses `lr=1e12` by default as a visibility scale for the one-step model update. This is not a recommended production inversion learning rate. It is only used to confirm that the optimizer step changes the model in float32.

## Next Step

The next refinement should decide whether to add this reduced inversion smoke to a separate opt-in runner suite. Because the CPU path takes about one minute, it should not be part of the default quick `case-checks` suite.
