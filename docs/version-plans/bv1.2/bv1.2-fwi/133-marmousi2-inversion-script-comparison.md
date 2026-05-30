# 133 - Marmousi2 Inversion Script Comparison

## Optimization Path

Validate that the separated inversion script reproduces the manually verified
inversion notebook when it reads the same forward-generated observed data.

## Run

The script inversion used the forward data from the prior script-forward
comparison:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/inversion.py \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/script_forward_compare \
  --device npu:0 \
  --shots 3 \
  --iterations 10 \
  --checkpoint-segments 1
```

## Comparison

Reference:

- `examples/validation/marmousi2_acoustic_reduced/outputs/minimal_notebook/inversion/`

Script output:

- `examples/validation/marmousi2_acoustic_reduced/outputs/script_forward_compare/inversion/`

Result:

- Loss history matched exactly:
  - initial loss: `6375.7919921875`;
  - final loss: `4776.75927734375`;
  - loss difference norm: `0.0`;
  - max absolute loss difference: `0.0`;
  - `np.array_equal`: `True`.
- Iteration velocity history matched exactly:
  - shape: `(10, 88, 200)`;
  - full-history relative difference: `0.0`;
  - final-model relative difference: `0.0`;
  - max absolute final-model difference: `0.0`;
  - `np.array_equal`: `True`.
- Initial-to-final update norm matched:
  - notebook: `8410.615234375`;
  - script: `8410.615234375`.
- Script runtime for 3 shots and 10 iterations on NPU was about `353.19 s`.

## Scientific Contract

- No FWI core code changed.
- Inversion numerical output from the script is bitwise identical to the
  notebook reference output for this validation case.

## Validation

Completed validation:

- `conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/inversion.py --output-root examples/validation/marmousi2_acoustic_reduced/outputs/script_forward_compare --device npu:0 --shots 3 --iterations 10 --checkpoint-segments 1`: passed.
- Script-vs-notebook `iter_loss.npz` comparison: passed with zero difference.
- Script-vs-notebook `iter_vp.npz` comparison: passed with zero difference.

## Next Direction

The validation script and notebooks are now aligned for both forward modeling
and 10-iteration inversion. Use this as the release validation baseline before
running optional 100-iteration checks.
