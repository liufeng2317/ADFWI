# 07 - Model Closeout Real Case Validation

## Goal

在 `ADFWI/model` 阶段收尾后，运行真实 Marmousi2 acoustic validation case，确认模型层整理没有改变真实 forward/inversion 数值结果。

## Case

Validation workflow:

```text
examples/validation/marmousi2_acoustic_reduced
```

Command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py all \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/model_closeout_validation \
  --device npu:0 \
  --shots 3 \
  --checkpoint-segments 1 \
  --iterations 10
```

Reference output:

```text
examples/validation/marmousi2_acoustic_reduced/outputs/script_forward_compare
```

Current output:

```text
examples/validation/marmousi2_acoustic_reduced/outputs/model_closeout_validation
```

Generated outputs are ignored by git through the validation folder `.gitignore`.

## Runtime Result

- Backend: `npu:0`, `float32`
- Shots: 3
- Receivers: 200
- Time samples: 3000
- `checkpoint_segments`: 1
- Forward runtime: `7.828263320028782 s`
- Inversion runtime: `341.66268572583795 s`
- Initial loss: `6375.7919921875`
- Final loss: `4776.75927734375`
- `vp_update_norm`: `8410.6181640625`

## Numerical Comparison

Compared against `script_forward_compare`:

| Quantity | Shape | max_abs | max_rel | diff_norm |
| --- | ---: | ---: | ---: | ---: |
| `obs.data.p` | `(3, 3000, 200)` | `0.000000000e+00` | `0.000000000e+00` | `0.000000000e+00` |
| `obs.data.u` | `(3, 3000, 200)` | `0.000000000e+00` | `0.000000000e+00` | `0.000000000e+00` |
| `obs.data.w` | `(3, 3000, 200)` | `0.000000000e+00` | `0.000000000e+00` | `0.000000000e+00` |
| `obs.data.forward_wavefield_p` | `(88, 200)` | `0.000000000e+00` | `0.000000000e+00` | `0.000000000e+00` |
| `obs.data.forward_wavefield_u` | `(88, 200)` | `0.000000000e+00` | `0.000000000e+00` | `0.000000000e+00` |
| `obs.data.forward_wavefield_w` | `(88, 200)` | `0.000000000e+00` | `0.000000000e+00` | `0.000000000e+00` |
| `inversion/iter_loss` | `(10,)` | `0.000000000e+00` | `0.000000000e+00` | `0.000000000e+00` |
| `inversion/iter_vp` | `(10, 88, 200)` | `0.000000000e+00` | `0.000000000e+00` | `0.000000000e+00` |

Summary value comparison:

- Forward `record.p.norm`: `1.3137102127075195`, diff `0.000000000e+00`
- Forward `record.p.min`: `-0.04947379231452942`, diff `0.000000000e+00`
- Forward `record.p.max`: `0.09417726844549179`, diff `0.000000000e+00`
- Inversion `initial_loss`: `6375.7919921875`, diff `0.000000000e+00`
- Inversion `final_loss`: `4776.75927734375`, diff `0.000000000e+00`
- Inversion `vp_update_norm`: `8410.6181640625`, diff `0.000000000e+00`

Overall maximum absolute difference: `0.000000000e+00`.

## Conclusion

The model-layer optimization closeout is numerically stable for the real Marmousi2 acoustic validation case. Forward waveform outputs, inversion loss history, and saved inverted velocity history are bitwise identical to the existing validation reference.
