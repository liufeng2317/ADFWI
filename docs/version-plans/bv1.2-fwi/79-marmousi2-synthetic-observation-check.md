# 79. Marmousi2 Synthetic Observation Check

## Question

The reduced Marmousi2 notebook-like checks produced NaN on the 300-sample
window. That raised a valid concern: the issue might be caused by the reduced
test data rather than by the objective function. We therefore need to test with
observed data generated from the true Marmousi2 model using the current code.

## Data Paths

The original `01_forward.ipynb` generates `data/waveform/obs_data.npz` from the
true Marmousi2 model:

- true model: `data/model/true_model.npz`;
- source geometry: 40 shots, `src_x = 2, 7, ..., 197`;
- receiver geometry: 200 receivers;
- time axis: 3000 samples, `dt = 0.003`;
- source frequency: `f0 = 5`;
- observed pressure shape: `(40, 3000, 200)`.

The reduced inversion script previously read the saved `obs_data.npz` and then
selected the first `shot_count` shots and first `nt_samples` samples.

## Change

- Added `--observed-source saved|synthetic-true` to
  `scripts/examples/marmousi2_acoustic_reduced_inversion.py`.
- `saved` preserves the historical behavior.
- `synthetic-true` builds the survey subset, loads `true_model.npz`, runs the
  current acoustic propagator, and records observed data in memory without
  writing files.
- Routed the option through `scripts/smoke/run_backend_smoke_suite.py` as
  `--case-inversion-observed-source`.

## Tests

### 300-Sample Reduced Window

Command:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 300 --observed-source synthetic-true --iterations 3 --optimizer adam --lr 10 --scheduler-step-size 200 --scheduler-gamma 0.75 --misfit legacy-l2 --waveform-normalize --auto-update-rho --checkpoint-segments 10
```

Result: failed at `loss[1]` with NaN.

Changing only the objective to `safe-squared-l2` while keeping waveform
normalization also failed at `loss[1]` with NaN. Disabling waveform normalization
avoided NaN but produced `vp_update_norm = 0.0` for Adam `lr=10`.

Interpretation: the 300-sample window is not representative of the notebook-like
inversion setting. It is too short for this normalization/update combination.

### Full-Length One-Shot Window

Command:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 3000 --observed-source synthetic-true --iterations 2 --optimizer adam --lr 10 --scheduler-step-size 200 --scheduler-gamma 0.75 --misfit legacy-l2 --waveform-normalize --auto-update-rho --checkpoint-segments 10
```

Result: passed with `status: ok`.

| Field | Value |
| --- | ---: |
| observed_source | synthetic-true |
| synthetic_true_forward_seconds | 6.287238856777549 |
| device | npu:0 |
| shot_count | 1 |
| nt_samples | 3000 |
| iterations | 2 |
| optimizer | Adam |
| lr | 10 |
| misfit | legacy-l2 |
| waveform_normalize | true |
| auto_update_rho | true |
| seconds | 76.27518152073026 |
| initial_loss | 2197.3671875 |
| final_loss | 2058.807861328125 |
| loss_delta | -138.559326171875 |
| loss_relative_delta | -0.0630569742554122 |
| vp_grad_norm | 0.2128792554140091 |
| vp_update_norm | 2300.30810546875 |

## Conclusion

- Saved observed data is not the only explanation for the reduced-window NaN.
  Current-code synthetic true observations show the same NaN on the 300-sample
  reduced window.
- The objective function is not inherently broken in the notebook-like setting:
  full-length one-shot synthetic-true data with legacy L2 and waveform
  normalization gives finite loss and clear loss decrease.
- Future real-case baselines should use full-length waveforms, even when the
  shot count is reduced.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_marmousi2_reduced_inversion.py tests/test_backend_smoke_suite.py
conda run -n adfwi python -m py_compile scripts/examples/marmousi2_acoustic_reduced_inversion.py scripts/smoke/run_backend_smoke_suite.py tests/test_marmousi2_reduced_inversion.py tests/test_backend_smoke_suite.py
git diff --check
```

## Next Step

Run a full-length NPU baseline with a small shot subset, for example
`shot_count=3-5`, `nt_samples=3000`, Adam `lr=10`, legacy L2, waveform
normalization, and synthetic-true observations. If runtime is acceptable, use
that setup for the 10/100-iteration real-case baseline.
