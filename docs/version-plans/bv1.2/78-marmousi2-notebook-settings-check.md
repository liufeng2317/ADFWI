# 78. Marmousi2 Notebook Settings Check

## Question

The previous sensitivity check used `lr=1e14`, which is not a reasonable FWI
setting. The reduced real-case gate should be interpreted against the original
Marmousi2 notebook configuration.

## Notebook Reference

The active inversion cell in
`examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb` uses:

| Setting | Notebook value |
| --- | --- |
| iterations | 300 |
| optimizer | `torch.optim.Adam` |
| learning rate | 10 |
| scheduler | `StepLR(step_size=200, gamma=0.75)` |
| misfit | `Misfit_waveform_L2(dt=1)` |
| gradient processor | `GradProcessor(grad_mask=grad_mask)` |
| waveform_normalize | True |
| auto_update_rho | True |
| batch_size | 40 |
| checkpoint_segments | 10 |
| observed data | 40 shots, 3000 samples, 200 receivers |

The saved historical `data/inversion/iter_loss.npz` has 300 values and changes
substantially from about `-1228.714` to `-7986.756`. That stored output is not
directly comparable to the current reduced smoke because the reduced smoke uses
one shot and 300 samples.

## Change

- Added `--optimizer sgd|adam` to
  `scripts/examples/marmousi2_acoustic_reduced_inversion.py`.
- Added scheduler controls `--scheduler-step-size` and `--scheduler-gamma`.
- Routed these options through `scripts/smoke/run_backend_smoke_suite.py` as
  `--case-inversion-optimizer`,
  `--case-inversion-scheduler-step-size`, and
  `--case-inversion-scheduler-gamma`.
- Kept the default reduced smoke behavior unchanged.
- Relaxed loss validation from positive-only to finite-only so legacy objectives
  with non-positive values can be reported instead of rejected.

## Reduced-Subset Diagnostics

Notebook-like reduced command:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 300 --iterations 10 --optimizer adam --lr 10 --scheduler-step-size 200 --scheduler-gamma 0.75 --misfit legacy-l2 --waveform-normalize --auto-update-rho --checkpoint-segments 10
```

Result: failed at `loss[1]` with NaN. This is consistent with the known
singularity risk in the legacy waveform L2 objective on reduced windows.

Stable-loss variant using the same optimizer and lr:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 300 --iterations 10 --optimizer adam --lr 10 --scheduler-step-size 200 --scheduler-gamma 0.75 --misfit safe-squared-l2 --auto-update-rho --checkpoint-segments 10
```

Result: failed because `vp_update_norm` was `0.0`. With one shot and 300
samples, Adam `lr=10` and safe squared-L2 produce gradients below the effective
update scale for this reduced smoke.

## Interpretation

- `lr=1e14` should not be treated as a recommended real-case setting. It only
  proved that the loss path can respond when the model update is made large.
- The original notebook setting `Adam(lr=10)` belongs to the full 40-shot,
  3000-sample inversion and should not be expected to move the 1-shot,
  300-sample safe-squared-L2 smoke.
- The next meaningful real-case baseline should either run a larger subset
  closer to the notebook configuration or keep using the current reduced smoke
  only as a stability gate, not as a loss-decrease gate.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_marmousi2_reduced_inversion.py tests/test_backend_smoke_suite.py
conda run -n adfwi python -m py_compile scripts/examples/marmousi2_acoustic_reduced_inversion.py scripts/smoke/run_backend_smoke_suite.py tests/test_marmousi2_reduced_inversion.py tests/test_backend_smoke_suite.py
git diff --check
```

## Next Step

Design the next NPU baseline around full-length waveforms. A follow-up
synthetic-true observation check showed that the 300-sample window is not
representative, while 3000 samples with current-code true-model observations
does produce stable loss decrease.
