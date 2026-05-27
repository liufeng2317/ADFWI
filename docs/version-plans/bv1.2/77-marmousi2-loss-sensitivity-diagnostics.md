# 77. Marmousi2 Loss Sensitivity Diagnostics

## Question

The reduced Marmousi2 10-iteration NPU gate recorded identical initial and final
loss values with the default learning rate. This needed clarification before
using the run as a real-case optimization signal.

## Finding

The loss path is responsive, but the default reduced-case update is too small to
produce a visible loss change at the recorded precision.

Default 10-iteration NPU gate:

| Field | Value |
| --- | ---: |
| lr | 1e12 |
| initial_loss | 4.9887585191754624e-06 |
| final_loss | 4.9887585191754624e-06 |
| vp_update_norm | 0.4906421899795532 |

Higher-learning-rate sensitivity check:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 300 --iterations 10 --lr 1e14
```

| Field | Value |
| --- | ---: |
| lr | 1e14 |
| initial_loss | 4.9887585191754624e-06 |
| final_loss | 4.988756700186059e-06 |
| loss_delta | -1.8189894035458565e-12 |
| relative_loss_delta | -3.6461798405289344e-07 |
| vp_update_norm | 48.52224349975586 |
| seconds | 36.13907388597727 |

Loss history for the `lr=1e14` run:

```text
[
  4.9887585191754624e-06,
  4.9887585191754624e-06,
  4.9887585191754624e-06,
  4.988757609680761e-06,
  4.988757609680761e-06,
  4.988757609680761e-06,
  4.98875715493341e-06,
  4.988756700186059e-06,
  4.988756700186059e-06,
  4.988756700186059e-06
]
```

## Change

- Added `loss_delta` and `loss_relative_delta` to
  `scripts/examples/marmousi2_acoustic_reduced_inversion.py`.
- Added a focused unit test for the loss summary calculation.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_marmousi2_reduced_inversion.py
conda run -n adfwi python -m py_compile scripts/examples/marmousi2_acoustic_reduced_inversion.py tests/test_marmousi2_reduced_inversion.py
git diff --check
```

## Next Step

For the 100-iteration baseline, record both default `lr=1e12` and a sensitivity
run such as `lr=1e14`, or choose a more informative reduced window before using
loss decrease as the primary success criterion.
