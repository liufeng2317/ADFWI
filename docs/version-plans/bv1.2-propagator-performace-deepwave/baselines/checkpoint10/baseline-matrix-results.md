# Acoustic FWI Baseline Matrix

This file records the production ADFWI baseline before Deepwave-inspired
custom-operator work starts on this branch.

## Configuration

- validation case: `full_record`
- device: `npu:0`
- dtype: `float32`
- checkpoint_segments: `10`
- generated at: `2026-06-04T15:22:36`

## Results

| Shots | Iterations | Checkpoints | Batch | Mean iter (s) | Steady mean (s) | Forward (s) | Backward (s) | Peak memory (MiB) | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 10 | 1 | 30.7032 | 30.3681 | 3.7933 | 26.0122 | 159.07 | yes |
| 1 | 10 | 10 | 1 | 30.2574 | 30.1068 | 3.9762 | 25.5620 | 159.07 | yes |
| 3 | 3 | 10 | 3 | 25.5584 | 25.3252 | 3.7287 | 20.9394 | 268.02 | yes |
| 3 | 10 | 10 | 3 | 25.2591 | 25.1091 | 3.7117 | 20.8215 | 268.02 | yes |
| 40 | 3 | 10 | 40 | 25.4532 | 25.1312 | 3.7155 | 20.7941 | 2290.02 | yes |
| 40 | 10 | 10 | 40 | 24.8899 | 24.7196 | 3.6071 | 20.5572 | 2290.02 | yes |


## Notes

- `Mean iter` averages all iterations in the run.
- `Steady mean` excludes the first iteration when more than one iteration is available.
- Peak memory uses the active PyTorch backend memory API when available.
- These runs are production baselines only; no Deepwave-inspired operator is used.
