# Acoustic FWI Baseline Matrix

This file records the production ADFWI baseline before Deepwave-inspired
custom-operator work starts on this branch.

## Configuration

- validation case: `full_record`
- device: `npu:0`
- dtype: `float32`
- checkpoint_segments: `1`
- generated at: `2026-06-04T16:04:26`

## Results

| Shots | Iterations | Checkpoints | Batch | Mean iter (s) | Steady mean (s) | Forward (s) | Backward (s) | Peak memory (MiB) | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 1 | 1 | 31.3001 | 29.6848 | 7.2320 | 23.1453 | 1417.25 | yes |
| 1 | 10 | 1 | 1 | 27.6472 | 27.2480 | 5.9406 | 20.9682 | 1417.25 | yes |
| 3 | 3 | 1 | 3 | 24.2706 | 22.7180 | 6.5554 | 16.7983 | 2132.31 | yes |
| 3 | 10 | 1 | 3 | 22.4559 | 21.9790 | 5.7287 | 15.9827 | 2132.31 | yes |
| 40 | 3 | 1 | 40 | 23.3464 | 21.7926 | 6.5830 | 15.8313 | 15608.87 | yes |
| 40 | 10 | 1 | 40 | 22.1485 | 21.6749 | 5.7327 | 15.6294 | 15608.87 | yes |


## Notes

- `Mean iter` averages all iterations in the run.
- `Steady mean` excludes the first iteration when more than one iteration is available.
- Peak memory uses the active PyTorch backend memory API when available.
- These runs are production baselines only; no Deepwave-inspired operator is used.
