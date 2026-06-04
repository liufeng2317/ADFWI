# Acoustic FWI Baseline Matrix

This file records the production ADFWI baseline before Deepwave-inspired
custom-operator work starts on this branch.

## Configuration

- validation case: `full_record`
- device: `npu:0`
- dtype: `float32`
- checkpoint_segments: `5`
- generated at: `2026-06-04T16:29:54`

## Results

| Shots | Iterations | Checkpoints | Batch | Mean iter (s) | Steady mean (s) | Forward (s) | Backward (s) | Peak memory (MiB) | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 5 | 1 | 33.2269 | 32.7048 | 4.3711 | 27.9570 | 293.93 | yes |
| 1 | 10 | 5 | 1 | 32.3970 | 32.2157 | 4.1236 | 27.5518 | 293.93 | yes |
| 3 | 3 | 5 | 3 | 27.1883 | 26.5124 | 4.0604 | 22.2184 | 467.60 | yes |
| 3 | 10 | 5 | 3 | 26.4964 | 26.2884 | 3.8318 | 21.9355 | 467.60 | yes |
| 40 | 3 | 5 | 40 | 27.5063 | 26.8019 | 4.2482 | 22.3473 | 3677.09 | yes |
| 40 | 10 | 5 | 40 | 26.5976 | 26.3914 | 3.9345 | 21.9360 | 3677.09 | yes |


## Notes

- `Mean iter` averages all iterations in the run.
- `Steady mean` excludes the first iteration when more than one iteration is available.
- Peak memory uses the active PyTorch backend memory API when available.
- These runs are production baselines only; no Deepwave-inspired operator is used.
