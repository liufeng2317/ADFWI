# Acoustic Checkpoint Sweep Results

This table compares ADFWI production acoustic FWI with `checkpoint_segments=10`, `5`, and `1` on the full-record Marmousi2 validation geometry. The baseline for speedup and memory ratio is `checkpoint_segments=10` for the same shot/iteration case.

| Shots | Iterations | Checkpoints | Steady sec/iter | Peak memory MiB | Speedup vs ckpt=10 | Memory ratio vs ckpt=10 | Finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 3 | 10 | 30.3681 | 159.07 | 1.000x | 1.00x | True |
| 1 | 3 | 5 | 32.7048 | 293.93 | 0.929x | 1.85x | True |
| 1 | 3 | 1 | 29.6848 | 1417.25 | 1.023x | 8.91x | True |
| 1 | 10 | 10 | 30.1068 | 159.07 | 1.000x | 1.00x | True |
| 1 | 10 | 5 | 32.2157 | 293.93 | 0.935x | 1.85x | True |
| 1 | 10 | 1 | 27.2480 | 1417.25 | 1.105x | 8.91x | True |
| 3 | 3 | 10 | 25.3252 | 268.02 | 1.000x | 1.00x | True |
| 3 | 3 | 5 | 26.5124 | 467.60 | 0.955x | 1.74x | True |
| 3 | 3 | 1 | 22.7180 | 2132.31 | 1.115x | 7.96x | True |
| 3 | 10 | 10 | 25.1091 | 268.02 | 1.000x | 1.00x | True |
| 3 | 10 | 5 | 26.2884 | 467.60 | 0.955x | 1.74x | True |
| 3 | 10 | 1 | 21.9790 | 2132.31 | 1.142x | 7.96x | True |
| 40 | 3 | 10 | 25.1312 | 2290.02 | 1.000x | 1.00x | True |
| 40 | 3 | 5 | 26.8019 | 3677.09 | 0.938x | 1.61x | True |
| 40 | 3 | 1 | 21.7926 | 15608.87 | 1.153x | 6.82x | True |
| 40 | 10 | 10 | 24.7196 | 2290.02 | 1.000x | 1.00x | True |
| 40 | 10 | 5 | 26.3914 | 3677.09 | 0.937x | 1.61x | True |
| 40 | 10 | 1 | 21.6749 | 15608.87 | 1.140x | 6.82x | True |

## Interpretation

- `checkpoint_segments=1` is the speed upper-bound reference in this matrix. It improves steady iteration time by about `1.02x-1.14x`, but raises peak memory by about `6.82x-8.91x`.
- `checkpoint_segments=5` is not a useful default for this case: it is slower than `checkpoint_segments=10` and uses about `1.61x-1.85x` more peak memory.
- The next Deepwave-inspired target should not simply disable checkpointing. The useful target is ckpt=1-like speed while preserving ckpt=10-like memory, or at least avoiding the 5x-9x memory increase seen at ckpt=1.

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_deepwave_baseline_matrix.py --checkpoint-segments 1 --output-dir docs/version-plans/bv1.2-propagator-performace-deepwave/baselines/checkpoint1
conda run -n adfwi python scripts/benchmark/acoustic_deepwave_baseline_matrix.py --checkpoint-segments 5 --output-dir docs/version-plans/bv1.2-propagator-performace-deepwave/baselines/checkpoint5
```
