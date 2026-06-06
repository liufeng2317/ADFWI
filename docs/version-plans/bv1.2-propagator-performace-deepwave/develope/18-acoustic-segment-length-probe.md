# Acoustic Segment Length Probe

## Focus

The single-step Ascend pressure micro-kernel route is currently not strong
enough:

- scalar aligned pressure is numerically correct but slower than PyTorch at
  40 shots;
- the quick vector/DataCopy copy gate is unstable.

This round moved to the next higher-value direction: fused multi-time-step
segments.

The first question was deliberately simple:

```text
Does the current production pressure-only path show per-step overhead that
could be amortized by a fused segment?
```

Production code was not changed.

## Test

Script:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_segment_length_probe.py \
  --source-amplitude 1e-3 \
  --output docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_segment_length_probe_stable_20260606.json
```

Configuration:

- device: `npu:0`;
- shots: `40`;
- receivers: `64`;
- model: `nx=64`, `nz=48`, `nabc=8`;
- tested segment lengths: `nt=1/4/8/16/32`;
- repeats: `5`;
- production function: `step_forward_pressure_only`;
- no production propagator changes.

## Results

| nt | Median total | Median per step | Per-step speedup vs nt=1 | Finite |
| ---: | ---: | ---: | ---: | --- |
| 1 | `1.51e-03 s` | `1.51e-03 s` | `1.00x` | yes |
| 4 | `5.32e-03 s` | `1.33e-03 s` | `1.13x` | yes |
| 8 | `9.44e-03 s` | `1.18e-03 s` | `1.28x` | yes |
| 16 | `1.72e-02 s` | `1.07e-03 s` | `1.40x` | yes |
| 32 | `3.37e-02 s` | `1.05e-03 s` | `1.43x` | no |

## Interpretation

For finite short segments, the production path already shows amortization:

```text
nt=1  -> 1.51e-03 s / step
nt=16 -> 1.07e-03 s / step
```

That is about `1.4x` lower per-step cost without changing the kernel. This
suggests that time-step fusion has a real direction: the overhead around each
step is not negligible.

The `nt=32` toy case becomes non-finite even with a smaller source amplitude.
This should not be treated as a performance acceptance result. For the next
round, the segment gate should stay at `nt<=16` or use the real Marmousi2
validation parameters instead of this synthetic toy setup.

## Decision

The fused segment direction is worth exploring, but the next step must stay
bounded:

1. define a short-segment contract with `segment_nt=4/8/16`;
2. preserve receiver output parity and state parity at the segment boundary;
3. only then decide whether to implement a compiled segment or a custom
   autograd segment.

Do not continue single-step scalar pressure micro-kernel tuning.
