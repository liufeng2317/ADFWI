# Acoustic Segment Contract

## Focus

This round defined the numerical boundary for the fused multi-time-step route.

The question was:

```text
Can a full pressure-only run over nt=16 be split into shorter production
segments without changing receiver output or final wavefield state?
```

Production propagator code was not changed.

## Test

Script:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_segment_contract_probe.py \
  --output docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_segment_contract_probe_20260606.json
```

Configuration:

- device: `npu:0`;
- shots: `40`;
- receivers: `64`;
- model: `nx=64`, `nz=48`, `nabc=8`;
- full segment: `nt=16`;
- tested segment sizes: `segment_nt=4/8/16`;
- repeats: `3`;
- source amplitude: `1e-3`.

## Results

| segment_nt | rcv_p max diff | p max diff | u max diff | w max diff | finite | chunked/full time |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 4 | `0.0` | `0.0` | `0.0` | `0.0` | yes | `1.02x` |
| 8 | `0.0` | `0.0` | `0.0` | `0.0` | yes | `1.04x` |
| 16 | `0.0` | `0.0` | `0.0` | `0.0` | yes | `0.99x` |

## Contract

A future fused short segment must preserve:

```text
inputs:
  p, u, w at segment start
  src_v[:, local_time]
  source/receiver indices
  kappa/alpha coefficients

outputs:
  p, u, w at segment end
  rcv_p for every local time sample
```

Acceptance gate:

```text
segment_nt in {4, 8, 16}
  |
  +-- receiver output parity vs production full run
  |
  +-- final p/u/w parity vs production full run
  |
  +-- finite output/state
```

For this toy gate, production chunking is exactly equivalent to the full run.

## Decision

The fused segment route can move from feasibility discussion to implementation
prototype.

The next implementation should be opt-in and benchmark-only:

1. keep production `step_forward_pressure_only` unchanged;
2. add a prototype segment function behind a benchmark script or experimental
   operator boundary;
3. compare against this contract before any performance claim;
4. only after forward parity passes, define the backward/gradient policy.

Do not return to single-step pressure micro-kernel tuning unless a new backend
implementation route changes the execution model.
