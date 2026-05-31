# Acoustic Timestep Update Microbenchmark

Date: 2026-05-31

## Purpose

This round follows the default backward operator profile:

```text
Many sliced assignments -> SliceBackward / CopySlices / copy / zero overhead.
```

Before changing the production kernel, this microbenchmark isolates one
pressure update and compares two update styles:

1. current-style cloned tensor with sliced assignment;
2. functional reconstruction with `torch.cat`.

No production propagator code was changed.

Benchmark script:

```text
scripts/benchmark/acoustic_timestep_update_microbenchmark.py
```

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_timestep_update_microbenchmark.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 5 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --topk 12 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_timestep_update_microbenchmark_20260531.json
```

## Case

| Field | Value |
| --- | --- |
| Device / dtype | `npu:0`, `float32` |
| Shots | `1` |
| Model | `nx=100`, `nz=50` |
| Boundary | `nabc=20` |
| Update | acoustic pressure update only |
| Reference | cloned tensor + sliced assignment |
| Candidate | functional reconstruction with `torch.cat` |

## Numerical Result

Across five measured pairs after one warmup pair:

| Item | Max difference |
| --- | ---: |
| loss abs diff | `0.0` |
| output max abs diff | `0.0` |
| `p.grad` max abs diff | `0.0` |
| `u.grad` max abs diff | `0.0` |
| `w.grad` max abs diff | `0.0` |

The candidate is mathematically equivalent for this isolated pressure update.

## Timing Result

Speedup is defined as:

```text
reference_time / candidate_time
```

Values below `1.0x` mean the candidate is slower.

| Component | Min | Mean | Max |
| --- | ---: | ---: | ---: |
| forward | `0.6804x` | `0.9089x` | `1.0236x` |
| backward | `0.6951x` | `0.7810x` | `0.8309x` |
| total | `0.7243x` | `0.8045x` | `0.8679x` |

The functional reconstruction candidate is slower in the stable measured runs.

## Operator Profile Observation

The functional reconstruction did not remove the dominant autograd pattern.
It still produced high counts of:

```text
aten::copy_
aclnnInplaceCopy
aclnnInplaceZero
aten::slice_backward
SliceBackward0
aten::zeros
empty_tensor
```

It also increased some counts relative to the sliced-assignment reference
because reconstruction introduces additional slicing/concatenation boundaries.

## Decision

Reject this candidate for production kernel work:

```text
Do not rewrite acoustic pressure updates using torch.cat reconstruction.
```

The microbenchmark preserved numerical results, but it did not improve the
stable NPU timing and did not remove the backward operator pattern identified
in `12-acoustic-default-backward-operator-profile.md`.

## Next Direction

Return to the Phase B method map. The likely remaining default-path options are
higher risk:

1. deeper checkpoint/rematerialization experiments for memory/time tradeoff;
2. very small custom-autograd or adjoint-state prototype for one isolated
   update, with strict gradient parity;
3. backend compile/fusion experiments only if NPU support is stable.

The next step should be selected explicitly before code changes. Do not attempt
another full kernel rewrite based only on the current slice/copy profile.
