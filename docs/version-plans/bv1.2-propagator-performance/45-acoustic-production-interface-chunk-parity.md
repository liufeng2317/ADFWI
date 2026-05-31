# Acoustic Production-Interface Chunk Parity

Date: 2026-05-31

## Optimization Path

Main line: Phase B, acoustic AD graph and backward cost.

This round connects the chunk-level custom autograd prototype to the existing
production-interface parity harness. The goal is to test the chunk design
against the real validation wrapper shape before any production kernel edit.

## What Changed

Benchmark-only files changed:

- `scripts/benchmark/acoustic_experimental_forward.py`
- `scripts/benchmark/acoustic_experimental_forward_iteration_parity.py`

Added `experimental_chunk_forward_kernel`, which mirrors the production
`forward_kernel` benchmark contract and calls the chunk-level
`CustomChunkForward`. The existing iteration parity script now accepts:

```text
--candidate-mode experimental-chunk
```

No files under `ADFWI/propagator` were changed.

## Validation Commands

Static check:

```bash
conda run -n adfwi python -m py_compile \
  scripts/benchmark/acoustic_experimental_forward.py \
  scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  scripts/benchmark/acoustic_custom_chunk_forward.py

git diff --check
```

Reduced validation wrapper with observed-pressure loss:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --candidate-mode experimental-chunk \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --batch-size 3 \
  --shots 3 \
  --nx 64 \
  --nz 32 \
  --nt 120 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_experimental_chunk_iteration_parity_reduced_20260531.json
```

Production-vs-production sanity for the same observed-pressure configuration:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --candidate-mode production \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --batch-size 3 \
  --shots 3 \
  --nx 64 \
  --nz 32 \
  --nt 120 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_experimental_chunk_iteration_parity_prod_sanity_20260531.json
```

Reduced validation wrapper with synthetic-energy loss:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --candidate-mode experimental-chunk \
  --loss-mode synthetic-energy \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --batch-size 3 \
  --shots 3 \
  --nx 64 \
  --nz 32 \
  --nt 120 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_experimental_chunk_iteration_parity_synthetic_energy_20260531.json
```

## Results

| Gate | Ref grad finite | Candidate grad finite | Output diff | Loss diff | Raw `vp.grad` max abs diff | Forward speedup | Backward speedup | Total speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| observed-pressure, production vs chunk | no | no | `0.0` | `0.0` | `NaN` | `1.985x` | `1.489x` | `1.736x` |
| observed-pressure, production vs production | no | no | `0.0` | `0.0` | `NaN` | `1.820x` | `0.879x` | `1.176x` |
| synthetic-energy, production vs chunk | yes | yes | `0.0` | `0.0` | `3.553e-15` | `1.744x` | `1.439x` | `1.584x` |

The observed-pressure reduced configuration is not a valid gradient parity
gate in this shape because production itself produces a non-finite raw
gradient. The chunk path must not be rejected based on that run, but it also
must not be promoted based on a `NaN` gradient comparison.

The synthetic-energy production-interface gate is valid and passes: outputs and
loss are exact, raw `vp.grad` is finite in both paths, and the gradient
difference is at numerical noise level.

## Decision

The chunk-level custom backward remains the active acoustic performance route.
It has now passed:

- formula-level CPU float64 recurrence gates;
- benchmark-only NPU recurrence timing gates;
- production-interface synthetic-energy gradient parity.

It has not yet passed a finite observed-pressure validation gate. Therefore no
production kernel edit is allowed yet.

## Next Direction

Stay on Phase B and build a finite observed-pressure gate rather than returning
to low-level formula debugging. The next test should use a validation shape and
loss setup where production raw `vp.grad` is finite first, then compare the
chunk candidate against that production baseline.
