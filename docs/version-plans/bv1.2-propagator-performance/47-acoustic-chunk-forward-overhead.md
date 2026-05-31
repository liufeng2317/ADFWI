# Acoustic Chunk Forward Overhead

Date: 2026-05-31

## Optimization Path

Main line: Phase B/C boundary.

The previous observed-pressure finite gate showed the chunk candidate improved
backward time but appeared slower in forward inside the full iteration timing.
This round isolates forward-only cost for production, timestep-custom, and
chunk-custom paths on the same fullshape validation geometry.

## What Changed

Added benchmark-only script:

- `scripts/benchmark/acoustic_chunk_forward_overhead.py`

No production propagator files changed.

The script compares receiver outputs and forward time for:

| Mode | Meaning |
| --- | --- |
| `production` | current `AcousticPropagator.forward` / `forward_kernel` path |
| `experimental` | benchmark-only timestep custom autograd path |
| `experimental-chunk` | benchmark-only chunk custom autograd path |

## Commands

Three-path single run:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_chunk_forward_overhead.py \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --batch-size 3 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --warmup 0 \
  --repeat 1 \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_chunk_forward_overhead_fullshape_20260531.json
```

Production-vs-chunk repeat:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_chunk_forward_overhead.py \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --batch-size 3 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --modes production,experimental-chunk \
  --warmup 0 \
  --repeat 2 \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_chunk_forward_overhead_fullshape_repeat_20260531.json
```

Static check:

```bash
conda run -n adfwi python -m py_compile scripts/benchmark/acoustic_chunk_forward_overhead.py
git diff --check
```

## Results

Single three-path run:

| Mode | Forward time | Output diff vs production | Speed vs production |
| --- | ---: | ---: | ---: |
| production | `7.8230 s` | reference | reference |
| timestep custom | `9.2536 s` | `0.0` | `0.845x` |
| chunk custom | `6.3656 s` | `0.0` | `1.229x` |

Production-vs-chunk repeat:

| Metric | Result |
| --- | ---: |
| production forward mean | `7.4932 s` |
| chunk forward mean | `6.8408 s` |
| chunk speedup mean | `1.089x` |
| chunk speedup range | `0.966x - 1.211x` |
| output max abs diff | `0.0` |
| output max rel diff | `0.0` |

## Decision

The earlier impression that chunk forward is inherently slower is not supported
by the isolated forward benchmark. The timestep-custom path is clearly slower,
but chunk-custom is comparable to or slightly faster than production in
forward-only timing, with exact receiver output parity.

The remaining integration question is no longer "is chunk forward too slow?".
It is:

```text
Can a production integration keep the observed backward gain while preserving
the exact receiver output contract and keeping checkpoint behavior explicit?
```

## Next Direction

Move from diagnostic profiling to guarded production-integration design.

The next implementation should still be conservative:

- add an explicit opt-in path, not a default replacement;
- support only `checkpoint_segments == 1` at first;
- require `save_forward_wavefield=False` unless forward wavefield summaries are
  implemented in the custom path;
- keep the existing production path as the default and fallback;
- validate with fullshape observed-pressure finite gate before any broader use.
