# 57. Acoustic Rematerialized Div-Cache Memory Tradeoff

Date: 2026-05-31

## Purpose

Explain the remaining memory overhead in the rematerialized custom checkpoint
prototype and test one memory-focused change.

Previous fullshape gate:

```text
production checkpoint peak: 292.98 MiB
remat candidate peak:       810.92 MiB
ratio:                      2.77x
total speedup:              1.229x
```

The likely source was inside rematerialized backward: it recomputed the chunk,
but then cached `p/u/w` states and `div_p/div_u/div_w` for the whole chunk
before running the reverse sweep.

## Change

Keep cached `p/u/w` states, but stop caching the divergence lists.

Instead, each reverse step recomputes its local divergence terms from the saved
`p/u/w` state and uses them immediately in the one-step manual adjoint.

Files:

- `ADFWI/propagator/acoustic_custom_kernels.py`
  - add `_step_divergences_from_state`;
  - remove persistent `div_p/div_u/div_w` lists from
    `_RematerializedCustomChunkForward.backward`.

This remains experimental and is not wired into the default propagator path.

## Small Gate

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --shots 3 \
  --batch-size 3 \
  --nx 64 \
  --nz 32 \
  --nt 300 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --waveform-normalize \
  --candidate-mode experimental-remat-chunk \
  --loss-mode observed-pressure \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/remat_chunk_synthetic_nt300 \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_remat_chunk_no_div_cache_small_gate_20260531.json
```

Result:

| Metric | Value |
| --- | ---: |
| Receiver output max abs diff | 0.0 |
| Loss abs diff | 0.0 |
| Raw `vp.grad` max abs diff | 6.984919309616089e-08 |
| Production checkpoint peak allocated | 39.24 MiB |
| Remat candidate peak allocated | 39.93 MiB |
| Remat / production peak allocated | 1.018x |
| Total speedup | 1.215x |

## Fullshape Gate

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --waveform-normalize \
  --candidate-mode experimental-remat-chunk \
  --loss-mode observed-pressure \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_remat_chunk_no_div_cache_fullshape_gate_20260531.json
```

Result:

| Metric | Before no-div-cache | After no-div-cache |
| --- | ---: | ---: |
| Receiver output max abs diff | 0.0 | 0.0 |
| Loss abs diff | 0.0 | 0.0 |
| Raw `vp.grad` max abs diff | 2.738088369369507e-07 | 2.738088369369507e-07 |
| Production checkpoint peak allocated | 292.98 MiB | 292.98 MiB |
| Remat candidate peak allocated | 810.92 MiB | 504.96 MiB |
| Remat / production peak allocated | 2.77x | 1.72x |
| Backward speedup | 1.344x | 1.059x |
| Total speedup | 1.229x | 1.002x |

## Interpretation

The analysis is confirmed:

- the previous `2.77x` memory ratio was mainly caused by storing chunk-local
  divergence tensors during rematerialized backward;
- removing those persistent divergence lists substantially lowers peak memory;
- the cost is extra recomputation inside the reverse loop, which removes most
  of the speed gain on the fullshape case.

This gives a clear tradeoff:

```text
cache div terms      -> faster, peak memory 2.77x production checkpoint
recompute div terms  -> checkpoint-like memory direction, almost no speedup
```

## Decision

The rematerialized path is scientifically viable, but the current Python-level
memory/speed tradeoff is not yet strong enough for default integration.

`1.72x` peak memory may be acceptable for some hardware, but the fullshape
speedup is only `1.002x`, so this exact variant should stay experimental.

## Next Direction

Further progress needs a more targeted design, not another broad benchmark:

1. reduce the need to store full `p/u/w` state lists during backward; or
2. keep divergence caching but make chunk length tunable independently from
   `checkpoint_segments`; or
3. move the rematerialized recurrence lower than Python-level tensor loops.

The next smallest useful experiment is a chunk-length sweep for the
rematerialized path, measuring memory and speed together. That can determine
whether an intermediate chunk size gives an acceptable memory ratio with real
speedup.
