# 55. Acoustic Rematerialized Custom Checkpoint Plan

Date: 2026-05-31

## Purpose

Start the checkpoint-compatible acoustic custom backward line.

The previous custom chunk path is fast but stores chunk-internal states during
forward, so it is a high-memory speed mode. This round targets the real
`checkpoint_segments > 1` use case:

```text
forward saves only chunk boundary states
backward rematerializes chunk internals
manual backward reduces autograd graph overhead
```

## Contract

This line must preserve the checkpoint memory direction before it can be
considered for production:

- no default path changes;
- no elastic path changes;
- no `save_forward_wavefield=True` support in the prototype;
- compare against production `torch.utils.checkpoint` for
  `checkpoint_segments > 1`;
- require receiver output, loss, and raw `vp.grad` parity before timing claims.

## Implementation Step

Add an experimental `rematerialized_custom_chunk_forward_kernel` that:

1. stores only chunk input states (`p`, `u`, `w`) plus coefficients and metadata
   in forward;
2. recomputes all step states and divergence terms inside backward;
3. reuses the already validated one-step manual backward;
4. remains outside the default `forward_kernel`.

## Validation Gates

Gate 1: compile and tiny parity.

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/propagator/acoustic_custom_kernels.py \
  scripts/benchmark/acoustic_experimental_forward_iteration_parity.py
```

Gate 2: reduced NPU synthetic-energy parity.

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
  --candidate-mode experimental-remat-chunk \
  --loss-mode synthetic-energy
```

Gate 3: reduced NPU observed-pressure parity.

Use the same geometry with `--loss-mode observed-pressure` and
`--waveform-normalize`.

Gate 4: memory/timing comparison against production checkpoint.

Only run this if gates 1-3 pass.

## Stop Criteria

Stop this line immediately if:

- receiver output or loss diverges;
- raw `vp.grad` differs beyond the existing custom chunk tolerance band;
- peak memory is close to the high-memory custom chunk path;
- backward becomes slower than production checkpoint after rematerialization.

## Expected Risk

This path may be slower than production checkpoint because it rematerializes
states in Python and then runs manual backward. If so, the result is still
useful: it proves that a practical checkpoint-compatible speedup likely needs a
lower-level fused implementation rather than more Python-level rearrangement.
