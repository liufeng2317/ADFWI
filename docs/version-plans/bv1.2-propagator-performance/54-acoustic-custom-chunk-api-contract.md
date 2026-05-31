# 54. Acoustic Custom Chunk API Contract

Date: 2026-05-31

## Purpose

Move the validated custom chunk acoustic path from benchmark-only plumbing to a
clear expert opt-in FWI parameter, without changing the default production
behavior.

## API Contract

The public acoustic FWI entry point now accepts:

```python
fwi.forward(
    iteration=...,
    batch_size=...,
    checkpoint_segments=...,
    save_forward_wavefield=False,
    use_custom_chunk_backward=True,
)
```

Contract:

- default remains `use_custom_chunk_backward=False`;
- `use_custom_chunk_backward=True` requires `save_forward_wavefield=False`;
- it is an acoustic-only expert speed mode;
- it is not a checkpoint memory-saving replacement;
- users should reduce `batch_size` if full-batch execution exceeds device
  memory.

The same parameter is threaded through the acoustic closure path for API
consistency. The default path and elastic FWI are unchanged.

## Code Changes

| File | Change |
| --- | --- |
| `ADFWI/fwi/acoustic_fwi.py` | add public `use_custom_chunk_backward` argument to `forward` and `forward_closure`, with guard against `save_forward_wavefield=True` |
| `ADFWI/fwi/iteration/step.py` | pass the option from FWI iteration step into acoustic batch forward |
| `ADFWI/propagator/acoustic_propagator.py` | clarify docstring and guard invalid custom wavefield-output use |

## Validation

The change is a wiring/API contract change, not a numerical kernel edit.
Default behavior remains unchanged because the new argument defaults to
`False`.

Required checks:

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/fwi/acoustic_fwi.py \
  ADFWI/fwi/iteration/step.py \
  ADFWI/propagator/acoustic_propagator.py
```

Public API smoke:

```bash
conda run -n adfwi python -c '... fwi.forward(iteration=1, batch_size=3, checkpoint_segments=10, save_forward_wavefield=False, use_custom_chunk_backward=True) ...'
```

Observed behavior:

- the public `AcousticFWI.forward(...)` call completes one reduced validation
  iteration;
- the progress output reports `Loss:6.376e+03`;
- the post-step model is finite.

Invalid-contract guard:

```bash
conda run -n adfwi python -c '... fwi.forward(iteration=1, save_forward_wavefield=True, use_custom_chunk_backward=True) ...'
```

Observed error:

```text
use_custom_chunk_backward=True requires save_forward_wavefield=False
```

## Next Direction

Do not continue expanding this high-memory custom path as a default
optimization.

The next performance branch decision should be one of:

1. stop the current custom-chunk line and move to another measured bottleneck;
2. start a separate design for true rematerializing custom backward, where
   forward saves only chunk boundary states and backward recomputes chunk
   internals.
