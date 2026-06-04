# Remat Backend Timing Decision

## Focus

This round tested whether the Python `custom_autograd_remat` backend is worth
optimizing further.

## Tests

Command template:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_operator_backend_compare.py \
  --device npu:0 --checkpoint-segments <N>
```

Valid result files:

- `acoustic_operator_backend_compare_20260604.json`
- `acoustic_operator_backend_compare_reduced_20260604.json`

## Results

| Case | Checkpoint | Reference median | Remat median | Remat speed | Remat memory | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tiny `32x24 nt=80 shot=2 rec=16` | 1 | `0.5414 s` | `0.6371 s` | `0.85x` | `1.02x` | output/grad exact |
| reduced `64x32 nt=120 shot=3 rec=64` | 1 | `0.7393 s` | `0.8718 s` | `0.85x` | `1.05x` | output/grad exact |

`checkpoint_segments=10` did not complete for `custom_autograd_remat`.
The cause is structural:

```text
custom_autograd_remat backward
  |
  +-- torch.autograd.grad(...)
       |
       +-- production forward_kernel checkpoint(..., use_reentrant=True)
            |
            +-- incompatible with autograd.grad
```

## Decision

Stop optimizing the Python remat backend as a main performance route.

Reasons:

- it is slower than the reference path in both valid tests;
- it does not reduce measured peak memory in these tests;
- it does not support the important `checkpoint_segments=10` target;
- fixing this in Python would require changing checkpoint semantics, which is
  not the intended Deepwave-inspired direction.

## Code Boundary

`custom_autograd_remat` is now explicitly limited to `checkpoint_segments=1`.
This prevents a misleading deep backward error when it is accidentally used
with production-style checkpointing.

## Next Step

Move to the compiled/backend implementation route. The next useful target is a
low-level pressure update or pressure-forward operator prototype that reduces
time-step Python/Torch graph overhead directly, rather than wrapping the same
Python kernel with rematerialization.
