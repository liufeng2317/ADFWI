# 76 - Ascend Vectorized Copy Gate

## Goal

Test the next AscendC custom-op feasibility step without touching production
propagator code.

This gate checks whether a standard AscendC vectorized `DataCopy` copy kernel
can pass multi-block correctness before any pressure stencil logic is moved into
that path.

## Change

Two standalone benchmark scripts were extended:

- `scripts/benchmark/ascend_fused_pressure_update_prototype.py`
  - added `--kernel-mode copy_vector`,
  - generated a vectorized AscendC copy kernel using `TPipe`, `TQue`, local
    tensor allocation, and `DataCopy`.
- `scripts/benchmark/ascend_pressure_update_wrapper_probe.py`
  - accepts `--runtime-shape`,
  - treats all `copy*` kernel modes as exact-copy contracts.

No production files under `ADFWI/propagator` were modified.

## Validation

Syntax:

```bash
python -m py_compile \
  scripts/benchmark/ascend_fused_pressure_update_prototype.py \
  scripts/benchmark/ascend_pressure_update_wrapper_probe.py
```

Correctness gate:

```bash
conda run -n adfwi python scripts/benchmark/ascend_pressure_update_wrapper_probe.py \
  --kernel-mode copy_vector \
  --block-dim 8 \
  --runtime-shape 2 16 16 \
  --compile-timeout 600 \
  --wrapper-build-timeout 600 \
  --runtime-timeout 240 \
  --output docs/version-plans/bv1.2-propagator-performance/ascend_pressure_update_copy_vector_block8_aligned_20260601.json
```

Result:

```text
custom-op compile: ok
install: ok
wrapper build: ok
runtime: ok
numerical status: mismatch
max_abs_diff: 3.654768943786621
allclose(0, 0): false
allclose(1e-6, 1e-6): false
```

The aligned runtime shape avoids the obvious unaligned-tail explanation, but the
multi-block vector copy still does not preserve exact copy semantics.

## Decision

Do not promote the AscendC vectorized path to pressure update or FWI.

The current evidence is:

- `copy + block_dim=1`: exact,
- `pressure + block_dim=1`: float32-level parity,
- `copy + block_dim=8`: mismatch,
- `copy_vector + block_dim=8`: mismatch even for aligned shape.

Therefore the unresolved issue is at the multi-block launch/partition contract
or generated custom-op integration layer, not at pressure math itself.

## Next Direction

Pause AscendC pressure-kernel implementation unless the block launch contract is
debugged with a focused custom-op test outside ADFWI.

For ADFWI performance work, return to production-safe PyTorch acoustic hot-path
optimization where every change can be checked against:

1. forward waveform parity,
2. loss parity,
3. raw `vp` gradient parity,
4. short real-FWI iteration timing.

