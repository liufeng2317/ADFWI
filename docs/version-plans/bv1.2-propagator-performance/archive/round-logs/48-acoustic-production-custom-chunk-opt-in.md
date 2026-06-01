# Acoustic Production Custom Chunk Opt-In

Date: 2026-05-31

## Optimization Path

Main line: Phase B, acoustic AD graph and backward cost.

The previous gates showed that chunk-level custom backward has valid
observed-pressure gradient parity and that forward-only overhead is not a
blocking issue. This round adds a guarded production opt-in path while keeping
the default propagator behavior unchanged.

## What Changed

Production code:

- `ADFWI/propagator/acoustic_custom_kernels.py`
- `ADFWI/propagator/acoustic_propagator.py`
- `ADFWI/fwi/runtime/forward.py`

Benchmark code:

- `scripts/benchmark/acoustic_experimental_forward_iteration_parity.py`

Tests:

- `tests/test_backend_integration.py`

New production opt-in API:

```python
record = propagator.forward(
    checkpoint_segments=1,
    save_forward_wavefield=False,
    use_custom_chunk_backward=True,
)
```

The default remains:

```python
use_custom_chunk_backward=False
```

## Guardrails

The custom path is intentionally narrow:

| Constraint | Behavior |
| --- | --- |
| `checkpoint_segments != 1` | raise `ValueError` |
| `save_forward_wavefield=True` | raise `ValueError` |
| default call | unchanged production `forward_kernel` |

This avoids silently changing checkpoint behavior or forward-wavefield summary
contracts.

## Validation Commands

Static and targeted unit tests:

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/propagator/acoustic_custom_kernels.py \
  ADFWI/propagator/acoustic_propagator.py \
  ADFWI/fwi/runtime/forward.py \
  scripts/benchmark/acoustic_experimental_forward_iteration_parity.py

conda run -n adfwi python -m unittest \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_custom_chunk_backward_matches_default_receiver_loss \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_custom_chunk_backward_rejects_unsupported_options \
  tests.test_fwi_runtime.FWIRuntimeTests.test_acoustic_forward_batch_keeps_shot_index_with_record
```

Production opt-in fullshape observed-pressure gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --candidate-mode production-custom-chunk \
  --waveform-normalize \
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
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_production_custom_chunk_fullshape_20260531.json
```

## Results

Targeted tests passed.

Fullshape observed-pressure production opt-in result:

| Metric | Result |
| --- | ---: |
| Reference raw `vp.grad` finite | yes |
| Candidate raw `vp.grad` finite | yes |
| Receiver output max abs diff | `0.0` |
| Receiver output max rel diff | `0.0` |
| Loss abs diff | `0.0` |
| Raw `vp.grad` max abs diff | `2.8871e-7` |
| Raw `vp.grad` max rel diff | `6.5558e-2` |
| Forward speedup | `1.168x` |
| Backward speedup | `1.246x` |
| Total speedup | `1.222x` |

Timing split:

| Component | Production | Custom chunk opt-in |
| --- | ---: | ---: |
| forward | `8.5290 s` | `7.2995 s` |
| loss evaluation | `0.0586 s` | `0.0072 s` |
| backward | `17.3065 s` | `13.8870 s` |
| total | `25.8954 s` | `21.1939 s` |

## Decision

The opt-in production path passes the current fullshape observed-pressure gate
and gives a measured total speedup of `1.22x` for the tested one-iteration
workflow. It remains opt-in because:

- checkpoint segmentation is not implemented in the custom path;
- forward-wavefield summaries are not implemented in the custom path;
- NPU float32 raw-gradient differences are small but nonzero due to changed
  backward accumulation order.

## Next Direction

Do not make this the default yet.

Next step is a short real FWI validation with the opt-in path enabled at the FWI
batch runner level. The goal is to compare iteration loss trajectory and
seconds/iteration over a small number of iterations, not just a single raw
gradient gate.
