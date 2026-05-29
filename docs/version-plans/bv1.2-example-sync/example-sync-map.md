# Example Sync Map

## Top-Level Flow

```text
branch bv1.2-example-sync
  |
  +-- reference behavior
  |     |
  |     +-- examples/validation/marmousi2_acoustic_bv12
  |           +-- forward modeling script/notebook
  |           +-- short inversion script/notebook
  |           +-- saved summaries and figures
  |
  +-- user-facing examples
        |
        +-- acoustic Marmousi2
        |     +-- preserve original notebook style
        |     +-- align imports and API calls
        |     +-- compare forward and short inversion outputs
        |
        +-- minimal acoustic / elastic examples
        |     +-- smoke-run public API entry points
        |
        +-- multiscale / regularization examples
        |     +-- validate current transform and loss setup
        |
        +-- heavy research examples
              +-- migrate only after core workflows are stable
```

## What Should Change

- Example imports that still point to removed compatibility shims.
- Example calls that assume pre-bv1.2 FWI helper layout.
- Paired scripts that no longer match their notebooks.
- Example documentation that describes outdated backend or runtime behavior.

## What Should Not Change In This Branch

- Core propagator kernels.
- FWI loss formulas, gradient processing, regularization behavior, or model
  constraints unless a reproducible example failure proves a bug.
- Notebook visual structure without a clear usability reason.
- All examples at once.

## Current Status

| Area | Status | Next action |
| --- | --- | --- |
| Validation Marmousi2 acoustic | Reference exists | Use as comparison baseline. |
| Original acoustic Marmousi2 forward | Pattern identified | Review generated outputs, then close forward sync. |
| Original acoustic Marmousi2 inversion | Pending | Apply the forward import/backend pattern before short inversion validation. |
| Minimal API examples | Pending | Handle after Marmousi2. |
| Heavy research examples | Deferred | Do not start until core examples are stable. |

## Validation Ladder

Use the lightest check that proves the example still works:

1. import/compile check for pure import updates;
2. forward modeling run for forward examples;
3. short inversion run for inversion examples;
4. saved output comparison when a reference result exists;
5. full or long iteration runs only when the example itself is being released as
   a benchmark.
