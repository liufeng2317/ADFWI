# bv1.2 Example Sync Outline

## Goal / Scope / Validation / Stop

```text
Goal:
Synchronize representative examples with the bv1.2 framework so users can run
the old public workflows without being exposed to obsolete internal structure.

Scope:
Examples, example-local scripts, example documentation, and validation records.
Core library changes are out of scope unless a real example failure identifies a
clear bug.

Validation:
For each updated workflow, run the smallest matching validation path:
forward modeling for forward examples, short inversion for inversion examples,
and numerical comparison when saved reference outputs are available.

Stop:
Stop each round after one example workflow is synchronized, validated, recorded,
committed, and pushed. Record adjacent issues as next tasks instead of expanding
the round.
```

## Strategy

The bv1.2 framework changes were mostly internal: backend ownership, FWI
iteration structure, transforms, runtime helpers, survey/model contracts, and
plot utility cleanup. Therefore the example sync should be conservative. The
best result is that old examples remain familiar while their calls match the
current implementation.

Use the validation Marmousi2 acoustic case as the reference for known-good
behavior, but do not make the old examples import from the validation case. Each
example should remain understandable on its own.

## Example Priority

| Priority | Target | Purpose | Validation |
| --- | --- | --- | --- |
| Reference | `examples/validation/marmousi2_acoustic_bv12` | Known bv1.2 reference case. | Already has forward and short inversion scripts/notebooks. |
| P0 | `examples/acoustic/01-model-test/01-Marmousi2` | Main user-facing acoustic benchmark. | Run forward notebook/script path and short inversion path. |
| P1 | Minimal acoustic and elastic usage examples | Confirm common API entry points. | Import/run smoke tests. |
| P2 | Multiscale and regularization examples | Confirm FWI transform and loss options. | Short reduced workflow. |
| P3 | DIP, DR-FWI, and heavy research cases | Larger migration only after the core examples are stable. | Case-specific validation. |

## Rules For Notebook Updates

- Keep the visual order close to the original notebook.
- Keep parameter definition, model definition, survey definition, wavelet
  definition, forward modeling, and plotting in visible cells.
- Do not turn notebooks into CLI wrappers.
- Do not hide important case setup in shared external helper files.
- When a paired script is useful, keep it similar to the notebook rather than a
  separate abstraction.

## First Bounded Task

Audit and synchronize:

```text
examples/acoustic/01-model-test/01-Marmousi2/01_forward.ipynb
examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb
```

The first pass should identify:

- obsolete imports or call paths;
- differences from the validation case that are intentional parameters;
- differences that are only caused by bv1.2 internal restructuring;
- the smallest runnable validation command or notebook execution path.

No broad migration of all examples should start before this case is closed.
