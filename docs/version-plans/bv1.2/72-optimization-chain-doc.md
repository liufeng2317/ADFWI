# 72. Optimization Chain Documentation

## Purpose

Add a top-level documentation map that summarizes the completed bv1.2
optimization chain. The goal is to help contributors quickly understand how the
backend, transform, data-contract, FWI-loop, validation, benchmark, and
torch-native opt-in work connect.

## Change

- Added `docs/optimization-chain.md` with a Mermaid chain diagram.
- Summarized the main optimization layers, code landing points, validation
  commands, and next optimization direction.
- Linked the new page from `docs/backend-usage.md` so backend users can find
  the broader optimization context.

## Numerical Impact

Documentation-only change. No FWI core, propagator, gradient, transform, loss,
or backend execution code was modified.

## Validation

```bash
rg -n "ADFWI bv1.2 Optimization Chain|flowchart TD|TorchGradProcessor|Next Optimization Direction" docs/optimization-chain.md docs/backend-usage.md
git diff --check
```

## Next Step

Use the chain map to guide the next performance pass:

1. add standard-size legacy-vs-torch gradient smoke comparisons;
2. extend benchmark output to include gradient processor runtime and memory;
3. validate torch-native smoothing and illumination branches before any default
   path migration.
