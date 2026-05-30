# bv1.2 Example Sync

This folder records the work for branch `bv1.2-example-sync`.

The goal is not another framework cleanup. The goal is to make the public
examples match the bv1.2 internal structure while preserving the original
example style, figures, and user-facing workflow as much as possible.

## Documents

- [00 Example Sync Outline](./00-example-sync-outline.md)
- [Example Sync Map](./example-sync-map.md)
- [01 Marmousi2 Forward Import Cleanup](./01-marmousi2-forward-imports.md)
- [02 Editable Install For Example Imports](./02-editable-install-imports.md)
- [03 Forward Case Sync Pattern](./03-forward-case-sync-pattern.md)
- [04 Acoustic Forward Bulk Sync](./04-acoustic-forward-bulk-sync.md)
- [05 Elastic Anomaly Forward Import Cleanup](./05-elastic-anomaly-forward-imports.md)
- [06 Elastic Forward Bulk Sync](./06-elastic-forward-bulk-sync.md)
- [07 Multiscale Forward Template](./07-multiscale-forward-template.md)
- [08 Gradient Checking Forward Sync](./08-gradient-forward-sync.md)
- [09 All Forward Wrapper Sync](./09-all-forward-wrapper-sync.md)
- [10 Marmousi2 Inversion Wrapper Sync](./10-marmousi2-inversion-wrapper-sync.md)
- [11 Inversion Wrapper Sync](./11-inversion-wrapper-sync.md)
- [12 Python Script Wrapper Template](./12-python-script-wrapper-template.md)
- [13 Article Figure Notebook Sync](./13-article-figure-notebook-sync.md)
- [14 Analysis Notebook Path Sync](./14-analysis-notebook-path-sync.md)

## Branch Boundary

This branch should focus on examples only:

- update example imports and calls when they no longer match bv1.2;
- keep notebooks self-contained unless the original example already used a
  helper;
- keep scripts and notebooks parallel when both are provided;
- validate with short, reproducible forward or inversion runs;
- avoid changing core `ADFWI` modules unless an example exposes a clear bug.

The first active target is the original acoustic Marmousi2 example:

```text
examples/acoustic/01-model-test/01-Marmousi2/
```

The existing validation case remains the reference workflow:

```text
examples/validation/marmousi2_acoustic_bv12/
```
