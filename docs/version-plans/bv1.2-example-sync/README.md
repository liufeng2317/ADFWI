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
- [15 FootHill Forward Import Sync](./15-foothill-forward-import-sync.md)
- [16 FootHill Inversion Script Path Sync](./16-foothill-inversion-script-path-sync.md)
- [17 DR-FWI Dataset Path Sync](./17-dr-fwi-dataset-path-sync.md)
- [18 Example Matplotlib Backend Sync](./18-example-matplotlib-backend-sync.md)
- [19 DR-FWI Local Import Dedup](./19-dr-fwi-local-import-dedup.md)
- [20 DR-FWI Path Consistency Audit](./20-dr-fwi-path-consistency-audit.md)
- [21 Marmousi2 Forward Import Dedup](./21-marmousi2-forward-import-dedup.md)
- [22 Forward Notebook Import Dedup](./22-forward-notebook-import-dedup.md)
- [23 Inversion Notebook Import Audit](./23-inversion-notebook-import-audit.md)
- [24 Marmousi2 Smoke Run](./24-marmousi2-smoke-run.md)
- [25 Ricker Misfit Notebook Import Sync](./25-ricker-misfit-notebook-import-sync.md)
- [26 Ricker Misfit Sibling Import Sync](./26-ricker-misfit-sibling-import-sync.md)
- [27 Gradient Checking Notebook Import Sync](./27-gradient-checking-notebook-import-sync.md)
- [28 Global Example Residual Scan](./28-example-global-residual-scan.md)
- [29 Sys Path Residual Cleanup](./29-sys-path-residual-cleanup.md)
- [30 ADFWI Wildcard Import Cleanup](./30-adfwi-wildcard-import-cleanup.md)
- [31 Matplotlib Backend Cleanup](./31-matplotlib-backend-cleanup.md)
- [32 Tracked Notebook Path Cleanup](./32-tracked-notebook-path-cleanup.md)
- [33 Final Global Scan And Smoke Run](./33-final-global-scan-smoke-run.md)

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
