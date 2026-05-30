# Marmousi2 Forward Import Dedup

## Scope

This pass only cleaned the import cell in:

```text
examples/acoustic/01-model-test/01-Marmousi2/01_forward.ipynb
```

No forward modeling parameters, model definitions, survey setup, propagation
calls, saved outputs, or notebook outputs were changed.

## Problem

The notebook import cell contained two import styles at the same time:

- a small, file-specific import set;
- a broad import block copied from a larger example template.

The broad block imported many unused model, propagator, utility, view, and DIP
symbols, making the public example harder to read.

## Change

Kept only the symbols used by the forward notebook:

- `ADFWI`
- `AcousticModel`
- `AcousticPropagator`
- `Receiver`, `SeismicData`, `Source`, `Survey`
- `load_marmousi_model`, `resample_marmousi_model`, `wavelet`
- `plot_damp`

## Validation

```text
python notebook JSON/code-cell parse: passed
conda run -n adfwi python /tmp/marmousi2_forward_import_cell.py: passed
```

The full forward run was not repeated because this change only removes unused
imports and leaves all numerical cells untouched.

## Next Direction

Use this notebook as the import-cleanup reference for other forward notebooks:
an example should import only the symbols it uses, not a global template block.
