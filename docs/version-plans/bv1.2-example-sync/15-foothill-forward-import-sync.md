# 15 FootHill Forward Import Sync

## Scope

Target notebook:

```text
examples/DR-FWI/reparameterization-strategy/FootHill/00_forward.ipynb
```

This was a forward example cleanup only. No propagation settings, survey
geometry, source wavelet, model construction, or saved result names were
changed.

## Changes

- Replaced broad ADFWI imports with the symbols used by the notebook:
  `AcousticModel`, `AcousticPropagator`, `Source`, `Receiver`, `Survey`,
  `SeismicData`, `wavelet`, and `plot_damp`.
- Kept `import ADFWI` because the notebook calls `ADFWI.set_backend(...)`.
- Changed output path setup from local `./data` to the repository-relative
  case folder:

```text
./examples/DR-FWI/reparameterization-strategy/FootHill/data
```

- Changed the Foothill velocity model path to the repository-relative dataset
  folder:

```text
./examples/datasets/foothill_source
```

## Validation

- Notebook JSON validation passed.
- Notebook remains output-free.
- Residual scan found no old broad ADFWI imports or old dataset relative path.

## Next Direction

Scan the remaining DR-FWI forward notebooks for the same import pattern. Apply
the same rule file by file: import only what is used, keep backend setup
explicit, and use paths that are valid from the repository root.
