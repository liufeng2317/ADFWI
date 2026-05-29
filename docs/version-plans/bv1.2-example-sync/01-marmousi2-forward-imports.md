# Marmousi2 Forward Import Cleanup

## Boundary

```text
Goal:
Make the original Marmousi2 forward notebook imports explicit and easier to
read.

Scope:
examples/acoustic/01-model-test/01-Marmousi2/01_forward.ipynb

Validation:
Check notebook JSON validity and execute the import/setup cell in the `adfwi`
conda environment.

Stop:
Do not change forward modeling parameters, model construction, survey setup,
propagator calls, recorded data, or plotting behavior in this round.
```

## Change

Replaced broad wildcard imports with the names used by this notebook:

- `AcousticModel`
- `AcousticPropagator`
- `Source`, `Receiver`, `Survey`, `SeismicData`
- `load_marmousi_model`, `resample_marmousi_model`, `wavelet`
- `plot_damp`

The old relative `sys.path.append("../../../../")` was replaced with a repo-root
check based on `Path.cwd()` plus the same relative fallback. Output directory
creation was kept local to the notebook and simplified with `exist_ok=True`.

## Validation Result

```text
python -m json.tool examples/acoustic/01-model-test/01-Marmousi2/01_forward.ipynb
conda run -n adfwi python -c "<execute import/setup cell>"
```

Result:

```text
missing []
project_path ./data
```

No numerical or plotting cells were changed.

## Next Direction

Run the same import cleanup pattern on
`examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb`, then run a
short forward/inversion validation comparison for the Marmousi2 example pair.
