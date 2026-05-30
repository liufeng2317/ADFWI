# Global Example Residual Scan

## Scope

This scan checked all Jupyter notebooks and Python scripts under `examples/`
for remaining wrapper/import/path issues after the forward, inversion, Ricker
misfit, and gradient-checking notebook cleanup passes.

Scanned markers:

- `sys.path.append(...)` or `sys.path.insert(...)`;
- broad `from ADFWI... import *` imports;
- forced Matplotlib backend calls such as `matplotlib.use("agg")`;
- user/machine absolute paths such as `/liufeng1afs/` or
  `/ailab/user/liufeng1/`;
- repo-root style paths such as `./examples/datasets/...`.

## Tracked Files

Tracked example files:

```text
tracked notebooks: 181
tracked python scripts: 633
```

Tracked notebook residuals:

```text
sys_path_append: 4 files
adfwi_wildcard: 3 files
matplotlib_agg: 0 files
repo_abs_path: 7 files
repo_root_relative: 26 files
```

Tracked Python script residuals:

```text
sys_path_append: 7 files
adfwi_wildcard: 4 files
matplotlib_agg: 65 files
repo_abs_path: 0 files
repo_root_relative: 0 files
```

## Main Tracked Residuals

### Notebook Import Wrappers

The remaining tracked notebook wildcard imports are concentrated in DIP result
notebooks:

```text
examples/dip/DIP-ADFWI/01_Multi-CNN/03_plot_inverted_res.ipynb
examples/dip/DIP-ADFWI/01_Multi-CNN/04_uncertainty_assesment.ipynb
examples/dip/DIP-ADFWI/02_Unet/03_plot_inverted_res.ipynb
```

One validation notebook still has a local `sys.path.insert(...)` helper:

```text
examples/validation/marmousi2_acoustic_reduced/notebooks/02_inversion.ipynb
```

This should be reviewed carefully because validation notebooks may have been
kept self-contained for direct local execution.

### Python Import Wrappers

The remaining tracked Python wildcard imports are the finite-difference
gradient-checking scripts:

```text
examples/gradient_checking/Acoustic-Marmousi2/02_gradient_check_FD.py
examples/gradient_checking/Elastic-Marmousi2/02_gradient_check_FD-rho.py
examples/gradient_checking/Elastic-Marmousi2/02_gradient_check_FD-vp.py
examples/gradient_checking/Elastic-Marmousi2/02_gradient_check_FD-vs.py
```

The validation scripts also contain controlled local `sys.path.insert(...)`
lines:

```text
examples/validation/marmousi2_acoustic_reduced/scripts/forward_modeling.py
examples/validation/marmousi2_acoustic_reduced/scripts/inversion.py
examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py
```

These are intentional script bootstrap paths unless the validation scripts are
changed to require editable package installation.

### Path Residuals

Tracked notebooks still have path-style residuals, mostly in plotting or
comparison notebooks. Important examples include:

```text
examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb
examples/gradient_checking/Acoustic-Marmousi2/02_gradient_check_FD.ipynb
examples/gradient_checking/Acoustic-Marmousi2/03_1_compare_gradient.ipynb
examples/elastic/Iso-elastic-Anomaly/03_plot_inverted_res.ipynb
examples/new_features/source_encoding/cmp_8source.ipynb
examples/validation/marmousi2_acoustic_reduced/notebooks/01_forward_modeling.ipynb
examples/validation/marmousi2_acoustic_reduced/notebooks/02_inversion.ipynb
```

Some hits may be comments, notebook outputs, or intentionally repo-root paths.
They need per-file review before editing.

### Matplotlib Backend Calls

Tracked Python scripts still contain 65 `matplotlib.use("agg")` or
`matplotlib.use("Agg")` calls. These are mainly batch inversion scripts.

This is a real compatibility concern for interactive use, but it should not be
mixed with import cleanup. Removing these calls from hundreds of scripts needs
a separate policy decision because some long-running batch scripts may rely on
headless rendering.

## All Files Including Ignored/Backup Content

When ignored, backup, and local research output files are included, the numbers
are much larger:

```text
all notebooks: 569
all python scripts: 1141

notebook sys_path_append: 70 files
notebook adfwi_wildcard: 59 files
notebook repo_abs_path: 178 files
notebook repo_root_relative: 112 files

python sys_path_append: 43 files
python adfwi_wildcard: 37 files
python matplotlib_agg: 542 files
python repo_abs_path: 2 files
```

Most of the large residual count comes from ignored `backup/`, `cmp/`, DIP, and
DR-FWI research notebooks/scripts. Those should not be bulk-edited unless they
are promoted to release targets.

## Recommended Next Steps

1. Clean the four tracked gradient-checking `.py` finite-difference scripts.
   This mirrors the notebook cleanup already completed and is a bounded task.
2. Decide whether DIP result notebooks are in scope. If yes, clean only the
   three tracked DIP result notebooks with per-file symbol analysis.
3. Review path residuals in tracked notebooks, but do not mass-rewrite
   `./examples/...` strings without checking whether they are comments,
   outputs, or intended repo-root execution paths.
4. Treat `matplotlib.use("agg")` as a separate script policy pass. Do not mix it
   into import/path cleanup.
