# Sys Path Residual Cleanup

## Scope

This pass removed all remaining tracked `sys.path.append(...)` and
`sys.path.insert(...)` usage under `examples/`.

Tracked files touched:

```text
examples/dip/DIP-ADFWI/01_Multi-CNN/03_plot_inverted_res.ipynb
examples/dip/DIP-ADFWI/01_Multi-CNN/04_uncertainty_assesment.ipynb
examples/dip/DIP-ADFWI/02_Unet/03_plot_inverted_res.ipynb
examples/gradient_checking/Acoustic-Marmousi2/02_gradient_check_FD.py
examples/gradient_checking/Elastic-Marmousi2/02_gradient_check_FD-rho.py
examples/gradient_checking/Elastic-Marmousi2/02_gradient_check_FD-vp.py
examples/gradient_checking/Elastic-Marmousi2/02_gradient_check_FD-vs.py
examples/validation/marmousi2_acoustic_reduced/notebooks/02_inversion.ipynb
examples/validation/marmousi2_acoustic_reduced/scripts/forward_modeling.py
examples/validation/marmousi2_acoustic_reduced/scripts/inversion.py
examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py
```

## Optimization Path

- Removed notebook-local and script-local `sys.path` mutation.
- Replaced gradient-checking finite-difference script wildcard imports with
  explicit ADFWI imports matching the corresponding cleaned notebooks.
- Removed unused broad ADFWI imports from the three tracked DIP result
  notebooks; kept only the symbols they actually use.
- Removed validation script path injection while preserving script-path
  execution.
- Removed the validation inversion notebook's absolute fallback path and
  replaced it with an explicit runtime error if the notebook is not run from the
  repository root or an installed environment.

This pass did not change numerical loops, optimizer settings, devices, or
Matplotlib backend policy.

## Validation

```text
tracked sys.path.append hits: 0
tracked sys.path.insert hits: 0
Python AST parse for touched scripts: passed
notebook JSON/AST parse for touched notebooks: passed
conda env adfwi py_compile for touched scripts: passed
```

Validation command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py check \
  --device npu:0 --dtype float32 --shots 1 --checkpoint-segments 1 \
  --nx 40 --nz 30 --nt 300 --dx 40 --dz 40 --nabc 20 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/sys_path_cleanup_check
```

Result:

```text
status=ok
backend=npu:0 float32
model.vp.shape=[30, 40]
model.vp.finite=true
survey.shots=1
survey.receivers=40
propagator.device=npu:0
propagator.dtype=float32
```

The command emitted only the known local `torch_npu` Ascend toolkit owner
warnings.

## Next Direction

The next cleanup should not revisit `sys.path`; tracked example files are clean
for that marker. Remaining work should be handled by separate policies:

- review tracked notebook path residuals per file;
- decide whether to remove `matplotlib.use("agg")` from batch scripts;
- decide whether ignored backup/cmp research files should be promoted or left
  untouched.
