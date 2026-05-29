# Elastic Forward Bulk Sync

## Boundary

```text
Goal:
Use the cleaned Iso-elastic-Anomaly forward notebook as the template for the
remaining elastic forward examples.

Scope:
Elastic forward notebooks under examples/elastic, including isotropic elastic,
VTI elastic, and the acoustic-in-elastic layer forward variant.

Validation:
Notebook JSON parse, code-cell compile, import/setup cell execution in the
adfwi conda environment, and residual old-pattern scan.

Stop:
Do not run heavy forward simulations, change model physics, change source or
receiver geometry, or alter plotting intent.
```

## Applied Pattern

- Removed notebook-local `sys.path.append(...)`.
- Replaced broad `from ADFWI... import *` imports with explicit symbols used by
  each notebook.
- Added `import ADFWI` and `backend = ADFWI.set_backend(device, dtype=dtype)`.
- Standardized the device comment to `CPU/GPU/NPU`.
- Removed redundant `device=device` and `dtype=dtype` from model construction.
- Removed redundant `device=device` from propagator construction.
- Replaced repeated output-directory creation with an `exist_ok=True` loop.
- Cleared notebook outputs and execution counts.

## Files

The pass covered 18 elastic forward notebooks:

```text
examples/elastic/Iso-elastic-Marmousi2-shotTop-recBottom/01_forward.ipynb
examples/elastic/Iso-elastic-Anomaly/02_forward-shear.ipynb
examples/elastic/Iso-elastic-Anomaly/01_forward-explosion.ipynb
examples/elastic/Iso-elastic-Layer/02_forward-shear.ipynb
examples/elastic/Iso-elastic-Layer/01_forward-explosion.ipynb
examples/elastic/Iso-elastic-Layer/00_forward-explosion-acoustic.ipynb
examples/elastic/Iso-elastic-Marmousi2-noWater-varyingRho/01_forward-explosion-vp_vs_rho.ipynb
examples/elastic/Iso-elastic-Marmousi2-shotTop-recTop-highResolution/01_forward.ipynb
examples/elastic/Iso-elastic-Marmousi2-shotTop-recTop/01_forward.ipynb
examples/elastic/VTI-elastic-Anomaly-eps/01_forward.ipynb
examples/elastic/VTI-acoustic-Hess/01_forward_delta_epsilon.ipynb
examples/elastic/VTI-elastic-Anomaly-eps-delta-2/01_forward.ipynb
examples/elastic/VTI-elastic-Anomaly-eps-delta/01_forward.ipynb
examples/elastic/Iso-elastic-Marmousi2-shotWater-recWater/01_forward.ipynb
examples/elastic/VTI-acoustic-Hess/02_forward_vp_rho_delta_epsilon.ipynb
examples/elastic/VTI-acoustic-Hess/03_forward_vp_rho.ipynb
examples/elastic/Iso-elastic-Marmousi2-noWater-constantRho/shear-or-explosion/01_forward-explosion-vp_vs.ipynb
examples/elastic/Iso-elastic-Marmousi2-noWater-constantRho/shear-or-explosion/02_forward-shear-vp_vs.ipynb
```

The ignored local forward notebooks were intentionally promoted into version
control in this pass. Only the listed notebook files were force-added; generated
data, logs, scripts, and result directories remain ignored.

## Validation Result

```text
json_ok 18
compile_bad []
output_bad_count 0
checked 18
fail []
residual old-pattern scan: clean
```

## Remaining Risk

This was a wrapper/API synchronization pass. It did not execute full forward
simulations, so it proves import/setup validity and constructor compatibility,
not full numerical output equivalence.

## Next Direction

Use the same pattern for elastic inversion notebooks only after selecting a
small representative case and defining a short validation run.
