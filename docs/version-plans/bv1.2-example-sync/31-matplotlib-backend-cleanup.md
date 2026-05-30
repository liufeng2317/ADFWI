# Matplotlib Backend Cleanup

## Scope

This pass removed forced Matplotlib backend selection from tracked example
Python scripts.

Before cleanup:

```text
tracked matplotlib.use("agg"/"Agg") files: 65
```

After cleanup:

```text
tracked matplotlib.use("agg"/"Agg") files: 0
```

Only tracked files under `examples/` were modified. Ignored backup, comparison,
and local research files were not promoted into version control.

## Optimization Path

- Removed `matplotlib.use("agg")` and `matplotlib.use("Agg")` calls.
- Removed bare `import matplotlib` lines when the backend call was the only use
  of the module.
- Left `matplotlib.pyplot` imports, figure generation, save paths, and numerical
  logic unchanged.

This makes scripts respect the active environment's Matplotlib backend, which
is better for notebooks, IDE runs, local display sessions, and user-managed
headless execution.

## Validation

```text
tracked matplotlib backend hits: 0
changed Python scripts: 65
conda env adfwi py_compile for all changed scripts: passed
```

Validation command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py check \
  --device npu:0 --dtype float32 --shots 1 --checkpoint-segments 1 \
  --nx 40 --nz 30 --nt 300 --dx 40 --dz 40 --nabc 20 \
  --output-root examples/validation/marmousi2_acoustic_bv12/outputs/matplotlib_backend_cleanup_check
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

## Remaining Risk

Some batch jobs may have relied on hard-coded `Agg` implicitly. After this
change, headless execution should set the backend outside the script, for
example through environment configuration or the runner.

## Next Direction

Tracked examples are now clean for:

```text
sys.path.append/sys.path.insert: 0
from ADFWI... import *: 0
matplotlib.use("agg"/"Agg"): 0
```

The next separate cleanup should review tracked notebook path residuals. Do
that per file because some hits are comments, stored outputs, or intentionally
repo-root execution paths.
