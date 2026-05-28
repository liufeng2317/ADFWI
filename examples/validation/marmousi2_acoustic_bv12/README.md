# Marmousi2 Acoustic bv1.2 Validation

This folder contains a separate validation workflow for the existing Marmousi2
acoustic example:

```text
examples/acoustic/01-model-test/01-Marmousi2
```

It does not modify the original example notebooks. The goal is to validate the
current `bv1.2` framework with reproducible script and Jupyter entry points.

## Layout

```text
scripts/       Python command-line validation entry point
  run_validation.py
notebooks/     Jupyter validation entry points
  01_forward_modeling.ipynb
  02_inversion.ipynb
outputs/       Generated validation artifacts, ignored by git
```

## Stages

| Stage | Purpose | Default behavior |
| --- | --- | --- |
| `check` | Read the case, rebuild model/survey/observed data, and verify backend setup. | No forward modeling. |
| `forward` | Run a single-shot forward check from the true model. | Saves JSON/stdout/stderr under `outputs/forward_*`. |
| `inversion10` | Run a short synthetic-true inversion. | 3 shots, 10 iterations, NPU, `checkpoint_segments=10`. |
| `inversion100` | Run a longer sanity inversion. | 3 shots, 100 iterations, NPU, `checkpoint_segments=10`. |
| `all` | Run `check`, `forward`, and `inversion10`. | Does not run `inversion100` by default. |

## Python Usage

Preview commands without running:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py all --dry-run
```

Run the staged validation:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py check --overwrite
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py forward --overwrite
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py inversion10 --overwrite
```

Run the longer check only when needed:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py inversion100 --overwrite
```

## Jupyter Usage

Forward modeling and inversion are separated:

```text
notebooks/01_forward_modeling.ipynb
notebooks/02_inversion.ipynb
```

Run the forward-modeling notebook first. The notebooks intentionally follow the
original Marmousi2 example order and only change the validation-specific
parameters such as output folder, device, shot count, and iteration count. They
do not call the command-line runner.

The notebooks explicitly show:

- parameter definitions;
- true or initial model construction from the Marmousi2 dataset;
- source/receiver observation-system construction;
- source wavelet construction;
- propagator, observed-data, FWI, and visualization cells.

Notebook case definitions are written directly in each notebook. They do not
import setup helpers from another example, backend-check script, separate
case-definition helper, or command-line wrapper.

## Outputs

Default output root:

```text
examples/validation/marmousi2_acoustic_bv12/outputs/
```

Each stage writes:

- `command.json`
- `stdout.txt`
- `stderr.txt`
- `summary.json` when the wrapped command prints JSON

Inversion stages also write the artifacts produced by
`scripts/examples/marmousi2_acoustic_reduced_inversion.py`, including loss CSV
and PNG summaries.
