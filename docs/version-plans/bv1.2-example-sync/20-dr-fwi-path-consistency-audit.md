# DR-FWI Path Consistency Audit

## Scope

This pass checked tracked DR-FWI Python scripts for:

- machine-specific absolute paths;
- old `../datasets/...` model paths;
- `SCRIPT_DIR.parents[n]` directory-depth assumptions;
- current-working-directory output roots such as `./data`;
- direct tracked `.py` / `.ipynb` pairs that need wrapper consistency checks.

It did not change inversion parameters, model dimensions, optimizer settings,
or numerical expressions.

## Findings

- No tracked DR-FWI `.py` file contains machine-specific absolute paths such as
  `/home`, `/project`, `/mnt`, or `/liufeng`.
- No tracked DR-FWI `.py` file still contains `sys.path.append`, wildcard ADFWI
  imports, forced `matplotlib.use("agg")`, or `project_path = "./data"`.
- No tracked DR-FWI `.py` file has a same-stem tracked notebook pair, so there
  is no direct script/notebook pair to synchronize in this pass.
- The FootHill reparameterization scripts still used
  `SCRIPT_DIR.parents[2] / "datasets" / "foothill_source"`, which depends on
  a fixed directory depth.

## Change

Updated 49 FootHill scripts to resolve the dataset root by searching for the
`examples` parent directory:

```python
next(parent for parent in SCRIPT_DIR.parents if parent.name == "examples") / "datasets" / "foothill_source"
```

This matches the path policy used by the rest of the synchronized DR-FWI
scripts and keeps the scripts runnable from any current working directory.

## Validation

```text
conda run -n adfwi python /tmp/adfwi_py_compile_changed.py
checked=49
failed=0
```

Residual tracked-script scans:

- `SCRIPT_DIR.parents[n]`: 0
- machine-specific absolute paths: 0
- literal `../` dataset paths: 0
- tracked `.py` / `.ipynb` same-stem pairs: 0

## Remaining Risk

Some ignored legacy backup scripts under DR-FWI still contain old wrapper
patterns. They are not Git-tracked and were not included in this commit.
If those backups need to become supported examples, they should first be
promoted into the tracked example set and then synchronized deliberately.

## Next Direction

Stop broad DR-FWI mechanical edits unless a new repeated tracked-script pattern
is found. The next useful example-sync step is a small runnable smoke test on a
representative DR-FWI script with reduced iterations or a dry-run mode.
