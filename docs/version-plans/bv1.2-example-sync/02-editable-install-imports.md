# Editable Install For Example Imports

## Boundary

```text
Goal:
Allow examples to import `ADFWI` as an installed package instead of modifying
`sys.path` inside notebooks.

Scope:
Root packaging metadata and the Marmousi2 forward notebook import cell.

Validation:
Install the package in editable mode in the `adfwi` environment without
dependency changes or build isolation downloads, then import `ADFWI` from
outside the repository.

Stop:
Do not change package metadata semantics beyond the minimal build-system entry.
Do not change notebook numerical cells.
```

## Change

Added a minimal `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"
```

The existing `setup.py` remains the source of package metadata and dependency
definitions. This keeps the change small while enabling modern editable install
behavior:

```bash
conda activate adfwi
python -m pip install -e . --no-deps --no-build-isolation
```

After installation, notebooks can import:

```python
from ADFWI.model import AcousticModel
from ADFWI.propagator import AcousticPropagator
```

without:

```python
sys.path.append("../../../../")
```

## Next Direction

Apply the same installed-package import style to
`examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb`, then run the
forward/inversion example validation.

## Validation Result

```text
conda run -n adfwi python -m pip install -e . --no-deps --no-build-isolation
```

Result:

```text
Successfully installed ADFWI-Torch-0.1.2
```

Import from outside the repository:

```text
ADFWI /liufeng1afs/project/04_Inversion/ADFWI-github/ADFWI/__init__.py
ok AcousticModel AcousticPropagator
```

Notebook import/setup cell:

```text
missing []
source_lines 15
```
