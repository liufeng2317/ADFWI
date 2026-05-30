# Elastic Anomaly Forward Import Cleanup

## Boundary

```text
Goal:
Clean import/setup boilerplate in the representative elastic forward notebook.

Scope:
examples/elastic/Iso-elastic-Anomaly/01_forward.ipynb

Validation:
Check notebook JSON validity and execute the import/setup cell in the `adfwi`
conda environment.

Stop:
Do not change elastic model construction, backend behavior, propagator
construction, forward parameters, plotting, or generated outputs in this round.
```

## Change

Replaced the old path and wildcard import setup:

```python
sys.path.append("../../../../")
from ADFWI.propagator import *
from ADFWI.model import *
from ADFWI.view import *
from ADFWI.utils import *
from ADFWI.survey import *
```

with explicit imports used by this notebook:

```python
from ADFWI.model import IsotropicElasticModel
from ADFWI.propagator import ElasticPropagator
from ADFWI.survey import Receiver, SeismicData, Source, Survey
from ADFWI.utils import wavelet
from ADFWI.view import plot_bcx_bcz, plot_damp
```

The repeated output-directory creation blocks were replaced with a local
`exist_ok=True` loop.

## Validation Result

```text
python -m json.tool examples/elastic/Iso-elastic-Anomaly/01_forward.ipynb
conda run -n adfwi python -c "<execute import/setup cell>"
```

Result:

```text
missing []
source_lines 14
```

## Backend Sync Follow-Up

The notebook was then aligned with the bv1.2 backend pattern:

```python
import ADFWI

device = "npu:0"
dtype = torch.float32
backend = ADFWI.set_backend(device, dtype=dtype)

model = IsotropicElasticModel(...)
F = ElasticPropagator(model, survey)
```

`IsotropicElasticModel` and `ElasticPropagator` no longer need explicit
`device`/`dtype` arguments when the global backend has already been set. A
minimal construction check confirmed that model tensors, propagator buffers,
and source wavelets inherit `npu:0` and `torch.float32`.

Notebook outputs and execution counts were cleared after the check so this file
can be used as the clean elastic forward synchronization template.

## Validation Result

```text
python -m json.tool examples/elastic/Iso-elastic-Anomaly/01_forward.ipynb
conda run -n adfwi python -c "<minimal elastic backend inheritance check>"
```

Result:

```text
inherits_backend True
```

## Next Direction

Use `examples/elastic/Iso-elastic-Anomaly/01_forward.ipynb` as the elastic
forward notebook template when synchronizing the remaining elastic examples:
explicit imports, one backend setup cell, no `sys.path`, and no repeated
`device`/`dtype` arguments when construction inherits the active backend.
