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

## Next Direction

Validate elastic backend inheritance separately before removing explicit
`device=device, dtype=dtype` from elastic model construction or
`device=device` from `ElasticPropagator`.
