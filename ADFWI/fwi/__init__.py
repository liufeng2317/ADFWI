"""High-level FWI driver classes.

Most implementation details live in subpackages:

- ``iteration`` owns batch/epoch/loss-loop mechanics;
- ``runtime`` owns shared execution helpers;
- ``transforms`` owns waveform preprocessing;
- ``misfit`` and ``regularization`` own objective terms.

The public top-level API intentionally stays small.
"""

from .elastic_fwi import ElasticFWI
from .acoustic_fwi import AcousticFWI

__all__ = [
    "AcousticFWI",
    "ElasticFWI",
]
