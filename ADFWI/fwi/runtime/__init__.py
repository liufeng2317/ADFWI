"""Runtime helpers for AcousticFWI/ElasticFWI driver execution.

This package is not a high-level forward or inversion framework. It contains
small owner modules for mechanics shared by the concrete FWI drivers. Physical
choices such as active model parameters, inversion components, loss components,
and regularization weights remain owned by ``AcousticFWI`` and ``ElasticFWI``.

Import helpers from their owner modules:

- ``ADFWI.fwi.runtime.backend`` for construction-time backend/device alignment.
- ``ADFWI.fwi.runtime.cache`` for inversion-history bookkeeping.
- ``ADFWI.fwi.runtime.forward`` for one-batch propagator execution records.
- ``ADFWI.fwi.runtime.gradient`` for gradient processor dispatch.
- ``ADFWI.fwi.runtime.regularization`` for model regularization loss helpers.
- ``ADFWI.fwi.runtime.wavefield`` for forward-wavefield extraction/accumulation
  used by gradient processors.
"""
