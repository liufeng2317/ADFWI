"""Runtime helpers for FWI driver construction and execution.

Import helpers from their owner modules:

- ``ADFWI.fwi.runtime.backend`` for backend/device validation.
- ``ADFWI.fwi.runtime.cache`` for result-cache bookkeeping.
- ``ADFWI.fwi.runtime.forward`` for per-batch forward records.
- ``ADFWI.fwi.runtime.gradient`` for gradient parameter specs and dispatch.
- ``ADFWI.fwi.runtime.regularization`` for model regularization loss helpers.
- ``ADFWI.fwi.runtime.wavefield`` for wavefield selection and accumulation.
"""
