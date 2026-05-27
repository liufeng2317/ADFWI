"""FWI iteration helpers grouped by responsibility.

Import helpers from their owner modules:

- ``ADFWI.fwi.iteration.range`` for batch ranges.
- ``ADFWI.fwi.iteration.components`` for acoustic/elastic component bookkeeping.
- ``ADFWI.fwi.iteration.pairs`` for synthetic/observed loss-input records.
- ``ADFWI.fwi.iteration.preparation`` for pre-misfit pair preparation.
- ``ADFWI.fwi.iteration.misfit`` for misfit dispatch and weighted loss sums.
- ``ADFWI.fwi.iteration.loss`` for batch loss/backward steps.
- ``ADFWI.fwi.iteration.progress`` for progress labels.
- ``ADFWI.fwi.iteration.epoch`` for optimizer/scheduler epoch updates.
"""
