"""FWI iteration helpers grouped by responsibility.

Import helpers from their owner modules:

- ``ADFWI.fwi.iteration.batches`` for shot batch scheduling.
- ``ADFWI.fwi.iteration.loss`` for batch loss construction and evaluation.
- ``ADFWI.fwi.iteration.step`` for one-batch forward/loss/backward steps.
- ``ADFWI.fwi.iteration.epoch`` for optimizer/scheduler epoch updates.
"""
