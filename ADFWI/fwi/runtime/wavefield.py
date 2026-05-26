"""Forward-wavefield accumulation helpers for FWI gradient processing.

These helpers keep the historical ``GradProcessor`` contract: wavefields passed
to gradient processors are detached CPU NumPy arrays accumulated over shot
batches. They do not decide which physical wavefield component is appropriate
for a given inversion parameter.
"""


def wavefield_to_numpy(wavefield):
    """Return the detached CPU NumPy wavefield used by legacy processors."""
    return wavefield.cpu().detach().numpy()


def accumulate_wavefield(accumulator, wavefield):
    """Accumulate one detached wavefield into an optional NumPy accumulator."""
    snapshot = wavefield_to_numpy(wavefield)
    if accumulator is None:
        return snapshot
    accumulator += snapshot
    return accumulator


def accumulate_named_wavefields(accumulators, wavefields):
    """Accumulate a mapping of named wavefield tensors over shot batches."""
    for name, wavefield in wavefields.items():
        accumulators[name] = accumulate_wavefield(accumulators.get(name), wavefield)
    return accumulators


def select_elastic_gradient_wavefield(accumulated_wavefields):
    """Select the legacy elastic wavefield passed to ``GradProcessor``.

    Historical elastic FWI used pressure when present, otherwise ``vz``. The
    final ``vx`` fallback makes vx-only component inversions usable while
    preserving existing pressure and vz behavior.
    """
    for name in ("pressure", "vz", "vx"):
        if name in accumulated_wavefields:
            return accumulated_wavefields[name]
    raise ValueError("no accumulated elastic wavefield is available for gradient processing")
