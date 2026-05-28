"""Base class for waveform misfit objectives."""

from abc import ABC, abstractmethod


class Misfit(ABC):
    """Abstract interface used by FWI drivers and loss dispatch helpers."""

    @abstractmethod
    def forward(self, obs, syn):
        """Return the misfit tensor for observed and synthetic waveforms."""
        raise NotImplementedError
