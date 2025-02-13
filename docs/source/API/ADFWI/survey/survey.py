from typing import Optional
from .receiver import Receiver
from .source import Source
from ADFWI.utils import list2numpy
from ADFWI.view import plot_survey
import numpy as np

class Survey(object):
    """Survey class describes the seismic acquisition geometry (2D). Assumes that all sources share the same receivers, time samples, and time interval.
    
    Parameters
    ----------
    source : Source
        Source object containing the source information.
    receiver : Receiver
        Receiver object containing the receiver information.
    receiver_masks : numpy.ndarray, optional
        The index of useful receivers at each shot, of shape [shot number, receiver number].
        By default, None. This parameter is useful for designing special observation systems
        where only certain receivers are active (e.g., line one-side).
    receiver_masks_obs : bool, optional
        Indicates whether the observation waveforms need to be masked or not.
        By default, True.
    """
    
    def __init__(self, source: Source, receiver: Receiver, receiver_masks=None, receiver_masks_obs=True) -> None:
        """
        Initializes the Survey object with source, receiver, and optional receiver masks.
        """
        self.source = source
        self.receiver = receiver
        self.receiver_masks = None  # Masks for receiver availability
        self.receiver_masks_obs = receiver_masks_obs  # Flag for waveform masking
        if receiver_masks is not None:
            self.set_receiver_masks(receiver_masks)
    
    def set_receiver_masks(self, receiver_masks: np.ndarray) -> None:
        """
        Set the receiver masks for the survey. Masks specify which receivers are useful for each shot.
        
        Parameters
        ----------
        receiver_masks : numpy.ndarray
            The receiver mask array with shape [number of shots, number of receivers].
        """
        src_x, src_z = list2numpy(self.source.loc_x), list2numpy(self.source.loc_z)
        rcv_x, rcv_z = list2numpy(self.receiver.loc_x), list2numpy(self.receiver.loc_z)
        if receiver_masks.shape[0] == len(src_x) and receiver_masks.shape[1] == len(rcv_x):        
            self.receiver_masks = receiver_masks
        else:
            raise ValueError(
                "Receiver Mask Error: The number of receivers or sources does not match the Mask dimensions."
            )
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the Survey object, including the source and receiver information.
        """
        info = f"Survey Information:\n"
        info += repr(self.source)
        info += "\n"
        info += repr(self.receiver)
        return info
    
    def plot(self, model_data: np.ndarray, **kwargs) -> None:
        """
        Plots the entire survey including all sources and receivers with associated model data.
        """
        src_x = list2numpy(self.source.loc_x)
        src_z = list2numpy(self.source.loc_z)
        rcv_x = list2numpy(self.receiver.loc_x)
        rcv_z = list2numpy(self.receiver.loc_z)
        
        plot_survey(src_x, src_z, rcv_x, rcv_z, model_data, **kwargs)
    
    def plot_single_shot(self, model_data: np.ndarray, src_idx: int, **kwargs) -> None:
        """
        Plots a single shot's survey with associated model data.
        """
        src_x = list2numpy(self.source.loc_x[src_idx])
        src_z = list2numpy(self.source.loc_z[src_idx])
        rcv_x = list2numpy(self.receiver.loc_x)
        rcv_z = list2numpy(self.receiver.loc_z)
        
        # Apply receiver mask if available
        if self.receiver_masks is None:
            receiver_mask = np.ones(len(rcv_x))  # All receivers are active if no mask is set
        else:
            receiver_mask = self.receiver_masks[src_idx]
        
        # Filter receiver positions based on the mask
        rcv_x = rcv_x[np.argwhere(receiver_mask)]
        rcv_z = rcv_z[np.argwhere(receiver_mask)]
        
        plot_survey(src_x, src_z, rcv_x, rcv_z, model_data, **kwargs)
