"""Survey-level acquisition geometry built from sources and receivers.

`Survey` is the acquisition-state object. It composes a `Source`, a `Receiver`,
and optional receiver masks. It does not own waveform arrays, propagator
execution, or FWI loss logic.
"""

from .receiver import Receiver
from .source import Source
from ADFWI.utils import list2numpy
from ADFWI.view import plot_survey
import numpy as np

class Survey(object):
    """2D seismic acquisition geometry.

    A survey combines one `Source` collection and one `Receiver` collection.
    Current propagator paths assume all shots share the same receiver array.

    Parameters
    ----------
    source : Source 
        Source object
    receiver : Receiver
        Receiver object
    receiver_masks: Active receiver mask for each shot
        array-like with shape ``(source.num, receiver.num)``, by default None.
        Nonzero entries mark active receivers.
        This parameter is useful for designing special observation systems.
                * v v v v v v v v
                    * v v v v v v
                        * v v v v
                            * v v 
                (* means source, v means receiver)
    receiver_masks_obs : bool
        Whether observed waveforms are expected to use receiver masks.
    """
    def __init__(self,source:Source,receiver:Receiver,receiver_masks=None,receiver_masks_obs=True) -> None:
        self.source         = source
        self.receiver       = receiver
        # receiver_masks marks active receivers for each shot. Survey stores
        # this state; propagator/FWI code decides when to apply it.
        self.receiver_masks = None
        # Whether observed waveforms are already masked before entering FWI.
        self.receiver_masks_obs = receiver_masks_obs
        if receiver_masks is not None:
            self.set_receiver_masks(receiver_masks)
    
    def set_receiver_masks(self,receiver_masks):
        """Set receiver masks with shape ``(source.num, receiver.num)``."""
        receiver_masks = np.asarray(receiver_masks)
        if receiver_masks.ndim != 2:
            raise ValueError(
                f"Receiver Mask Error: receiver_masks must be 2-D [source, receiver], got {receiver_masks.shape}"
            )
        src_x = list2numpy(self.source.loc_x)
        rcv_x = list2numpy(self.receiver.loc_x)
        expected_shape = (len(src_x), len(rcv_x))
        if receiver_masks.shape != expected_shape:
            raise ValueError(
                "Receiver Mask Error: receiver_masks shape must match "
                f"(source.num, receiver.num)={expected_shape}, got {receiver_masks.shape}"
            )
        self.receiver_masks = receiver_masks
        
    
    def __repr__(self):
        """Return a readable acquisition summary."""
        info = f"Survey Information:\n"
        info += repr(self.source)
        info += "\n"
        info += repr(self.receiver)
        return info
    
    def plot(self,model_data,**kwargs):
        src_x = list2numpy(self.source.loc_x)
        src_z = list2numpy(self.source.loc_z)
        rcv_x = list2numpy(self.receiver.loc_x)
        rcv_z = list2numpy(self.receiver.loc_z)
        
        plot_survey(src_x,src_z,rcv_x,rcv_z,model_data,**kwargs)
    
    def plot_single_shot(self,model_data,src_idx,**kwargs):
        src_x = list2numpy(self.source.loc_x[src_idx])
        src_z = list2numpy(self.source.loc_z[src_idx])
        rcv_x = list2numpy(self.receiver.loc_x)
        rcv_z = list2numpy(self.receiver.loc_z)
        if self.receiver_masks is None:
            receiver_mask = np.ones(len(rcv_x))
        else:
            receiver_mask = self.receiver_masks[src_idx]
        rcv_x = rcv_x[np.argwhere(receiver_mask)]
        rcv_z = rcv_z[np.argwhere(receiver_mask)]

        plot_survey(src_x,src_z,rcv_x,rcv_z,model_data,**kwargs)
