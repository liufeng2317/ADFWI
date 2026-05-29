"""Survey-level acquisition geometry built from sources and receivers."""

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
    receiver_masks: The index of useful receiver at each shot
        numpy: [shot number, receiver number], by default None
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
        # receive mask  -> mask some of the receiver are useful while other are not
        self.receiver_masks = None
        # receiver_masks_obs -> mark if the obs waveform need to be masked or not
        self.receiver_masks_obs = receiver_masks_obs
        if receiver_masks is not None:
            self.set_receiver_masks(receiver_masks)
    
    def set_receiver_masks(self,receiver_masks):
        """Set receiver masks with shape ``(source.num, receiver.num)``."""
        src_x,src_z = list2numpy(self.source.loc_x),list2numpy(self.source.loc_z)
        rcv_x,rcv_z = list2numpy(self.receiver.loc_x),list2numpy(self.receiver.loc_z)
        if receiver_masks.shape[0] == len(src_x) and receiver_masks.shape[1] == len(rcv_x):        
            self.receiver_masks = receiver_masks
        else:
            raise ValueError(
                "Receiver Mask Errror: the number of receiver/source are not equal to the Mask"
            )
        
    
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
