from typing import Optional
from .receiver import Receiver
from .source import Source
from ADFWI.utils import list2numpy
from ADFWI.view import plot_survey

class Survey(object):
    """Survey class describes the seismic acquisition geometry (2D). I assume 
    that all sources share the same receivers, time samples, and time interval.

    Parameters
    ----------
    source : Source 
        Source object
    receiver : Receiver
        Receiver object
    device : str, optional
        Device for computation: cpu or gpu, by default 'cpu'
    cpu_num : int, optional
        Maximum number of CPU cores, if cpu, by default 1
    gpu_num : int, optional
        Maximum number of GPU cards, if cuda, by default 1
    """
    def __init__(self,source:Source,receiver:Receiver) -> None:
        self.source     = source
        self.receiver   = receiver
    
    def __repr__(self):
        """ Reimplement the repr function for printing the survey information
        """
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
        