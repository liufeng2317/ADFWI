from typing import List
import numpy as np
from ADFWI.utils import list2numpy,numpy2list

class Receiver(object):
    """Seismic Receiver class
    Parameters:
    -------------
    nt (int)    : number of time samples in the receiver data
    dt (float)  : Time inverval of data
    
    Notes:  1. The seismic data is assumed to start at time 0, e.g., ot = 0.
            2. The receiver locations should be added using the add_receiver method.
            3. The receiver is assumed to be the same for all shots, i.e., the
                receiver locations are not shot-dependent. This consideration is
                based on the fact the regular data (nshot, nrec, nt) can be dealt
                more efficiently on GPUs than irregular data. However, this may 
                limit the flexibility of modeling streamer data where the receiver
                locations are shot-dependent. The workaround solution is to apply 
                offset masking to the regular data to mimic the streamer data.
    """
    def __init__(self,nt:int,dt:float) -> None:
        self.nt = nt
        self.dt = dt
        
        self.loc_x  = []
        self.loc_z  = []
        self.locs   = []
        self.type   = []
        self.num    = 0
    
    def __repr__(self):
        """Print the receiver information
        """
        try:
            rcv_x = np.array(self.loc_x)
            rcv_z = np.array(self.loc_z)
            xmin = rcv_x.min()
            xmax = rcv_x.max()
            zmin = rcv_z.min()
            zmax = rcv_z.max()

            info = f"Seismic Receiver:\n"
            info += (
                f"  Receiver data   : {self.nt} samples at {self.dt * 1000:.2f} ms\n"
            )
            info += f"  Receiver number : {self.num}\n"
            info += f"  Receiver types  : {self.get_type(unique = True)}\n"
            info += f"  Receiver x range: {xmin} - {xmax} (grids)\n"
            info += f"  Receiver z range: {zmin} - {zmax} (grids)\n"
        except:
            info = f"Seismic Receiver:\n"
            info += f"  empty\n"

        return info
    
    def add_receivers(self, rcv_x: np.array,rcv_z:np.array, rcv_type: str) -> None:
        """add multiple receiver with same type
        """
        if rcv_x.shape != rcv_z.shape:
            raise ValueError(
                "Receiver Error: Inconsistant number of receiver in X and Z directions"
            )
        if rcv_type.lower() not in ["pr", "vx", "vz"]:
            raise ValueError("Receiver type must be either pr, vx, vz")
        rcv_n = len(rcv_x)
        # add the receiver
        self.loc_x.extend(numpy2list(rcv_x.reshape(-1)))
        self.loc_z.extend(numpy2list(rcv_z.reshape(-1)))
        self.type.extend([rcv_type]*rcv_n)
        self.num += rcv_n
    
    def add_receiver(self, rcv_x:int,rcv_z:int, rcv_type: str) -> None:
        """Append single receiver
        """
        if rcv_type.lower() not in ["pr", "vx", "vz"]:
            raise ValueError("Receiver type must be either pr, vx, vz")
        # add the receiver
        self.loc_x.append(rcv_x)
        self.loc_z.append(rcv_z)
        self.type.append(rcv_type)
        self.num += 1
    
    def get_loc(self):
        """Return the source location
        """
        rcv_x = list2numpy(self.loc_x).reshape(-1,1)
        rcv_z = list2numpy(self.loc_z).reshape(-1,1)
        rcv_loc = np.hstack((rcv_x,rcv_z))
        self.loc = rcv_loc.copy()
        return rcv_loc 
    
    def get_type(self, unique=False) -> List[str]:
        """Return the source type
        """
        type = list2numpy(self.type)
        
        if unique:
            type = list2numpy(list(set(self.type)))
        return type