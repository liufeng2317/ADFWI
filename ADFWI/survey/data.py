"""Recorded seismic waveform data and survey metadata snapshots.

`SeismicData` stores waveform arrays and the survey metadata snapshot used to
interpret them. Parsing and plotting methods are compatibility helpers around
that stored state; source/receiver geometry mutation belongs to `Survey`,
`Source`, and `Receiver`.
"""

import numpy as np 

from ADFWI.survey import Survey
from ADFWI.utils import gpu2cpu,tensor2numpy
from ADFWI.view import plot_waveform2D,plot_waveform_wiggle,plot_waveform_trace

class SeismicData():
    """Container for recorded acoustic or elastic waveform dictionaries.

    `SeismicData` snapshots source/receiver metadata from a `Survey`, records
    propagator output dictionaries as numpy arrays, and saves/loads the current
    `.npz` format. It does not own acquisition geometry mutation, propagator
    execution, or FWI loss construction.
    """
    def __init__(self,survey:Survey):
        self.survey     = survey
        # get the survey information
        self.src_num    = survey.source.num
        self.rcv_num    = survey.receiver.num
        self.src_loc    = survey.source.get_loc()
        self.rcv_loc    = survey.receiver.get_loc()
        self.src_type   = survey.source.get_type()
        self.rcv_type   = survey.receiver.get_type()
        self.nt         = survey.receiver.nt
        self.dt         = survey.receiver.dt
        self.t          = np.arange(self.nt)*self.dt
        
        # data
        self.data = None
        self.data_masks = None
        
    def __repr__(self):
        """Return a readable seismic-data summary."""

        info = f"Seismic Data:\n"
        info += f"  Source number : {self.src_num}\n"
        info += f"  Receiver number : {self.rcv_num}\n"
        info += f"  Time samples : {self.nt} samples at {self.dt * 1000:.2f} ms\n"

        return info

    def record_data(self, data: dict):
        """Record a propagator waveform dictionary as numpy arrays.

        Parameters:
        ----------
        data: dict
            shot gather data in dictionary format
        """
        recorded = {}
        for key,value in data.items():
            recorded[key] = tensor2numpy(gpu2cpu(value)).copy()
        self.data = recorded

    def _require_components(self, keys, data_kind):
        """Return stored component arrays after checking the recorded state."""
        if self.data is None:
            raise ValueError(f"No {data_kind} waveform data has been recorded or loaded")
        missing = [key for key in keys if key not in self.data]
        if missing:
            raise ValueError(
                f"Missing {data_kind} waveform component(s): {', '.join(missing)}"
            )
        return tuple(self.data[key] for key in keys)
    
    def save(self,path:str):
        """Save waveform data and survey metadata to the current `.npz` format."""
        data_save = {   'data'      : self.data,
                        'src_loc'   : self.src_loc,
                        'rcv_loc'   : self.rcv_loc,
                        'src_num'   : self.src_num,
                        'rcv_num'   : self.rcv_num,
                        'rcv_type'  : self.rcv_type,
                        'src_type'  : self.src_type,
                        't'         : self.t,
                        'nt'        : self.nt,
                        'dt'        : self.dt
                    }
        np.savez(path, **data_save) 
    
    def load(self, path: str):
        """Load waveform data and survey metadata from the current `.npz` format.

        Parameters:
        ----------
        path: str
            load path
        """

        data = np.load(path, allow_pickle=True)

        # load the data
        self.data       = data['data'].item()
        self.src_loc    = data['src_loc']
        self.rcv_loc    = data['rcv_loc']
        self.src_num    = data['src_num']
        self.rcv_num    = data['rcv_num']
        self.rcv_type   = data['rcv_type']
        self.src_type   = data['src_type']
        self.t          = data['t']
        self.nt         = data['nt']
        self.dt         = data['dt']
        return
    
    def normalize_and_mask(self,array):
        """Normalize each trace along time while preserving all-zero traces."""
        time_sum = np.sum(np.abs(array), axis=1, keepdims=True)
        mask = time_sum == 0
        max_val  = np.max(np.abs(array), axis=1, keepdims=True)
        max_val = np.where(mask, 1, max_val)
        array = array / max_val
        return array
    
    def parse_elastic_data(self,normalize=False):
        """Return elastic receiver components as ``pressure, txz, vx, vz``."""
        txx, tzz, txz, vx, vz = self._require_components(
            ("txx", "tzz", "txz", "vx", "vz"),
            "elastic",
        )
        pressure = -(txx + tzz)
        if normalize:
            pressure = self.normalize_and_mask(pressure)
            txz      = self.normalize_and_mask(txz)
            vx       = self.normalize_and_mask(vx)
            vz       = self.normalize_and_mask(vz)
        return pressure,txz,vx,vz

    def parse_acoustic_data(self,normalize=False):
        """Return acoustic receiver components as ``pressure, u, w``."""
        pressure, u, w = self._require_components(("p", "u", "w"), "acoustic")
        if normalize:
            pressure = self.normalize_and_mask(pressure)
            u = self.normalize_and_mask(u)
            w = self.normalize_and_mask(w)

        return pressure,u,w    
    
    def plot_waveform2D(self,i_shot,rcv_type="pressure",acoustic_or_elastic="acoustic",normalize=True,**kwargs):
        """Plot one shot gather with the stored waveform plotting helpers."""
        if acoustic_or_elastic == "acoustic":
            pressure,vx,vz = self.parse_acoustic_data(normalize=normalize)
        elif acoustic_or_elastic == "elastic":
            pressure,txz,vx,vz = self.parse_elastic_data(normalize=normalize)
        
        if rcv_type     == "pressure":
            plot_waveform2D(pressure[i_shot].T,**kwargs)
        elif rcv_type   == "vx":
            plot_waveform2D(vx[i_shot].T,**kwargs)
        elif rcv_type   == "vz":
            plot_waveform2D(vz[i_shot].T,**kwargs)
        elif rcv_type   == "txz":
            plot_waveform2D(txz[i_shot].T,**kwargs)
        return
    
    def plot_waveform_wiggle(self,i_shot,rcv_type="pressure",acoustic_or_elastic="acoustic",normalize=True,**kwargs):
        """Plot one shot gather as wiggle traces."""
        if acoustic_or_elastic == "acoustic":
            pressure,vx,vz = self.parse_acoustic_data(normalize=normalize)
        elif acoustic_or_elastic == "elastic":
            pressure,txz,vx,vz = self.parse_elastic_data(normalize=normalize)
        
        if rcv_type == "pressure":
            plot_waveform_wiggle(pressure[i_shot],self.survey.source.t,**kwargs)
        elif rcv_type == "vx":
            plot_waveform_wiggle(vx[i_shot],self.survey.source.t,**kwargs)
        elif rcv_type == "vz":
            plot_waveform_wiggle(vz[i_shot],self.survey.source.t,**kwargs)
        elif rcv_type == "txz":
            plot_waveform_wiggle(txz[i_shot],self.survey.source.t,**kwargs)
        return
    
    def plot_waveform_trace(self,i_shot,i_trace,rcv_type="pressure",acoustic_or_elastic="acoustic",normalize=True,**kwargs):
        """Plot one waveform trace from stored data."""
        if acoustic_or_elastic == "acoustic":
            pressure,vx,vz = self.parse_acoustic_data(normalize=normalize)
        elif acoustic_or_elastic == "elastic":
            pressure,txz,vx,vz = self.parse_elastic_data(normalize=normalize)
        
        if rcv_type == "pressure":
            plot_waveform_trace(pressure,i_shot,i_trace,self.dt,**kwargs)
        elif rcv_type == "vx":
            plot_waveform_trace(vx,i_shot,i_trace,self.dt,**kwargs)
        elif rcv_type == "vz":
            plot_waveform_trace(vz,i_shot,i_trace,self.dt,**kwargs)
        elif rcv_type == "txz":
            plot_waveform_trace(txz,i_shot,i_trace,self.dt,**kwargs)
        return
