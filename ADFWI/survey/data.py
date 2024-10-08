import numpy as np 
import matplotlib.pyplot as plt

from typing import Optional,List,Union
from ADFWI.survey import Survey
from ADFWI.utils import gpu2cpu,tensor2numpy
from ADFWI.view import plot_waveform2D,plot_waveform_wiggle,plot_waveform_trace

class SeismicData():
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
        
    def __repr__(self):
        """ Print the survey information
        """

        info = f"Seismic Data:\n"
        info += f"  Source number : {self.src_num}\n"
        info += f"  Receiver number : {self.rcv_num}\n"
        info += f"  Time samples : {self.nt} samples at {self.dt * 1000:.2f} ms\n"

        return info

    def record_data(self, data: dict):
        """ Add the shot gather data to the class

        Parameters:
        ----------
        data: dict
            shot gather data in dictionary format
        """
        for key,value in data.items():
            value      = tensor2numpy(gpu2cpu(value)).copy()
            data[key]  = value
        self.data = data
    
    def save(self,path:str):
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
        """ Load the shot gather data

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
    
    def parse_elastic_data(self,normalize=False):
        txx = self.data["txx"]
        tzz = self.data["tzz"]
        txz = self.data["txz"]
        vx  = self.data["vx"]
        vz  = self.data["vz"]
        pressure = -(txx + tzz)
        if normalize:
            pressure = pressure/np.max(np.abs(pressure),axis=1,keepdims=True)
            txz      = txz/np.max(np.abs(txz),axis=1,keepdims=True)
            vx       = vx/np.max(np.abs(vx),axis=1,keepdims=True)
            vz       = vz/np.max(np.abs(vz),axis=1,keepdims=True)
        return pressure,txz,vx,vz

    def parse_acoustic_data(self,normalize=False):
        pressure = self.data["p"]
        u = self.data["u"]
        w = self.data["w"]
        if normalize:
            pressure = pressure/np.max(np.abs(pressure),axis=1,keepdims=True)
            u      = u/np.max(np.abs(u),axis=1,keepdims=True)
            w      = w/np.max(np.abs(w),axis=1,keepdims=True)
        return pressure,u,w    
    
    def plot_waveform2D(self,i_shot,rcv_type="pressure",acoustic_or_elastic="acoustic",normalize=True,**kwargs):
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