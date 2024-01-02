'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-12-10 16:56:16
* LastEditors: LiuFeng
* LastEditTime: 2023-12-11 14:39:20
* FilePath: /Acoustic_AD/TorchInversion/config.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import numpy as np
import json 
import os
from scipy import integrate
from TorchInversion.utils import source_wavelet,dictToObj,set_damp,list2numpy


class Config():
    def __init__(self,save_path) -> None:
        self.save_path = save_path
    
    def base_param(self,nx,ny,dx,dy,pml,fs,nt,dt,vmin,vmax):
        """set the parameters of model"""
        nx_pml = nx+2*pml
        ny_pml = ny+2*pml
        param = {
            "nx":nx,"ny":ny,
            "dx":dx,"dy":dy,
            "nt":nt,"dt":dt,
            "pml":pml,"fs":fs,
            "nx_pml":nx_pml,"ny_pml":ny_pml,
            "vmax":vmax,"vmin":vmin
        }
        param = dictToObj(param)
        return param
    
    def source(self,f0,src_x,src_y,param):
        """set the parameters of source"""
        nt = param.nt
        dt = param.dt
        pml = param.pml
        
        # free surface
        mask =  (src_x==pml)
        src_x[mask] = src_x[mask] + 1
        
        # wavelet
        src = source_wavelet(nt, dt, f0, 'Ricker') #Ricker wavelet
        stf_val = integrate.cumtrapz(src, axis=-1, initial=0) #Integrate
        stf_t = np.arange(nt)*dt
        
        acoustic_src = {
            "f0":f0,
            "src_x":src_x,"src_y":src_y,"src_n":len(src_x),
            "stf_val":stf_val,"stf_t":stf_t
        }
        acoustic_src = dictToObj(acoustic_src)
        
        return acoustic_src
        
    
    def receiver(self,rcv_x,rcv_y):
        acoustic_rcv = {
            'rcv_x':rcv_x,
            'rcv_y':rcv_y,
            'rcv_n':len(rcv_x)
        }
        acoustic_rcv = dictToObj(acoustic_rcv)
        return acoustic_rcv
    
    def vel_model(self,v,rho,param):
        vmax = param.vmax
        nx_pml = param.nx_pml
        ny_pml = param.ny_pml
        pml = param.pml
        dx = param.dx
        damp_global = set_damp(vmax,nx_pml,ny_pml,pml,dx)
        # velocity model
        vel_model ={
            "v":v,
            "rho":rho,
            "damp_global":damp_global
        }
        vel_model= dictToObj(vel_model)
        return vel_model
    
    def optimizer(self,lr,iteration,step_size,gamma,optim_method="Adam",device="cpu"):
        """parameters for inversion"""
        optimizer_param = {
            "lr":lr,
            "iteration":iteration,
            "step_size":step_size,
            "gamma":gamma,
            "optim_method":optim_method,
            "device":device,
        }
        optimizer_param = dictToObj(optimizer_param)
        return optimizer_param
    
    def save(self,param,source,receiver,vel_model,optimizer={},inversion=False):
        """save the model parameters"""
        class NdarrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        param_save = {
            "settings":param,
            "source":source,
            "receiver":receiver,
            "vel_model":vel_model,
            "optimizer":optimizer
        }
        param_save = dictToObj(param_save)
        if inversion:
            with open(os.path.join(self.save_path,"param_inv.json"),'w') as f:
                json.dump(param_save,f,cls=NdarrayEncoder)
        else:
            with open(os.path.join(self.save_path,"param_for.json"),'w') as f:
                json.dump(param_save,f,cls=NdarrayEncoder)
        return param_save
    
    def load(self,param_path=""):
        """get the parameter from json file"""
        if not param_path=="":
            param = json.load(open(self.param_path,"r"))
        elif os.path.exists(os.path.join(self.save_path,"param_for.json")):
            file_path = os.path.join(self.save_path,"param_for.json")
            param = json.load(open(file_path))
        
        param = dictToObj(param)
        param_setting = param.settings
        # source
        source = param.source
        source.src_x = list2numpy(source.src_x)
        source.src_y = list2numpy(source.src_y)
        source.stf_val = list2numpy(source.stf_val)
        source.stf_t = list2numpy(source.stf_t)
        # receiver
        receiver = param.receiver
        receiver.rcv_x = list2numpy(receiver.rcv_x)
        receiver.rcv_y = list2numpy(receiver.rcv_y)
        # velocity model
        vel_model = param.vel_model
        vel_model.v = list2numpy(vel_model.v)
        vel_model.rho = list2numpy(vel_model.rho)
        vel_model.damp_global = list2numpy(vel_model.damp_global)
        
        return param_setting,source,receiver,vel_model
        