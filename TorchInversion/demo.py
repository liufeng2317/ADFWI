'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-12-10 20:30:00
* LastEditors: LiuFeng
* LastEditTime: 2023-12-31 15:58:46
* FilePath: /ADFWI/TorchInversion/demo.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

import numpy as np 
import matplotlib.pyplot as plt
import os 
import obspy
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter


############################################################
#                   Marmousi Model
############################################################
def extract_data(meta):
    data = []
    for trace in meta:
        data.append(trace.data)
    return np.array(data)


def load_marmousi_model(in_dir):
    # in_dir = os.path.join("./data/Marmousi2/marmousi_source/")
    if not os.path.exists(os.path.join(in_dir, "vp_marmousi-ii.segy.gz")):
        os.system("wget {} -P {}".format("http://www.agl.uh.edu/downloads/vp_marmousi-ii.segy.gz", in_dir))
    if not os.path.exists(os.path.join(in_dir, "vs_marmousi-ii.segy.gz")):
        os.system("wget {} -P {}".format("http://www.agl.uh.edu/downloads/vs_marmousi-ii.segy.gz", in_dir))
    if not os.path.exists(os.path.join(in_dir, "density_marmousi-ii.segy.gz")):
        os.system("wget {} -P {}".format("http://www.agl.uh.edu/downloads/density_marmousi-ii.segy.gz", in_dir))
    meta = obspy.read(os.path.join(in_dir, "vs_marmousi-ii.segy.gz"), format='segy')
    vs = extract_data(meta) * 1e3 #m/s^2
    meta = obspy.read(os.path.join(in_dir, "vp_marmousi-ii.segy.gz"), format='segy')
    vp = extract_data(meta) * 1e3 #m/s^2
    meta = obspy.read(os.path.join(in_dir, "density_marmousi-ii.segy.gz"), format='segy')
    rho = extract_data(meta) * 1e3 #kg/m^3
    
    x_range = [0, 17000] #m
    y_range = [0, 3500] #m
    nx, ny = vp.shape
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)

    marmousi_model = {}
    marmousi_model['vp'] = vp
    marmousi_model['vs'] = vs
    marmousi_model['rho'] = rho
    marmousi_model['x'] = x
    marmousi_model['y'] = y
    marmousi_model['dx'] = x[1] - x[0]
    marmousi_model['dy'] = y[1] - y[0]
    
    return marmousi_model

def resample_model(x, y, model):
    vs = interp2d(model['y'], model['x'], model['vs'], kind='cubic')(y, x)
    vp = interp2d(model['y'], model['x'], model['vp'], kind='cubic')(y, x)
    rho = interp2d(model['y'], model['x'], model['rho'], kind='cubic')(y, x)
    
    new_model = {}
    new_model['vp'] = vp
    new_model['vs'] = vs
    new_model['rho'] = rho
    new_model['x'] = x
    new_model['y'] = y
    new_model['dx'] = x[1] - x[0]
    new_model['dy'] = y[1] - y[0]
    
    return new_model

def get_smooth_marmousi_model(model):
    mask_extra_detph = 2
    rcv_depth = 12
    if mask_extra_detph > 0:
        vp = model['vp'].copy()
        vp[:,rcv_depth+mask_extra_detph:] = gaussian_filter(model['vp'][:, rcv_depth+mask_extra_detph:], [10,10], mode='reflect')
        vs = model['vs'].copy()
        vs[:,rcv_depth+mask_extra_detph:] = gaussian_filter(model['vs'][:, rcv_depth+mask_extra_detph:], [10,10], mode='reflect')
        rho = model['rho'].copy()
        rho[:,rcv_depth+mask_extra_detph:] = gaussian_filter(model['rho'][:, rcv_depth+mask_extra_detph:], [10,10], mode='reflect')
    else:
        vp = model['vp'].copy()
        vp = gaussian_filter(model['vp'], [10,10], mode='reflect')
        vs = model['vs'].copy()
        vs = gaussian_filter(model['vs'], [10,10], mode='reflect')
        rho = model['rho'].copy()
        rho = gaussian_filter(model['rho'], [10,10], mode='reflect')
    
    new_model = {}
    new_model['vp'] = vp
    new_model['vs'] = vs
    new_model['rho'] = rho
    new_model['x'] = model['x']
    new_model['y'] = model['y']
    new_model['dx'] = model['dx']
    new_model['dy'] = model['dy']
    return new_model


############################################################
#                   Layer Model
############################################################
from scipy.interpolate import interp1d
def step_profile_layerModel(x_range, y_range, step):
    y_step1   =  np.round(np.arange(y_range[0], y_range[1]+step, step)/step) * step
    vp_step1  =  y_step1/(y_range[1]-y_range[0])  * (6.5-5) + 3
    vs_step1  =  y_step1/(y_range[1]-y_range[0])  * (4.48-3.46) + 3.46
    rho_step1 =  y_step1/(y_range[1]-y_range[0])  * (3.32-2.72) + 2.72
    
    y_step2   =  y_step1 + (y_step1[1] - y_step1[0] - step/5)
    vp_step2  =  vp_step1
    vs_step2  =  vs_step1
    rho_step2 =  rho_step1

    idy       = np.argsort(np.hstack([y_step1, y_step2]))
    y_step    = np.hstack([y_step1, y_step2])[idy]
    vp_step   = np.hstack([vp_step1, vp_step2])[idy]
    vs_step   = np.hstack([vs_step1, vs_step2])[idy]
    rho_step  = np.hstack([rho_step1, rho_step2])[idy]
    vp_step[-1:]  = vp_step[-2]
    vs_step[-1:]  = vs_step[-2]
    rho_step[-1:] = rho_step[-2]
    
    return y_step, vp_step, vs_step, rho_step

def build_layer_model(x, y, step):
    y_step, vp_step, vs_step, rho_step = step_profile_layerModel([x[0], x[-1]], [y[0], y[-1]], step)

    vp = interp1d(y_step, vp_step, kind='slinear')(y)
    vs = interp1d(y_step, vs_step, kind='slinear')(y)
    rho = interp1d(y_step, rho_step, kind='slinear')(y)
    
    vp = np.tile(vp[np.newaxis,:], [len(x),1])
    vs = np.tile(vs[np.newaxis,:], [len(x),1])
    rho = np.tile(rho[np.newaxis,:], [len(x),1])
    
    model = {}
    model['vp'] = vp
    model['vs'] = vs
    model['rho'] = rho
    model['x'] = x
    model['y'] = y
    model['dx'] = x[1] - x[0]
    model['dy'] = y[1] - y[0]
    return model

def get_smooth_layer_model(model):
    vp = model['vp'].copy()
    vp = gaussian_filter(model['vp'], [10,10], mode='reflect')
    vs = model['vs'].copy()
    vs = gaussian_filter(model['vs'], [10,10], mode='reflect')
    rho = model['rho'].copy()
    rho = gaussian_filter(model['rho'], [10,10], mode='reflect')
    new_model = {}
    new_model['vp'] = vp
    new_model['vs'] = vs
    new_model['rho'] = rho
    new_model['x'] = model['x']
    new_model['y'] = model['y']
    new_model['dx'] = model['dx']
    new_model['dy'] = model['dy']
    return new_model


############################################################
#                   Layer Anomaly Model
############################################################
from scipy.interpolate import interp1d
def step_profile_anomaly(x_range, y_range, step):
    y_step1 = np.round(np.arange(y_range[0], y_range[1]+step, step)/step) * step
    vp_step1 = y_step1/(y_range[1]-y_range[0]) * (8.04-5.8) + 5.8
    vs_step1 = y_step1/(y_range[1]-y_range[0]) * (4.48-3.46) + 3.46
    rho_step1 = y_step1/(y_range[1]-y_range[0]) * (3.32-2.72) + 2.72
    
    y_step2 = y_step1 + (y_step1[1] - y_step1[0] - 1)
    vp_step2 = vp_step1
    vs_step2 = vs_step1
    rho_step2 = rho_step1

    idy = np.argsort(np.hstack([y_step1, y_step2]))
    y_step = np.hstack([y_step1, y_step2])[idy]
    vp_step = np.hstack([vp_step1, vp_step2])[idy]
    vs_step = np.hstack([vs_step1, vs_step2])[idy]
    rho_step = np.hstack([rho_step1, rho_step2])[idy]
    vp_step[-1:] = vp_step[-2]
    vs_step[-1:] = vs_step[-2]
    rho_step[-1:] = rho_step[-2]
    
    return y_step, vp_step, vs_step, rho_step

def build_anomaly_backgroud_model(x, y, step):
    y_step, vp_step, vs_step, rho_step = step_profile_anomaly([x[0], x[-1]], [y[0], y[-1]], step)

    vp = interp1d(y_step, vp_step, kind='slinear')(y)
    vs = interp1d(y_step, vs_step, kind='slinear')(y)
    rho = interp1d(y_step, rho_step, kind='slinear')(y)
    
    vp = np.tile(vp[np.newaxis,:], [len(x),1])
    vs = np.tile(vs[np.newaxis,:], [len(x),1])
    rho = np.tile(rho[np.newaxis,:], [len(x),1])
    
    model = {}
    model['vp'] = vp
    model['vs'] = vs
    model['rho'] = rho
    model['x'] = x
    model['y'] = y
    model['dx'] = x[1] - x[0]
    model['dy'] = y[1] - y[0]
    
    return model

def get_anomaly_model(layer_model, n_pml):
    x = layer_model['x']
    y = layer_model['y']
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    vp = layer_model['vp'].copy()
    vs = layer_model['vs'].copy()
    rho = layer_model['rho'].copy()
    
    x0 = (x[-1]-1*n_pml*dx)*2/3 + 0.5*n_pml*dx
    y0 = (y[-1]-1*n_pml*dy)*1/3 + 0.5*n_pml*dy
    a = x[-1]/6
    b = y[-1]/10
    anomaly1 = np.zeros_like(vp)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            if ((xi-x0)/a)**2 + ((yj-y0)/b)**2 < 1:
                anomaly1[i, j] = 1
            
    x0 = (x[-1]-1*n_pml*dx)/3 + 0.5*n_pml*dx
    y0 = (y[-1]-1*n_pml*dy)*2/3 + 0.5*n_pml*dy
    a = x[-1]/6
    b = y[-1]/10
    anomaly2 = np.zeros_like(vp)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            if ((xi-x0)/a)**2 + ((yj-y0)/b)**2 < 1:
                anomaly2[i, j] = 1
                

    vp[anomaly1==1] = np.mean(vp[anomaly1==1])*1.1
    vp[anomaly2==1] = np.mean(vp[anomaly2==1])/1.1

    vs[anomaly1==1] = np.mean(vs[anomaly1==1])*1.1
    vs[anomaly2==1] = np.mean(vs[anomaly2==1])/1.1

    rho[anomaly1==1] = np.mean(rho[anomaly1==1])*1.1
    rho[anomaly2==1] = np.mean(rho[anomaly2==1])/1.1
    
    anomaly_model = {}
    anomaly_model['vp']  = vp
    anomaly_model['vs']  = vs
    anomaly_model['rho'] = rho
    anomaly_model['x']   = layer_model['x']
    anomaly_model['y']   = layer_model['y']
    anomaly_model['dx']  = layer_model['dx']
    anomaly_model['dy']  = layer_model['dy']
    return anomaly_model