'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-12-10 20:30:00
* LastEditors: LiuFeng
* LastEditTime: 2024-05-20 11:22:22
* FilePath: /ADFWI/TorchInversion/demo.py
* Description: 
* Copyright (c) 2023 by ${liufeng2317} email: ${liufeng2317}, All Rights Reserved.
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
    """Load the Marmousi model data from the specified directory.

    Args:
        in_dir (str): The directory where the Marmousi model files are stored.

    Returns:
        dict: A dictionary containing the velocity and density models 
              along with spatial coordinates and increments.
    """

    # Check for model files and download if missing
    for filename in ["vp_marmousi-ii.segy.gz", "vs_marmousi-ii.segy.gz", "density_marmousi-ii.segy.gz"]:
        if not os.path.exists(os.path.join(in_dir, filename)):
            os.system(f"wget http://www.agl.uh.edu/downloads/{filename} -P {in_dir}")

    # Read data from SEGY files and convert units
    vs = extract_data(obspy.read(os.path.join(in_dir, "vs_marmousi-ii.segy.gz"), format='segy')) * 1e3
    vp = extract_data(obspy.read(os.path.join(in_dir, "vp_marmousi-ii.segy.gz"), format='segy')) * 1e3
    rho = extract_data(obspy.read(os.path.join(in_dir, "density_marmousi-ii.segy.gz"), format='segy')) * 1e3

    # Define spatial ranges and create coordinate arrays
    x_range = [0, 17000]  # in meters
    y_range = [0, 3500]   # in meters
    nx, ny = vp.shape
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)

    # Create a dictionary to hold the Marmousi model data
    marmousi_model = {
        'vp': vp,
        'vs': vs,
        'rho': rho,
        'x': x,
        'y': y,
        'dx': x[1] - x[0],
        'dy': y[1] - y[0]
    }
    
    return marmousi_model  # Return the complete model data



def resample_marmousi_model(x, y, model):
    """Resample the Marmousi model to a new grid defined by x and y.

    Args:
        x (np.ndarray): The new x coordinates for resampling.
        y (np.ndarray): The new y coordinates for resampling.
        model (dict): The original Marmousi model containing velocity and density data.

    Returns:
        dict: A new model dictionary with resampled velocities and density along with new coordinates.
    """

    # Perform cubic interpolation for shear wave velocity (vs)
    vs = interp2d(model['y'], model['x'], model['vs'], kind='cubic')(y, x)
    
    # Perform cubic interpolation for primary wave velocity (vp)
    vp = interp2d(model['y'], model['x'], model['vp'], kind='cubic')(y, x)
    
    # Perform cubic interpolation for density (rho)
    rho = interp2d(model['y'], model['x'], model['rho'], kind='cubic')(y, x)

    # Create a new dictionary to hold the resampled model data
    new_model = {
        'vp': vp,
        'vs': vs,
        'rho': rho,
        'x': x,
        'y': y,
        'dx': x[1] - x[0],
        'dy': y[1] - y[0]
    }
    
    return new_model


def get_smooth_marmousi_model(model, gaussian_kernel=10, mask_extra_detph=2, rcv_depth=10):
    """Smooth the Marmousi model using a Gaussian filter.

    Args:
        model (dict): The original Marmousi model containing velocity and density data.
        gaussian_kernel (int): Standard deviation for the Gaussian kernel.
        mask_extra_detph (int): Additional depth levels to mask during smoothing.
        rcv_depth (int): Depth of the receiver.

    Returns:
        dict: A new model dictionary with smoothed velocities and density.
    """

    # Create copies of the velocity and density models for smoothing
    vp = model['vp'].copy()
    vs = model['vs'].copy()
    rho = model['rho'].copy()

    if mask_extra_detph > 0:
        # Smooth from a specified depth downwards
        vp[:, rcv_depth + mask_extra_detph:] = gaussian_filter(vp[:, rcv_depth + mask_extra_detph:], 
                                                               [gaussian_kernel, gaussian_kernel], mode='reflect')
        vs[:, rcv_depth + mask_extra_detph:] = gaussian_filter(vs[:, rcv_depth + mask_extra_detph:], 
                                                               [gaussian_kernel, gaussian_kernel], mode='reflect')
        rho[:, rcv_depth + mask_extra_detph:] = gaussian_filter(rho[:, rcv_depth + mask_extra_detph:], 
                                                                [gaussian_kernel, gaussian_kernel], mode='reflect')
    else:
        # Smooth the entire model
        vp = gaussian_filter(vp, [gaussian_kernel, gaussian_kernel], mode='reflect')
        vs = gaussian_filter(vs, [gaussian_kernel, gaussian_kernel], mode='reflect')
        rho = gaussian_filter(rho, [gaussian_kernel, gaussian_kernel], mode='reflect')

    # Create a new dictionary for the smoothed model data
    new_model = {
        'vp': vp,
        'vs': vs,
        'rho': rho,
        'x': model['x'],
        'y': model['y'],
        'dx': model['dx'],
        'dy': model['dy']
    }
    
    return new_model



def get_linear_vel_model(model, vp_min=None, vp_max=None, vs_min=None, vs_max=None):
    """Generate a linear velocity model based on the input Marmousi model.

    Args:
        model (dict): The original Marmousi model containing velocity and density data.
        vp_min (float, optional): Minimum value for the primary wave velocity.
        vp_max (float, optional): Maximum value for the primary wave velocity.
        vs_min (float, optional): Minimum value for the shear wave velocity.
        vs_max (float, optional): Maximum value for the shear wave velocity.

    Returns:
        dict: A new model dictionary with linearly varying velocities and original density.
    """
    
    vp_true = np.array(model['vp']).T
    vs_true = np.array(model['vs']).T
    rho_true = np.array(model['rho']).T
    nz, nx = vp_true.shape
    vp = np.ones_like(vp_true)
    vs = np.ones_like(vs_true)

    mask_depth = 10  # Depth below which linear variation is applied
    vp[:mask_depth, :] = vp_true[:mask_depth, :]
    vs[:mask_depth, :] = vs_true[:mask_depth, :]
    
    # Determine velocity limits if not provided
    if vp_min is None and vp_max is None:
        vp_min, vp_max = np.min(vp_true[mask_depth:, :]), np.max(vp_true[mask_depth:, :])
    if vs_min is None and vs_max is None:
        vs_min, vs_max = np.min(vs_true[mask_depth:, :]), np.max(vs_true[mask_depth:, :])

    # Create linearly spaced values for velocities below the mask depth
    vp_line = np.linspace(vp_min, vp_max, nz - mask_depth).reshape(-1, 1)
    vs_line = np.linspace(vs_min, vs_max, nz - mask_depth).reshape(-1, 1)
    vp[mask_depth:, :] *= vp_line
    vs[mask_depth:, :] *= vs_line
    rho = rho_true

    # Create a new dictionary to hold the new model data
    new_model = {
        'vp': vp.T,
        'vs': vs.T,
        'rho': rho.T,
        'x': model['x'],
        'y': model['y'],
        'dx': model['dx'],
        'dy': model['dy']
    }
    
    return new_model

    


############################################################
#                   Layer Model
############################################################
from scipy.interpolate import interp1d
def step_profile_layerModel(x_range, y_range, step):
    """Generate a step-profile layer model based on specified ranges and step size.

    Args:
        x_range (list): The x-coordinate range (not used in calculations).
        y_range (list): The y-coordinate range defining the depth limits.
        step (float): The step size for the profile.

    Returns:
        tuple: Arrays representing y-coordinates, primary wave velocity (vp),
               shear wave velocity (vs), and density (rho) for the model.
    """

    # Create rounded y-coordinates based on the specified range and step size
    y_step1 = np.round(np.arange(y_range[0], y_range[1] + step, step) / step) * step
    
    # Calculate velocities and density linearly based on y-coordinates
    vp_step1 = y_step1 / (y_range[1] - y_range[0]) * (6.5 - 5) + 3
    vs_step1 = y_step1 / (y_range[1] - y_range[0]) * (4.48 - 3.46) + 2.46
    rho_step1 = y_step1 / (y_range[1] - y_range[0]) * (3.32 - 2.72) + 2.72

    # Create second set of y-coordinates shifted by a small amount
    y_step2 = y_step1 + (y_step1[1] - y_step1[0] - step / 5)
    
    # Combine and sort the y-coordinates and corresponding properties
    idy = np.argsort(np.hstack([y_step1, y_step2]))
    y_step = np.hstack([y_step1, y_step2])[idy]
    vp_step = np.hstack([vp_step1, vp_step1])[idy]
    vs_step = np.hstack([vs_step1, vs_step1])[idy]
    rho_step = np.hstack([rho_step1, rho_step1])[idy]

    # Set the last entry to the second-to-last value to avoid discontinuity
    vp_step[-1:] = vp_step[-2]
    vs_step[-1:] = vs_step[-2]
    rho_step[-1:] = rho_step[-2]
    
    return y_step, vp_step, vs_step, rho_step


def build_layer_model(x, y, step):
    """Construct a layered model using specified x and y coordinates and a step size.

    Args:
        x (array): Array of x-coordinates defining the spatial extent.
        y (array): Array of y-coordinates defining the depth levels.
        step (float): The step size for the layer profile.

    Returns:
        dict: A dictionary containing the modeled properties: 
              primary wave velocity (vp), shear wave velocity (vs),
              density (rho), and the coordinate information.
    """

    # Generate step-profile layer model based on the specified x and y ranges
    y_step, vp_step, vs_step, rho_step = step_profile_layerModel([x[0], x[-1]], [y[0], y[-1]], step)

    # Interpolate velocities and density to match the specified y-coordinates
    vp = interp1d(y_step, vp_step, kind='slinear')(y)
    vs = interp1d(y_step, vs_step, kind='slinear')(y)
    rho = interp1d(y_step, rho_step, kind='slinear')(y)

    # Expand the velocity and density arrays to match the x-dimension
    vp = np.tile(vp[np.newaxis, :], [len(x), 1])
    vs = np.tile(vs[np.newaxis, :], [len(x), 1])
    rho = np.tile(rho[np.newaxis, :], [len(x), 1])

    # Create a model dictionary to hold the properties and coordinates
    model = {
        'vp': vp,
        'vs': vs,
        'rho': rho,
        'x': x,
        'y': y,
        'dx': x[1] - x[0],
        'dy': y[1] - y[0]
    }

    return model


def get_smooth_layer_model(model, smooth_kernel=10):
    """Apply Gaussian smoothing to the velocity and density fields of the input model.

    Args:
        model (dict): A dictionary containing the properties of the model 
                       (vp, vs, rho, x, y, dx, dy).
        smooth_kernel (int, optional): The standard deviation for Gaussian kernel smoothing. 
                                        Defaults to 10.

    Returns:
        dict: A new model dictionary with smoothed properties.
    """

    # Apply Gaussian smoothing to the velocity and density fields
    vp = gaussian_filter(model['vp'].copy(), [smooth_kernel, smooth_kernel], mode='reflect')
    vs = gaussian_filter(model['vs'].copy(), [smooth_kernel, smooth_kernel], mode='reflect')
    rho = gaussian_filter(model['rho'].copy(), [smooth_kernel, smooth_kernel], mode='reflect')

    # Prepare a new model dictionary with smoothed properties
    new_model = {
        'vp': vp,
        'vs': vs,
        'rho': rho,
        'x': model['x'],
        'y': model['y'],
        'dx': model['dx'],
        'dy': model['dy']
    }

    return new_model


############################################################
#                   Layer Anomaly Model
############################################################
from scipy.interpolate import interp1d
def step_profile_anomaly(x_range, y_range, step):
    """Generate a stepped profile with anomalies in velocity and density.

    Args:
        x_range (list): The range of x-coordinates (not used in this function).
        y_range (list): The range of y-coordinates defining the depth.
        step (float): The step size for creating the profile.

    Returns:
        tuple: A tuple containing arrays of depth (y_step), 
               primary wave velocity (vp_step), shear wave velocity (vs_step), 
               and density (rho_step).
    """

    # Create the first step profile based on the y_range and step
    y_step1 = np.round(np.arange(y_range[0], y_range[1] + step, step) / step) * step
    
    # Calculate properties for the first step profile
    vp_step1 = y_step1 / (y_range[1] - y_range[0]) * (8.04 - 5.8) + 5.8  # Primary wave velocity
    vs_step1 = y_step1 / (y_range[1] - y_range[0]) * (4.48 - 3.46) + 3.46  # Shear wave velocity
    rho_step1 = y_step1 / (y_range[1] - y_range[0]) * (3.32 - 2.72) + 2.72  # Density

    # Create the second step profile with an adjusted depth
    y_step2 = y_step1 + (y_step1[1] - y_step1[0] - 1)
    vp_step2 = vp_step1  # Keep primary wave velocity same for second step
    vs_step2 = vs_step1  # Keep shear wave velocity same for second step
    rho_step2 = rho_step1  # Keep density same for second step

    # Combine the two step profiles and sort them
    combined_y = np.hstack([y_step1, y_step2])
    combined_vp = np.hstack([vp_step1, vp_step2])
    combined_vs = np.hstack([vs_step1, vs_step2])
    combined_rho = np.hstack([rho_step1, rho_step2])

    # Sort the combined arrays based on depth
    idy = np.argsort(combined_y)  # Sort indices
    y_step = combined_y[idy]  # Combined and sorted depth
    vp_step = combined_vp[idy]  # Combined and sorted primary wave velocity
    vs_step = combined_vs[idy]  # Combined and sorted shear wave velocity
    rho_step = combined_rho[idy]  # Combined and sorted density

    # Ensure the last values are consistent with the second-to-last values
    vp_step[-1] = vp_step[-2]  # Set last vp value to second-to-last
    vs_step[-1] = vs_step[-2]  # Set last vs value to second-to-last
    rho_step[-1] = rho_step[-2]  # Set last rho value to second-to-last
    
    return y_step, vp_step, vs_step, rho_step  # Return the generated profiles


def build_anomaly_background_model(x, y, step):
    """Construct a model with an anomaly background based on step profiles.

    Args:
        x (numpy.ndarray): The x-coordinates for the model.
        y (numpy.ndarray): The y-coordinates (depths) for the model.
        step (float): The step size used to generate the profile.

    Returns:
        dict: A dictionary containing the velocity and density models along with coordinate information.
    """

    # Generate stepped profile with anomalies
    y_step, vp_step, vs_step, rho_step = step_profile_anomaly([x[0], x[-1]], [y[0], y[-1]], step)

    # Interpolate velocities and density values based on the step profile
    vp = interp1d(y_step, vp_step, kind='slinear')(y)  # Interpolated primary wave velocity
    vs = interp1d(y_step, vs_step, kind='slinear')(y)  # Interpolated shear wave velocity
    rho = interp1d(y_step, rho_step, kind='slinear')(y)  # Interpolated density

    # Create 2D arrays for the model parameters by tiling the interpolated values
    vp = np.tile(vp[np.newaxis, :], [len(x), 1])  # Expand vp to 2D
    vs = np.tile(vs[np.newaxis, :], [len(x), 1])  # Expand vs to 2D
    rho = np.tile(rho[np.newaxis, :], [len(x), 1])  # Expand rho to 2D
    
    # Prepare the model dictionary with relevant data
    model = {
        'vp': vp,  # Primary wave velocity
        'vs': vs,  # Shear wave velocity
        'rho': rho,  # Density
        'x': x,  # X-coordinates
        'y': y,  # Y-coordinates (depths)
        'dx': x[1] - x[0],  # X-coordinates spacing
        'dy': y[1] - y[0]   # Y-coordinates spacing
    }
    
    return model  # Return the constructed model



def get_anomaly_model(layer_model, n_pml):
    """Generate a model with anomalies based on the given layer model.

    Args:
        layer_model (dict): The base model containing velocity and density.
        n_pml (int): The number of PML (Perfectly Matched Layer) cells.

    Returns:
        dict: A dictionary containing the modified model with anomalies.
    """
    
    # Extracting coordinates and model parameters
    x = layer_model['x']
    y = layer_model['y']
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    vp = layer_model['vp'].copy()
    vs = layer_model['vs'].copy()
    rho = layer_model['rho'].copy()
    
    # Define anomaly 1 parameters
    x0 = (x[-1] - 1 * n_pml * dx) * 2 / 3 + 0.5 * n_pml * dx
    y0 = (y[-1] - 1 * n_pml * dy) * 1 / 3 + 0.5 * n_pml * dy
    a = x[-1] / 6
    b = y[-1] / 10
    anomaly1 = np.zeros_like(vp)
    
    # Create first anomaly
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            if ((xi - x0) / a) ** 2 + ((yj - y0) / b) ** 2 < 1:
                anomaly1[i, j] = 1
    
    # Define anomaly 2 parameters
    x0 = (x[-1] - 1 * n_pml * dx) / 3 + 0.5 * n_pml * dx
    y0 = (y[-1] - 1 * n_pml * dy) * 2 / 3 + 0.5 * n_pml * dy
    anomaly2 = np.zeros_like(vp)

    # Create second anomaly
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            if ((xi - x0) / a) ** 2 + ((yj - y0) / b) ** 2 < 1:
                anomaly2[i, j] = 1

    # Modify velocities and density based on anomalies
    vp[anomaly1 == 1] = np.mean(vp[anomaly1 == 1]) * 1.1
    vp[anomaly2 == 1] = np.mean(vp[anomaly2 == 1]) / 1.1

    vs[anomaly1 == 1] = np.mean(vs[anomaly1 == 1]) * 1.1
    vs[anomaly2 == 1] = np.mean(vs[anomaly2 == 1]) / 1.1

    rho[anomaly1 == 1] = np.mean(rho[anomaly1 == 1]) * 1.1
    rho[anomaly2 == 1] = np.mean(rho[anomaly2 == 1]) / 1.1
    
    # Prepare the new model with anomalies
    anomaly_model = {
        'vp': vp,
        'vs': vs,
        'rho': rho,
        'x': layer_model['x'],
        'y': layer_model['y'],
        'dx': layer_model['dx'],
        'dy': layer_model['dy']
    }
    
    return anomaly_model  # Return the anomaly model


############################################################
#                   Overthrust model
############################################################
import h5py
import math

def load_overthrust_model(in_dir):
    if not os.path.exists(in_dir):
        os.system("wget {} -P {}".format("https://zenodo.org/records/4252588/files/overthrust_3D_true_model.h5", in_dir))
    h5_data = h5py.File(os.path.join(in_dir,"overthrust_3D_true_model.h5"))
    data_m = np.array(h5_data["m"]).astype(float)
    data_n = np.array(h5_data["n"]).astype(float)
    data_o = np.array(h5_data["o"]).astype(float)
    data_d = np.array(h5_data["d"]).astype(float)
    # slice of the 3D velocity model
    vp = np.sqrt(1/data_m[:,:,120])*1e3
    rho = pow(vp,0.25)*310
    # get the velcoty and dencity
    nx,ny = vp.shape
    overthrust_model = {}
    overthrust_model['vp'] = vp.T
    overthrust_model['rho'] = rho.T
    overthrust_model['x'] = np.arange(ny)*data_d[0]
    overthrust_model['z'] = np.arange(nx)*data_d[1]
    return overthrust_model

def load_overthrust_initial_model(in_dir):
    if not os.path.exists(in_dir):
        os.system("wget {} -P {}".format("https://zenodo.org/records/4252588/files/overthrust_3D_initial_model.h5", in_dir))
    h5_data = h5py.File(os.path.join(in_dir,"overthrust_3D_initial_model.h5"))
    data_m = np.array(h5_data["m0"]).astype(float)
    data_n = np.array(h5_data["n"]).astype(float)
    data_o = np.array(h5_data["o"]).astype(float)
    data_d = np.array(h5_data["d"]).astype(float)
    # slice of the 3D velocity model
    vp = np.sqrt(1/data_m[:,:,120])*1e3
    rho = pow(vp,0.25)*310
    # get the velcoty and dencity
    nx,ny = vp.shape
    overthrust_model = {}
    overthrust_model['vp'] = vp.T
    overthrust_model['rho'] = rho.T
    overthrust_model['x'] = np.arange(ny)*data_d[0]
    overthrust_model['z'] = np.arange(nx)*data_d[1]
    return overthrust_model

def resample_overthrust_model(model):
    vp = model["vp"]
    vp_range = vp[50:450,:200][::2,::2]
    rho_range = pow(vp_range,0.25)*310
    nx,ny = vp_range.shape
    overthrust_model = {}
    overthrust_model['vp']  = vp_range
    overthrust_model['rho'] = rho_range
    overthrust_model['x']   = np.arange(ny)*50
    overthrust_model['y']   = np.arange(nx)*50
    return overthrust_model