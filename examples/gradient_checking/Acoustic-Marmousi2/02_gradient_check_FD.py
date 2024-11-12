import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")
from scipy import integrate
import sys
import os
sys.path.append("../../../")
from ADFWI.propagator  import *
from ADFWI.model       import *
from ADFWI.view        import *
from ADFWI.utils       import *
from ADFWI.survey      import *
from ADFWI.fwi         import *

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    project_path = "./data/"
    if not os.path.exists(os.path.join(project_path,"model")):
        os.makedirs(os.path.join(project_path,"model"))
    if not os.path.exists(os.path.join(project_path,"waveform")):
        os.makedirs(os.path.join(project_path,"waveform"))
    if not os.path.exists(os.path.join(project_path,"survey")):
        os.makedirs(os.path.join(project_path,"survey"))
    if not os.path.exists(os.path.join(project_path,"inversion-FD")):
        os.makedirs(os.path.join(project_path,"inversion-FD"))

    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "cuda:0"         # Specify the GPU device
    dtype  = torch.float32     # Set data type to 32-bit floating point
    ox, oz = 0, 0             # Origin coordinates for x and z directions
    nz, nx = 50, 50          # Grid dimensions in z and x directions
    dx, dz = 40, 40           # Grid spacing in x and z directions
    nt, dt = 1600, 0.003      # Time steps and time interval
    nabc = 30                 # Thickness of the absorbing boundary layer
    f0 = 5                    # Initial frequency in Hz
    free_surface = True       # Enable free surface boundary condition

    
    # Load the Marmousi model dataset from the specified directory.
    marmousi_model = load_marmousi_model(in_dir="../../datasets/marmousi2_source")

    # Create coordinate arrays for x and z based on the grid size.
    x = np.linspace(7400, 7400+dx*nx, nx)
    z = np.linspace(11*dz, (dz+11)*nz, nz)
    true_model   = resample_marmousi_model(x, z, marmousi_model)
    smooth_model = get_smooth_marmousi_model(true_model, gaussian_kernel=6,rcv_depth=0,mask_extra_detph=0)


    # Initialize primary wave velocity (vp) and density (rho) for the model.
    vp_init = smooth_model['vp'].T  # Transpose to match dimensions
    rho_init = np.power(vp_init, 0.25) * 310  # Calculate density based on vp

    # Extract true model properties for comparison.
    vp_true = true_model['vp'].T  # Transpose for consistency
    rho_true = np.power(vp_true, 0.25) * 310  # Calculate true density

    model = AcousticModel(ox,oz,nx,nz,dx,dz,
                        vp_init,rho_init,
                        vp_bound=[vp_true.min(),vp_true.max()],
                        vp_grad=False,
                        free_surface=free_surface,
                        abc_type="PML",abc_jerjan_alpha=0.007,
                        nabc=nabc,
                        device=device,dtype=dtype)
    
    model.save(os.path.join(project_path,"model/init_model.npz"))
    print(model.__repr__())
        
    model._plot_vp_rho(figsize=(12,5),wspace=0.15,cbar_pad_fraction=0.02,cbar_height=0.04,cmap='coolwarm',save_path=os.path.join(project_path,"model/init_vp_rho.png"),show=False)

    #------------------------------------------------------
    #                   Source And Receiver
    #------------------------------------------------------
    # Define source positions in the model
    src_z = np.array([45]) 
    src_x = np.array([25])
    src_t,src_v = wavelet(nt,dt,f0,amp0=1)
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0) #Integrate
    source = Source(nt=nt,dt=dt,f0=f0)
    for i in range(len(src_x)):
        source.add_source(src_x=src_x[i],src_z=src_z[i],src_wavelet=src_v,src_type="mt",src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.plot_wavelet(save_path=os.path.join(project_path,"survey/wavelets.png"),show=False)

    # Define receiver positions in the model
    rcv_z = np.array([1  for i in range(0,nx,1)])
    rcv_x = np.array([j  for j in range(0,nx,1)])
    receiver = Receiver(nt=nt,dt=dt)
    for i in range(len(rcv_x)):
        receiver.add_receiver(rcv_x=rcv_x[i],rcv_z=rcv_z[i],rcv_type="pr")
    
    # survey
    survey = Survey(source=source,receiver=receiver)
    print(survey.__repr__())
    survey.plot(model.vp,cmap='coolwarm',save_path=os.path.join(project_path,"survey/observed_system_init.png"),show=False)
    
    #------------------------------------------------------
    #                   Waveform Propagator
    #------------------------------------------------------
    from tqdm import tqdm
    for i in tqdm(range(nx), desc='Outer loop'):
        for j in tqdm(range(nz), desc='Inner loop', leave=False):
            vp_init_temp = vp_init.copy()
            vp_init_temp[i,j] = vp_init_temp[i,j] + 1
            model = AcousticModel(ox,oz,nx,nz,dx,dz,
                        vp_init_temp,rho_init,
                        vp_bound=[vp_true.min(),vp_true.max()],
                        vp_grad=False,
                        free_surface=free_surface,
                        abc_type="PML",abc_jerjan_alpha=0.007,
                        nabc=nabc,
                        device=device,dtype=dtype)
            
            # Initialize the wave propagator using the specified model and survey configuration
            F = AcousticPropagator(model, survey, device=device)
            
            # Perform the forward propagation to record waveforms
            record_waveform = F.forward()

            # Extract recorded pressure wavefield and particle velocities
            rcv_p = record_waveform["p"]  # Recorded pressure wavefield
            # rcv_u = record_waveform["u"]  # Recorded particle velocity in x-direction
            # rcv_w = record_waveform["w"]  # Recorded particle velocity in z-direction

            # Extract forward wavefields for analysis
            # forward_wavefield_p = record_waveform["forward_wavefield_p"]  # Forward pressure wavefield
            # forward_wavefield_u = record_waveform["forward_wavefield_u"]  # Forward particle velocity wavefield in x
            # forward_wavefield_w = record_waveform["forward_wavefield_w"]  # Forward particle velocity wavefield in z
            
            # Create a SeismicData object to store observed data from the survey
            # d_obs = SeismicData(survey)

            # # Record the waveform data into the SeismicData object
            # d_obs.record_data(record_waveform)

            # Save the recorded data to a specified file
            # d_obs.save(os.path.join(project_path, f"inversion-FD/obs_{i}_{j}.npz"))
            
            np.savez(os.path.join(project_path, f"inversion-FD/obs_{i}_{j}.npz"),data=rcv_p.cpu().detach().numpy())
            
    perturbation_grad = np.zeros((nx,nz))
    obs_csg = np.load("./data/waveform/obs_data.npz", allow_pickle=True)["data"].item()["p"]
    syn_csg = np.load("./data/waveform/syn_data.npz", allow_pickle=True)["data"].item()["p"]

    for i in tqdm(range(nx)):
        for j in range(nz):
            perturbation_csg = np.load("./data/inversion-FD/obs_{}_{}.npz".format(i,j))["data"]
            perturbation_grad[i][j] = np.sum((syn_csg - perturbation_csg)*(syn_csg-obs_csg))/30

    np.savez(os.path.join(project_path,"inversion-FD/grad_vp.npz"),data = perturbation_grad)