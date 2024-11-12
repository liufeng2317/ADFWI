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
    if not os.path.exists(os.path.join(project_path,"inversion-FD/vs")):
        os.makedirs(os.path.join(project_path,"inversion-FD/vs"))

    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "cuda:2"         # Specify the GPU device
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
    z = np.linspace(12*dz, (dz+12)*nz, nz)
    true_model   = resample_marmousi_model(x, z, marmousi_model)
    smooth_model = get_smooth_marmousi_model(true_model, gaussian_kernel=6,rcv_depth=0,mask_extra_detph=0)

    # Initialize primary wave velocity (vp), Shear wave velocity (vs) and density (rho) for the model.
    vp_init  = smooth_model['vp'].T
    vs_init  = smooth_model['vs'].T
    rho_init = smooth_model['rho'].T

    # Extract true model properties for comparison.
    vp_true = true_model['vp'].T 
    vs_true = true_model['vs'].T 
    rho_true = true_model['rho'].T 

    # processing the water layer
    model = IsotropicElasticModel(
                    ox,oz,nx,nz,dx,dz,
                    vp_init,vs_init,rho_init,
                    vp_grad = False,vs_grad = False, rho_grad=False,
                    auto_update_rho=False,auto_update_vp=False,
                    free_surface=free_surface,
                    abc_type="PML",abc_jerjan_alpha=0.007,nabc=nabc,
                    device=device,dtype=dtype)

    # Save the initialized model to a file for later use.
    model.save(os.path.join(project_path, "model/init_model.npz"))

    # Print the model's representation for verification.
    print(model.__repr__())
        
    model._plot_vp_vs_rho(figsize=(12,5),wspace=0.3,cbar_pad_fraction=0.18,cbar_height=0.04,cmap='coolwarm',save_path=os.path.join(project_path,"model/init_vp_vs_rho.png"),show=False)
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
    # vp
    from tqdm import tqdm
    for i in tqdm(range(nx), desc='Outer loop'):
        for j in tqdm(range(nz), desc='Inner loop', leave=False):
            vs_init_temp = vs_init.copy()
            vs_init_temp[i,j] = vs_init_temp[i,j] + 1
            
            model = IsotropicElasticModel(
                    ox,oz,nx,nz,dx,dz,
                    vp_init,vs_init_temp,rho_init,
                    vp_grad = False,vs_grad = False, rho_grad=False,
                    auto_update_rho=False,auto_update_vp=False,
                    free_surface=free_surface,
                    abc_type="PML",abc_jerjan_alpha=0.007,nabc=nabc,
                    device=device,dtype=dtype)
            
            # Initialize the wave propagator using the specified model and survey configuration
            F = ElasticPropagator(model,survey,device=device)
            
            # Perform the forward propagation to record waveforms
            record_waveform = F.forward(fd_order=4)

            # Extract recorded pressure wavefield and particle velocities
            rcv_txx = record_waveform["txx"]
            rcv_tzz = record_waveform["tzz"]
            # rcv_txz = record_waveform["txz"]
            # rcv_vx  = record_waveform["vx"]
            # rcv_vz  = record_waveform["vz"]

            # Extract forward wavefields for analysis
            # forward_wavefield_txx = record_waveform["forward_wavefield_txx"]
            # forward_wavefield_tzz = record_waveform["forward_wavefield_tzz"]
            # forward_wavefield_txz = record_waveform["forward_wavefield_txz"]
            # forward_wavefield_vx  = record_waveform["forward_wavefield_vx"]
            # forward_wavefield_vz  = record_waveform["forward_wavefield_vz"]
            
            rcv_p = -(rcv_txx+rcv_tzz)
            
            np.savez(os.path.join(project_path, f"inversion-FD/vs/obs_{i}_{j}.npz"),data=rcv_p.cpu().detach().numpy())
            
    perturbation_grad = np.zeros((nx,nz))
    obs_csg_txx = np.load("./data/waveform/obs_data.npz", allow_pickle=True)["data"].item()["txx"]
    obs_csg_tzz = np.load("./data/waveform/obs_data.npz", allow_pickle=True)["data"].item()["tzz"]
    obs_csg_p = -(obs_csg_txx+obs_csg_tzz)

    syn_csg_txx = np.load("./data/waveform/syn_data.npz", allow_pickle=True)["data"].item()["txx"]
    syn_csg_tzz = np.load("./data/waveform/syn_data.npz", allow_pickle=True)["data"].item()["tzz"]
    syn_csg_p = -(syn_csg_txx + syn_csg_tzz)

    for i in tqdm(range(nx)):
        for j in range(nz):
            perturbation_csg = np.load("./data/inversion-FD/vs/obs_{}_{}.npz".format(i,j))["data"]
            perturbation_grad[i][j] = np.sum((syn_csg_p - perturbation_csg)*(syn_csg_p-obs_csg_p))/1

    np.savez(os.path.join(project_path,"inversion-FD/vs/grad_vs.npz"),data = perturbation_grad)