import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")
from scipy import integrate
import sys
import os
sys.path.append("../../../../")
from ADFWI.propagator  import *
from ADFWI.model       import *
from ADFWI.view        import *
from ADFWI.utils       import *
from ADFWI.survey      import *
from ADFWI.fwi         import *

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    project_path = "./data-8source/"
    if not os.path.exists(os.path.join(project_path,"model")):
        os.makedirs(os.path.join(project_path,"model"))
    if not os.path.exists(os.path.join(project_path,"waveform")):
        os.makedirs(os.path.join(project_path,"waveform"))
    if not os.path.exists(os.path.join(project_path,"survey")):
        os.makedirs(os.path.join(project_path,"survey"))
    if not os.path.exists(os.path.join(project_path,"inversion-WECI")):
        os.makedirs(os.path.join(project_path,"inversion-WECI"))

    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "cuda:3"
    dtype  = torch.float32
    ox,oz  = 0,0
    nz,nx  = 88,200
    dx,dz  = 40, 40
    nt,dt  = 3000, 0.003
    nabc   = 30
    f0     = 5
    free_surface = True
    
    # Load the Marmousi model dataset from the specified directory.
    marmousi_model = load_marmousi_model(in_dir="../../../datasets/marmousi2_source")

    # Create coordinate arrays for x and z based on the grid size.
    x = np.linspace(5000, 5000 + dx * nx, nx)
    z = np.linspace(0, dz * nz, nz)
    true_model = resample_marmousi_model(x, z, marmousi_model)
    smooth_model = get_smooth_marmousi_model(true_model, gaussian_kernel=6)

    # Initialize primary wave velocity (vp) and density (rho) for the model.
    vp_init = smooth_model['vp'].T  # Transpose to match dimensions
    rho_init = np.power(vp_init, 0.25) * 310  # Calculate density based on vp

    # Extract true model properties for comparison.
    vp_true = true_model['vp'].T  # Transpose for consistency
    rho_true = np.power(vp_true, 0.25) * 310  # Calculate true density

    model = AcousticModel(ox,oz,nx,nz,dx,dz,
                        vp_init,rho_init,
                        vp_bound=[vp_true.min(),vp_true.max()],
                        vp_grad=True,
                        auto_update_rho=True,
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
    src_z = np.array([1 for i in range(2, nx-1, 5)])  # Z-coordinates for sources
    src_x = np.array([i for i in range(2, nx-1, 5)])  # X-coordinates for sources

    # Generate wavelet for the source
    src_t, src_v = wavelet(nt, dt, f0, amp0=1)  # Create time and wavelet amplitude
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0)  # Integrate wavelet to get velocity

    # Compute time delays
    ref_x, ref_z = 0, 0  # Reference point for delay calculation
    delays = np.sqrt(((src_x - ref_x)*dx)**2 + ((src_z - ref_z)*dz)**2) / 2000  # Time delay for each source

    # Apply time delays and sum sources
    encoded_source = []  # Initialize encoded source
    for i in range(len(src_x)):
        delay_samples = int(delays[i] / dt)  # Convert delay to sample index
        shifted_wavelet = np.zeros(nt)  # Create a zero-padded wavelet
        if delay_samples < nt:
            shifted_wavelet[delay_samples:] = src_v[:nt - delay_samples]  # Apply delay
        encoded_source.append(shifted_wavelet)  # Sum all delayed wavelets

    source = Source(nt=nt, dt=dt, f0=f0)  # Initialize source object

    source.add_encoded_sources(src_x=np.array([src_x[0::4]]),src_z=np.array([src_z[0::4]]),src_wavelet=np.array([encoded_source[0::4]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.add_encoded_sources(src_x=np.array([src_x[1::4]]),src_z=np.array([src_z[1::4]]),src_wavelet=np.array([encoded_source[1::4]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.add_encoded_sources(src_x=np.array([src_x[2::4]]),src_z=np.array([src_z[2::4]]),src_wavelet=np.array([encoded_source[2::4]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.add_encoded_sources(src_x=np.array([src_x[3::4]]),src_z=np.array([src_z[3::4]]),src_wavelet=np.array([encoded_source[3::4]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.add_encoded_sources(src_x=np.array([src_x[0::4]]),src_z=np.array([src_z[0::4]]),src_wavelet=np.array([encoded_source[0::4][::-1]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.add_encoded_sources(src_x=np.array([src_x[1::4]]),src_z=np.array([src_z[1::4]]),src_wavelet=np.array([encoded_source[1::4][::-1]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.add_encoded_sources(src_x=np.array([src_x[2::4]]),src_z=np.array([src_z[2::4]]),src_wavelet=np.array([encoded_source[2::4][::-1]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.add_encoded_sources(src_x=np.array([src_x[3::4]]),src_z=np.array([src_z[3::4]]),src_wavelet=np.array([encoded_source[3::4][::-1]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))

    # receiver
    rcv_z = np.array([1 for i in range(0, nx, 1)])  # Z-coordinates for receivers
    rcv_x = np.array([j for j in range(0, nx, 1)])  # X-coordinates for receivers
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
    F = AcousticPropagator(model,survey,device=device)
    damp = F.damp
    plot_damp(damp,save_path=os.path.join(project_path,"model/boundary_condition_init.png"),show=False)
    
    # load data
    d_obs = SeismicData(survey)
    d_obs.load(os.path.join(project_path,"waveform/obs_data.npz"))
    print(d_obs.__repr__())
    
    # optimizer
    iteration   =   300
    optimizer   =   torch.optim.Adam(model.parameters(), lr=10)
    scheduler   =   torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.75,last_epoch=-1)

    # Setup misfit function
    from ADFWI.fwi.misfit import Misfit_weighted_ECI
    loss_fn = Misfit_weighted_ECI(dt=1,p=1.5)

    # gradient processor
    grad_mask = np.ones_like(vp_init)
    grad_mask[:10,:] = 0
    gradient_processor = GradProcessor(grad_mask=grad_mask,forw_illumination=False,norm_grad=True)

    # Initialize the acoustic full waveform inversion (FWI) object.
    fwi = AcousticFWI(propagator=F,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    obs_data=d_obs,
                    gradient_processor=gradient_processor,
                    waveform_normalize=True, 
                    cache_result=True,  
                    save_fig_epoch=50,  
                    save_fig_path=os.path.join(project_path, "inversion-WECI"))

    fwi.forward(iteration=iteration,batch_size=None,checkpoint_segments=1)
    
    iter_vp     = fwi.iter_vp
    iter_loss   = fwi.iter_loss 
    np.savez(os.path.join(project_path,"inversion-WECI/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,"inversion-WECI/iter_loss.npz"),data=np.array(iter_loss))
    
    #------------------------------------------------------
    #            Visualize the Inversion Results
    #------------------------------------------------------
    from ADFWI.view.inverted_loss_model import plot_misfit,plot_initial_and_inverted,animate_inversion_process
    
    # misfit
    plot_misfit(iter_loss = iter_loss, save_path=os.path.join(project_path,"inversion-WECI/misfit.png"),show=False)
    
    # inverted results
    plot_initial_and_inverted(vp_init=vp_init,iter_vp=iter_vp,save_path=os.path.join(project_path,"inversion-WECI/inverted_res.png"),show=False)
    
    # inversion animation
    animate_inversion_process(iter_vp=iter_vp,vmin=vp_true.min(),vmax=vp_true.max(),save_path=os.path.join(project_path, "inversion-WECI/inversion_process.gif"),fps=10)