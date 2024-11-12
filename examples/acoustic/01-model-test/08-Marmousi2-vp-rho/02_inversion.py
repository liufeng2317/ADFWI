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
    project_path = "./data/"
    if not os.path.exists(os.path.join(project_path,"model")):
        os.makedirs(os.path.join(project_path,"model"))
    if not os.path.exists(os.path.join(project_path,"waveform")):
        os.makedirs(os.path.join(project_path,"waveform"))
    if not os.path.exists(os.path.join(project_path,"survey")):
        os.makedirs(os.path.join(project_path,"survey"))
    if not os.path.exists(os.path.join(project_path,"inversion")):
        os.makedirs(os.path.join(project_path,"inversion"))

    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "cuda:7"         # Specify the GPU device
    dtype = torch.float32     # Set data type to 32-bit floating point
    ox, oz = 0, 0             # Origin coordinates for x and z directions
    nz, nx = 88, 200          # Grid dimensions in z and x directions
    dx, dz = 40, 40           # Grid spacing in x and z directions
    nt, dt = 1600, 0.003      # Time steps and time interval
    nabc = 30                 # Thickness of the absorbing boundary layer
    f0 = 5                    # Initial frequency in Hz
    free_surface = True       # Enable free surface boundary condition
        
    # Load the Marmousi model dataset from the specified directory.
    marmousi_model = load_marmousi_model(in_dir="../../../datasets/marmousi2_source")

    # Create coordinate arrays for x and z based on the grid size.
    x = np.linspace(5000, 5000 + dx * nx, nx)
    z = np.linspace(0, dz * nz, nz)
    true_model   = resample_marmousi_model(x, z, marmousi_model)
    smooth_model = get_smooth_marmousi_model(true_model, gaussian_kernel=6)

    # Extract true model properties for comparison.
    vp_true  = true_model['vp'].T  # Transpose for consistency
    rho_true = true_model['rho'].T  # Calculate true density

    # Initialize primary wave velocity (vp) and density (rho) for the model.
    vp_init  = smooth_model['vp'].T   # Transpose to match dimensions
    rho_init = smooth_model['rho'].T  # Calculate density based on vp

    water_layer_mask =np.zeros_like(vp_init)
    water_layer_mask[:12] = 1 

    model = AcousticModel(ox,oz,nx,nz,dx,dz,
                        vp_init,rho_init,
                        vp_bound =[vp_true.min().min(),vp_true.max()],
                        rho_bound=[rho_true[12:,:].min(),rho_true[12:,:].max()],
                        vp_grad=True,rho_grad=True,
                        free_surface=free_surface,
                        abc_type="PML",abc_jerjan_alpha=0.007,
                        nabc=nabc,
                        auto_update_rho=False,
                        # water_layer_mask=water_layer_mask,
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
    src_t,src_v = wavelet(nt,dt,f0,amp0=1)
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0) #Integrate
    source = Source(nt=nt,dt=dt,f0=f0)
    for i in range(len(src_x)):
        source.add_source(src_x=src_x[i],src_z=src_z[i],src_wavelet=src_v,src_type="mt",src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.plot_wavelet(save_path=os.path.join(project_path,"survey/wavelets.png"),show=False)

    # Define receiver positions in the model
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
    # load data
    d_obs = SeismicData(survey)
    d_obs.load(os.path.join(project_path,"waveform/obs_data.npz"))
    print(d_obs.__repr__())
    
    # Import the L2 misfit function for waveform inversion.
    from ADFWI.fwi.misfit import Misfit_waveform_L2
    from ADFWI.fwi.regularization import regularization_TV_2order
        
    iterations  = [10, 100, 20, 100, 30, 100, 40, 100, 100]
    vp_grads    = [True,True,True,True,True,True,True,True,False]
    rho_grads   = [True,False,True,False,True,False,True,False,True]
    lrs         = [1,10,1,5,1,3,1,2,0.5]

    start_iter = 0
    iter_vp,iter_rho,iter_loss = [],[],[]

    for iteration,vp_grad,rho_grad,lr in zip(iterations,vp_grads,rho_grads,lrs):
        model.vp_grad = vp_grad
        model.rho_grad = rho_grad
        model.requires_grad["vp"] = vp_grad
        model.requires_grad["rho"] = rho_grad
        model._parameterization()
        
        # Initialize the wave propagator using the specified model and survey configuration
        F = AcousticPropagator(model, survey, device=device)
        
        # Initialize the optimizer (Adam) for model parameters with a learning rate.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Set up a learning rate scheduler to adjust the learning rate over time.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75, last_epoch=-1)

        # Configure the misfit function to compute the loss based on the observed data.
        loss_fn = Misfit_waveform_L2(dt=dt)
        regularization_fn = regularization_TV_2order(nx,nz,dx,dz,step_size=50,gamma=0.9,device=device,dtype=dtype)

        # gradient processor
        grad_mask = np.ones_like(vp_init)
        grad_mask[:12,:] = 0
        gradient_processor = GradProcessor(grad_mask=grad_mask,forw_illumination=False)
        
        # Initialize the acoustic full waveform inversion (FWI) object.
        fwi = AcousticFWI(propagator=F,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        regularization_fn=regularization_fn,
                        regularization_weights_x=[0,0],
                        regularization_weights_z=[0,0],
                        obs_data=d_obs,
                        gradient_processor=gradient_processor,
                        waveform_normalize=True, 
                        cache_result=True,  
                        save_fig_epoch=50,  
                        save_fig_path=os.path.join(project_path, "inversion"))

        # Run the forward modeling for the specified number of iterations.
        fwi.forward(iteration=iteration, batch_size=None, checkpoint_segments=2,start_iter=start_iter)
        start_iter += iteration
        # Retrieve the inversion results: updated velocity and loss  values.
        iter_vp.extend(fwi.iter_vp)
        iter_rho.extend(fwi.iter_rho)
        iter_loss.extend(fwi.iter_loss)
 
    np.savez(os.path.join(project_path,"inversion/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,"inversion/iter_rho.npz"),data=np.array(iter_rho))
    np.savez(os.path.join(project_path,"inversion/iter_loss.npz"),data=np.array(iter_loss))

    #------------------------------------------------------
    #            Visualize the Inversion Results
    #------------------------------------------------------
    from ADFWI.view.inverted_loss_model import plot_misfit,plot_initial_and_inverted,animate_inversion_process
    
    # misfit
    plot_misfit(iter_loss = iter_loss, save_path=os.path.join(project_path,"inversion/misfit.png"),show=False)
    
    # inverted results
    plot_initial_and_inverted(vp_init=vp_init,iter_vp=iter_vp,save_path=os.path.join(project_path,"inversion/inverted_vp.png"),show=False)
    plot_initial_and_inverted(vp_init=rho_init,iter_vp=iter_rho,save_path=os.path.join(project_path,"inversion/inverted_rho.png"),show=False)
    
    # inversion animation
    animate_inversion_process(iter_vp=iter_vp,vmin=vp_true.min(),vmax=vp_true.max(),save_path=os.path.join(project_path,"inversion/inversion_vp_process.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_rho,vmin=rho_true.min(),vmax=rho_true.max(),save_path=os.path.join(project_path,"inversion/inversion_rho_process.gif"),fps=10)