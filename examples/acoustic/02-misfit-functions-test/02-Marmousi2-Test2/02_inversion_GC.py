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
    if not os.path.exists(os.path.join(project_path,"inversion-GC")):
        os.makedirs(os.path.join(project_path,"inversion-GC"))

    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "cuda:1"
    dtype  = torch.float32
    ox,oz  = 0,0
    nz,nx  = 88,200
    dx,dz  = 40, 40
    nt,dt  = 1600, 0.003
    nabc   = 30
    f0     = 5
    free_surface = True
    
    # Load the Marmousi model dataset from the specified directory.
    marmousi_model = load_marmousi_model(in_dir="../../../datasets/marmousi2_source")

    # Create coordinate arrays for x and z based on the grid size.
    x = np.linspace(5000, 5000 + dx * nx, nx)
    z = np.linspace(0, dz * nz, nz)
    true_model = resample_marmousi_model(x, z, marmousi_model)
    smooth_model = get_smooth_marmousi_model(true_model, gaussian_kernel=12)

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
    # source    
    src_z = np.array([1 for i in range(2, nx-1, 5)])  # Z-coordinates for sources
    src_x = np.array([i for i in range(2, nx-1, 5)])  # X-coordinates for sources
    src_t,src_v = wavelet(nt,dt,f0,amp0=1)
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0) #Integrate
    source = Source(nt=nt,dt=dt,f0=f0)
    for i in range(len(src_x)):
        source.add_source(src_x=src_x[i],src_z=src_z[i],src_wavelet=src_v,src_type="mt",src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.plot_wavelet(save_path=os.path.join(project_path,"survey/wavelets.png"),show=False)

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
    from ADFWI.fwi.misfit import Misfit_global_correlation
    loss_fn = Misfit_global_correlation(dt=dt)

    # gradient processor
    grad_mask = np.ones_like(vp_init)
    grad_mask[:10,:] = 0
    gradient_processor = GradProcessor(grad_mask=grad_mask)

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
                    save_fig_path=os.path.join(project_path, "inversion-GC"))

    fwi.forward(iteration=iteration,batch_size=None,checkpoint_segments=1)
    
    iter_vp     = fwi.iter_vp
    iter_loss   = fwi.iter_loss 
    np.savez(os.path.join(project_path,"inversion-GC/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,"inversion-GC/iter_loss.npz"),data=np.array(iter_loss))
    
    ###########################################
    # visualize the inversion results
    ###########################################
    # the animation results
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML
    # plot the misfit
    plt.figure(figsize=(8,6))
    plt.plot(iter_loss,c='k')
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("L2-norm Misfits", fontsize=12)
    plt.tick_params(labelsize=12)
    plt.savefig(os.path.join(project_path,"inversion-GC/misfit.png"),bbox_inches='tight',dpi=100)
    plt.close()
    
    # plot the initial model and inverted resutls
    plt.figure(figsize=(12,8))
    plt.subplot(121)
    plt.imshow(vp_init,cmap='jet_r')
    plt.subplot(122)
    plt.imshow(iter_vp[-1],cmap='jet_r')
    plt.savefig(os.path.join(project_path,"inversion-GC/inverted_res.png"),bbox_inches='tight',dpi=100)
    plt.close()

    # Set up the figure for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(iter_vp[0], aspect='equal', cmap='jet_r', vmin=vp_true.min(), vmax=vp_true.max())
    ax.set_title('Inversion Process Visualization')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    # Create a horizontal colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', fraction=0.046, pad=0.2)
    cbar.set_label('Velocity (m/s)')
    # Adjust the layout to minimize white space
    plt.subplots_adjust(top=0.85, bottom=0.2, left=0.1, right=0.9)
    # Initialization function
    def init():
        cax.set_array(iter_vp[0])  # Use the 2D array directly
        return cax,
    # Animation function
    def animate(i):
        cax.set_array(iter_vp[i])  # Update with the i-th iteration directly
        return cax,
    # Create the animation
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(iter_vp), interval=100, blit=True)
    # Save the animation as a video file (e.g., MP4 format)
    ani.save(os.path.join(project_path, "inversion-GC/inversion_process.gif"), writer='pillow', fps=10)
    # Display the animation using HTML
    plt.close(fig)  # Prevents static display of the last frame