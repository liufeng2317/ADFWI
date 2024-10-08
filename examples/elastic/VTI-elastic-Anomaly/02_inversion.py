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
    if not os.path.exists(os.path.join(project_path,"inversion")):
        os.makedirs(os.path.join(project_path,"inversion"))

    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "cuda:0"
    dtype  = torch.float32
    ox,oz  = 0,0
    nz,nx  = 80,180
    dx,dz  = 10, 10
    nt,dt  = 1000,0.001
    nabc   = 50
    f0     = 30
    free_surface = True
    
    # Load the Marmousi model dataset from the specified directory.
    # velocity model 
    vp_true      = np.ones((nz,nx))*3000
    vs_true      = np.ones((nz,nx))*1500
    rho_true     = np.ones((nz,nx))*2450
    epsilon_true = np.ones((nz,nx))*0.1
    gamma_true   = np.ones((nz,nx))*0
    delta_true   = np.ones((nz,nx))*(-0.1)
    # anomaly 0 
    center_x = nx//2
    center_z = nz//2
    center_r = 10
    mask = vp_true == -1
    for i in range(nz):
        for j in range(nx):
            if np.sqrt((i-center_z)**2 + (j-center_x)**2)<center_r:
                mask[i,j] = True
    epsilon_true[mask] = 0.15

    # anomaly 1 
    center_x = nx//4
    center_z = nz//2
    mask = vp_true == -1
    length_square = 20
    mask[center_z-length_square//2:center_z+length_square//2,center_x-length_square//2:center_x+length_square//2] = True
    epsilon_true[mask] = 0.28

    # anomaly 2 
    center_x = 3*nx//4
    center_z = nz//2
    mask = vp_true == -1
    length_square = 20
    mask[center_z-length_square//2:center_z+length_square//2,center_x-length_square//2:center_x+length_square//2+5] = True
    epsilon_true[mask] = 0.2

    # init model
    vp_init      = np.ones((nz,nx))*3000
    vs_init      = np.ones((nz,nx))*1500
    rho_init     = np.ones((nz,nx))*2450
    epsilon_init = np.ones((nz,nx))*0.1
    gamma_init   = np.ones((nz,nx))*0
    delta_init   = np.ones((nz,nx))*(-0.1)
    
    model = AnisotropicElasticModel(
                        ox,oz,nx,nz,dx,dz,
                        vp=vp_init,vs=vs_init,rho=rho_init,
                        eps=epsilon_init,gamma=gamma_init,delta=delta_init,
                        vp_grad=False,vs_grad=False,rho_grad=False,
                        eps_grad=True,gamma_grad=False,delta_grad=False,
                        eps_bound=[epsilon_true.min(),epsilon_true.max()],
                        free_surface=free_surface,
                        anisotropic_type='vti',
                        abc_type="PML",abc_jerjan_alpha=0.007,
                        auto_update_rho=False,
                        auto_update_vp =False,
                        nabc=nabc,
                        device=device,dtype=dtype)
    model.save(os.path.join(project_path,"model/init_model.npz"))
    print(model.__repr__())
        
    model._plot_vp_vs_rho(figsize=(12,5),wspace=0.3,cbar_pad_fraction=0.18,cbar_height=0.04,cmap='coolwarm',save_path=os.path.join(project_path,"model/init_vp_rho.png"),show=False)
    model._plot_eps_delta_gamma(figsize=(12,5),wspace=0.3,cbar_pad_fraction=-0.1,cbar_height=0.04,cmap='coolwarm',save_path=os.path.join(project_path,"model/init_epsilon_gamma_delta.png"),show=False)
    
    #------------------------------------------------------
    #                   Source And Receiver
    #------------------------------------------------------
    # source    
    src_z = np.array([70 for i in range(1,nx-1,5)]) 
    src_x = np.array([i  for i in range(1,nx-1,5)])
    src_t,src_v = wavelet(nt,dt,f0,amp0=1)
    src_v       = integrate.cumtrapz(src_v, axis=-1, initial=0) #Integrate
    source      = Source(nt=nt,dt=dt,f0=f0)
    for i in range(len(src_x)):
        source.add_source(src_x=src_x[i],src_z=src_z[i],src_wavelet=src_v,src_type="mt",src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.plot_wavelet(save_path=os.path.join(project_path,"survey/wavelets_init.png"),show=False)

    # receiver
    rcv_z = np.array([10 for i in range(0,nx,1)])
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
    F = ElasticPropagator(model,survey,device=device)
    if model.abc_type == "PML":
        bcx = F.bcx
        bcz = F.bcz
        title_param = {'family':'Times New Roman','weight':'normal','size': 15}
        plot_bcx_bcz(bcx,bcz,dx=dx,dz=dz,wspace=0.25,title_param=title_param,cbar_height=0.04,cbar_pad_fraction=-0.05,save_path=os.path.join(project_path,"model/boundary_condition_init.png"),show=False)
    else:
        damp = F.damp
        plot_damp(damp)
    
    # load data
    d_obs = SeismicData(survey)
    d_obs.load(os.path.join(project_path,"waveform/obs_data.npz"))
    print(d_obs.__repr__())
    
    from ADFWI.fwi.misfit import Misfit_waveform_L2
    iteration = 500
    # optimizer
    optimizer   =   torch.optim.AdamW(model.parameters(), lr = 0.05,betas=(0.9,0.999), weight_decay=1e-4)
    scheduler   =   torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.75,last_epoch=-1)

    # Setup misfit function
    loss_fn = Misfit_waveform_L2(dt=dt)

    # gradient processor
    gradient_processor = GradProcessor()

    fwi = ElasticFWI(propagator=F,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    obs_data=d_obs,gradient_processor=gradient_processor,
                    waveform_normalize=True,
                    cache_result=True,cache_gradient=True,
                    save_fig_epoch=50,
                    save_fig_path=os.path.join(project_path,"inversion"),
                    inversion_component=["vx","vz"]
                    )

    fwi.forward(iteration=iteration,fd_order=4,
                batch_size=None,checkpoint_segments=4,start_iter=0)
    
    iter_vp     = fwi.iter_vp
    iter_vs     = fwi.iter_vs
    iter_rho    = fwi.iter_rho
    iter_eps    = fwi.iter_eps
    iter_delta  = fwi.iter_delta
    iter_loss   = fwi.iter_loss
    np.savez(os.path.join(project_path,"inversion/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,"inversion/iter_vs.npz"),data=np.array(iter_vs))
    np.savez(os.path.join(project_path,"inversion/iter_rho.npz"),data=np.array(iter_rho))
    np.savez(os.path.join(project_path,"inversion/iter_eps.npz"),data=np.array(iter_eps))
    np.savez(os.path.join(project_path,"inversion/iter_delta.npz"),data=np.array(iter_delta))
    np.savez(os.path.join(project_path,"inversion/iter_loss.npz"),data=np.array(iter_loss))
    
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
    plt.savefig(os.path.join(project_path,"inversion/misfit.png"),bbox_inches='tight',dpi=100)
    plt.close()
    
    # plot the initial model and inverted resutls
    plt.figure(figsize=(12,8))
    plt.subplot(121)
    plt.imshow(epsilon_init,cmap='jet_r')
    plt.subplot(122)
    plt.imshow(iter_eps[-1],cmap='jet_r')
    plt.savefig(os.path.join(project_path,"inversion/inverted_res.png"),bbox_inches='tight',dpi=100)
    plt.close()

    # Set up the figure for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(iter_eps[0], aspect='equal', cmap='jet_r', vmin=vp_true.min(), vmax=vp_true.max())
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
        cax.set_array(iter_eps[0])  # Use the 2D array directly
        return cax,
    # Animation function
    def animate(i):
        cax.set_array(iter_eps[i])  # Update with the i-th iteration directly
        return cax,
    # Create the animation
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(iter_eps), interval=100, blit=True)
    # Save the animation as a video file (e.g., MP4 format)
    ani.save(os.path.join(project_path, "inversion/inversion_process.gif"), writer='pillow', fps=10)
    # Display the animation using HTML
    plt.close(fig)  # Prevents static display of the last frame