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
    device = "cuda:3"
    dtype  = torch.float32
    ox,oz  = 0,0
    nz,nx  = 78,200
    dx,dz  = 45, 45
    nt,dt  = 2500, 0.003
    nabc   = 50
    f0     = 3
    free_surface = True
    
    # Load the Marmousi model dataset from the specified directory.
    marmousi_model = load_marmousi_model(in_dir="../../datasets/marmousi2_source")
    x         = np.linspace(5000, 5000+dx*nx, nx)
    z         = np.linspace(0, dz*nz, nz)
    vel_model = resample_marmousi_model(x, z, marmousi_model)
    vp_true   = vel_model['vp'].T
    vs_true   = vel_model['vs'].T
    rho_true = np.ones_like(vp_true)*2450

    smooth_model= get_smooth_marmousi_model(vel_model,gaussian_kernel=4,mask_extra_detph=2,rcv_depth=8)
    vp_init     = smooth_model['vp'].T
    vs_init     = smooth_model['vs'].T
    rho_init = np.ones_like(vp_init)*2450
    vp_init[:10]= vp_true[:10]
    vs_init[:10]= vs_init[:10]
    rho_init[:10]=rho_true[:10] 

    water_layer_mask = np.zeros_like(vp_init)
    water_layer_mask[:10] = 1

    # processing the water layer
    model = IsotropicElasticModel(
                    ox,oz,nx,nz,dx,dz,
                    vp_init,vs_init,rho_init,
                    vp_bound =[vp_true.min(),vp_true.max()],
                    vs_bound =[vs_true.min(),vs_true.max()],
                    # rho_bound=[rho_true.min(),rho_true.max()],
                    vp_grad = True,vs_grad = True, rho_grad=False,
                    auto_update_rho=False,auto_update_vp=False,
                    free_surface=free_surface,
                    abc_type="PML",abc_jerjan_alpha=0.007,nabc=nabc,
                    water_layer_mask=water_layer_mask,
                    device=device,dtype=dtype)
    
    model.save(os.path.join(project_path,"model/init_model.npz"))
    print(model.__repr__())
        
    model._plot_vp_vs_rho(figsize=(12,5),wspace=0.2,cbar_pad_fraction=0.18,cbar_height=0.04,cmap='coolwarm',save_path=os.path.join(project_path,"model/init_vp_vs_rho.png"),show=False)

    #------------------------------------------------------
    #                   Source And Receiver
    #------------------------------------------------------
    # source    
    src_z = np.array([2    for i in range(2,nx-2,5)]) 
    src_x = np.array([i    for i in range(2,nx-2,5)])
    src_t,src_v = wavelet(nt,dt,f0,amp0=1)
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0) #Integrate
    source = Source(nt=nt,dt=dt,f0=f0)
    for i in range(len(src_x)):
        source.add_source(src_x=src_x[i], src_z=src_z[i], src_wavelet=src_v, src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.plot_wavelet(save_path=os.path.join(project_path,"survey/wavelets.png"),show=False)

    # receiver
    rcv_z = np.array([2   for i in range(0,nx,1)])
    rcv_x = np.array([j   for j in range(0,nx,1)])
    receiver = Receiver(nt=nt,dt=dt)
    for i in range(len(rcv_x)):
        receiver.add_receiver(rcv_x=rcv_x[i], rcv_z=rcv_z[i], rcv_type="pr")
    # survey
    survey = Survey(source=source,receiver=receiver)
    print(survey.__repr__())
    survey.plot(model.vp,cmap='coolwarm',save_path=os.path.join(project_path,"survey/observed_system_init.png"),show=False)
    
    #------------------------------------------------------
    #                   Inversion
    #------------------------------------------------------
    from ADFWI.fwi.misfit import Misfit_waveform_L2
    from ADFWI.fwi.regularization import regularization_TV_2order

    # Setup misfit function
    loss_fn = Misfit_waveform_L2(dt=dt)
    regularization_fn = regularization_TV_2order(nx,nz,dx,dz,step_size=50,gamma=1,device=device,dtype=dtype)

    # gradient processor
    grad_mask = np.ones_like(vp_init)
    grad_mask[:10] = 0
    gradient_processor_vp = GradProcessor(grad_mask=grad_mask,forw_illumination=False)
    gradient_processor_vs = GradProcessor(grad_mask=grad_mask,forw_illumination=False,grad_mute=10,grad_smooth=2,marine_or_land='marine')
    gradient_processor = [gradient_processor_vp,gradient_processor_vs]

    # Initialize the wave propagator using the specified model and survey configuration
    F = ElasticPropagator(model,survey,device=device)

    # load data
    d_obs = SeismicData(survey)
    d_obs.load(os.path.join(project_path,"waveform/obs_data.npz"))

    # split data into different frequency
    iterations  = [100,100,100]
    freqs       = [2, 3, 5]
    lrs         = [10, 6, 2]
    start_iter = 0
    iter_vp,iter_vs,iter_rho,iter_loss = [],[],[],[]
    for iteration,freq,lr in zip(iterations,freqs,lrs):
        # optimizer
        optimizer   =   torch.optim.Adam(model.parameters(), lr = 10)
        scheduler   =   torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75, last_epoch=-1)

        # Initialize the acoustic full waveform inversion (FWI) object.
        fwi = ElasticFWI(propagator=F,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        regularization_fn=regularization_fn,
                        regularization_weights_x=[1e-6,1e-5,0,0,0,0],
                        regularization_weights_z=[1e-6,1e-5,0,0,0,0],
                        obs_data=d_obs,
                        gradient_processor=gradient_processor,
                        waveform_normalize=True,
                        cache_result=True,
                        save_fig_epoch=20,
                        save_fig_path=os.path.join(project_path,"inversion"),
                        inversion_component=["vx","vz"]
                        )
        
        # Run the forward modeling for the specified number of iterations.
        fwi.forward(iteration=iteration,fd_order=4,
                            batch_size=None,checkpoint_segments=4,
                            start_iter=start_iter,
                            cutoff_freq=freq
                            )
        start_iter += iteration
        
        # Retrieve the inversion results: updated velocity and loss  values.
        iter_vp.extend(fwi.iter_vp)
        iter_vs.extend(fwi.iter_vs)
        iter_rho.extend(fwi.iter_rho)
        iter_loss.extend(fwi.iter_loss)

    # Save the iteration results to files for later analysis.
    np.savez(os.path.join(project_path,"inversion/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,"inversion/iter_vs.npz"),data=np.array(iter_vs))
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
    plot_initial_and_inverted(vp_init=vs_init,iter_vp=iter_vs,save_path=os.path.join(project_path,"inversion/inverted_vs.png"),show=False)
    plot_initial_and_inverted(vp_init=rho_init,iter_vp=iter_rho,save_path=os.path.join(project_path,"inversion/inverted_rho.png"),show=False)
    
    # inversion animation
    animate_inversion_process(iter_vp=iter_vp,vmin=vp_true.min(),vmax=vp_true.max(),save_path=os.path.join(project_path,"inversion/inversion_vp.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_vs,vmin=vs_true.min(),vmax=vs_true.max(),save_path=os.path.join(project_path,"inversion/inversion_vs.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_rho,vmin=rho_true.min(),vmax=rho_true.max(),save_path=os.path.join(project_path,"inversion/inversion_rho.gif"),fps=10)