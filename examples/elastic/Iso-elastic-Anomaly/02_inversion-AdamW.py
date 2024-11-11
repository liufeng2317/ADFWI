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
    if not os.path.exists(os.path.join(project_path,"inversion-AdamW")):
        os.makedirs(os.path.join(project_path,"inversion-AdamW"))

    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "cuda:1"
    dtype = torch.float32     # Set data type to 32-bit floating point
    ox, oz = 0, 0             # Origin coordinates for x and z directions
    nz, nx = 80, 180          # Grid dimensions in z and x directions
    dx, dz = 10, 10           # Grid spacing in x and z directions
    nt, dt = 1000,0.001       # Time steps and time interval
    nabc   = 50                 # Thickness of the absorbing boundary layer
    f0     = 25                   # Initial frequency in Hz
    free_surface = True       # Enable free surface boundary condition
    
    # Anomaly Model
    vp_true      = np.ones((nz,nx))*3000
    vs_true      = np.ones((nz,nx))*2000
    rho_true     = np.ones((nz,nx))*2450
    
    # anomaly 0 
    center_x = nx//3
    center_z = nz//2
    center_r = 10
    mask = vp_true == -1
    for i in range(nz):
        for j in range(nx):
            if np.sqrt((i-center_z)**2 + (j-center_x)**2)<center_r:
                mask[i,j] = True
    vp_true[mask] *= 1.1  # Increase vp by 5%

    # anomaly 1
    center_x = nx//3*2
    center_z = nz//2
    center_r = 10
    mask = vp_true == -1
    for i in range(nz):
        for j in range(nx):
            if np.sqrt((i-center_z)**2 + (j-center_x)**2)<center_r:
                mask[i,j] = True
    vs_true[mask] *= 1.1  # Increase vs by 5%

    # init model
    vp_init      = np.ones((nz,nx))*3000
    vs_init      = np.ones((nz,nx))*2000
    rho_init     = np.ones((nz,nx))*2450

    # processing the water layer
    model = IsotropicElasticModel(
                    ox,oz,nx,nz,dx,dz,
                    vp_init,vs_init,rho_init,
                    vp_bound =[vp_true.min(),vp_true.max()],
                    vs_bound =[vs_true.min(),vs_true.max()],
                    vp_grad = True, vs_grad = True, rho_grad=False,
                    auto_update_rho=False,auto_update_vp=False,
                    free_surface=free_surface,
                    abc_type="PML",abc_jerjan_alpha=0.007,nabc=nabc,
                    device=device,dtype=dtype)
    
    model.save(os.path.join(project_path,"model/init_model.npz"))
    print(model.__repr__())
        
    model._plot_vp_vs_rho(figsize=(12,5),wspace=0.2,cbar_pad_fraction=0.18,cbar_height=0.04,cmap='coolwarm',save_path=os.path.join(project_path,"model/init_vp_vs_rho.png"),show=False)

    #------------------------------------------------------
    #                   Source And Receiver
    #------------------------------------------------------
    # Define source positions in the model
    src_z = np.array([5 for i in range(1,nx-1,5)]) 
    src_x = np.array([i  for i in range(1,nx-1,5)])
    src_t,src_v = wavelet(nt,dt,f0,amp0=1)
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0) #Integrate
    source = Source(nt=nt,dt=dt,f0=f0)
    for i in range(len(src_x)):
        source.add_source(src_x=src_x[i],src_z=src_z[i],src_wavelet=src_v,src_type="mt",src_mt=np.array([[0,0,1],[0,0,0],[0,0,0]]))
        # source.add_source(src_x=src_x[i],src_z=src_z[i],src_wavelet=src_v,src_type="mt",src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.plot_wavelet(save_path=os.path.join(project_path,"survey/wavelets.png"),show=False)

    # Define receiver positions in the model
    rcv_z = np.array([5 for i in range(0,nx,1)])
    rcv_x = np.array([j  for j in range(0,nx,1)])
    receiver = Receiver(nt=nt,dt=dt)
    for i in range(len(rcv_x)):
        receiver.add_receiver(rcv_x=rcv_x[i], rcv_z=rcv_z[i], rcv_type="pr")
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

    iteration = 300

    # optimizer
    optimizer   =   torch.optim.AdamW(model.parameters(), lr = 10,betas=(0.9,0.999), weight_decay=1e-4)
    # optimizer   =   torch.optim.Adam(model.parameters(), lr = 10)
    scheduler   =   torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.75,last_epoch=-1)

    # Setup misfit function
    loss_fn = Misfit_waveform_L2(dt=dt)

    # gradient processor
    gradient_processor = GradProcessor(forw_illumination=False)

    fwi = ElasticFWI(propagator=F,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    obs_data=d_obs,gradient_processor=gradient_processor,
                    waveform_normalize=True,
                    cache_result=True,cache_gradient=True,
                    save_fig_epoch=10,
                    save_fig_path=os.path.join(project_path,"inversion-AdamW"),
                    # inversion_component=["pressure"]
                    inversion_component=["vx","vz"]
                    )

    fwi.forward(iteration=iteration,fd_order=4,
                batch_size=None,checkpoint_segments=1,
                start_iter=0)
    
    iter_vp     = fwi.iter_vp
    iter_vs     = fwi.iter_vs
    iter_rho    = fwi.iter_rho
    iter_loss   = fwi.iter_loss
    np.savez(os.path.join(project_path,"inversion-AdamW/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,"inversion-AdamW/iter_vs.npz"),data=np.array(iter_vs))
    np.savez(os.path.join(project_path,"inversion-AdamW/iter_rho.npz"),data=np.array(iter_rho))
    np.savez(os.path.join(project_path,"inversion-AdamW/iter_loss.npz"),data=np.array(iter_loss))
    
    #------------------------------------------------------
    #            Visualize the Inversion Results
    #------------------------------------------------------
    from ADFWI.view.inverted_loss_model import plot_misfit,plot_initial_and_inverted,animate_inversion_process
    
    # misfit
    plot_misfit(iter_loss = iter_loss, save_path=os.path.join(project_path,"inversion-AdamW/misfit.png"),show=False)
    
    # inverted results
    plot_initial_and_inverted(vp_init=vp_init,iter_vp=iter_vp,save_path=os.path.join(project_path,"inversion-AdamW/inverted_vp.png"),show=False)
    plot_initial_and_inverted(vp_init=vs_init,iter_vp=iter_vs,save_path=os.path.join(project_path,"inversion-AdamW/inverted_vs.png"),show=False)
    plot_initial_and_inverted(vp_init=rho_init,iter_vp=iter_rho,save_path=os.path.join(project_path,"inversion-AdamW/inverted_rho.png"),show=False)
    
    # inversion animation
    animate_inversion_process(iter_vp=iter_vp,vmin=vp_true.min(),vmax=vp_true.max(),save_path=os.path.join(project_path,"inversion-AdamW/inversion_vp.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_vs,vmin=vs_true.min(),vmax=vs_true.max(),save_path=os.path.join(project_path,"inversion-AdamW/inversion_vs.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_rho,vmin=rho_true.min(),vmax=rho_true.max(),save_path=os.path.join(project_path,"inversion-AdamW/inversion_rho.gif"),fps=10)