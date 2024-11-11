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
                        eps_bound=[0.1,0.28],
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
    
    # iteration
    iteration   =   300
    optimizer   =   torch.optim.AdamW(model.parameters(), lr = 0.05,betas=(0.9,0.999), weight_decay=1e-4)
    scheduler   =   torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.75,last_epoch=-1)

    # Setup misfit function
    from ADFWI.fwi.misfit import Misfit_waveform_L2
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
                batch_size=None,checkpoint_segments=4,
                start_iter=0)
    
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
    
    #------------------------------------------------------
    #            Visualize the Inversion Results
    #------------------------------------------------------
    from ADFWI.view.inverted_loss_model import plot_misfit,plot_initial_and_inverted,animate_inversion_process
    
    # misfit
    plot_misfit(iter_loss = iter_loss, save_path=os.path.join(project_path,"inversion/misfit.png"),show=False)
    
    # inverted results
    plot_initial_and_inverted(vp_init=epsilon_init  ,iter_vp=iter_eps,save_path=os.path.join(project_path,"inversion/inverted_eps.png"),show=False)

    # inversion animation
    animate_inversion_process(iter_vp=iter_eps  ,vmin=0.1,vmax=0.28,save_path=os.path.join(project_path, "inversion/inversion_eps.gif"),fps=10)