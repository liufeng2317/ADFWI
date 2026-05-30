import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import integrate

import ADFWI
from ADFWI.propagator import ElasticPropagator, GradProcessor
from ADFWI.survey import Receiver, SeismicData, Source, Survey
from ADFWI.utils import (
    get_smooth_marmousi_model,
    load_marmousi_model,
    numpy2tensor,
    resample_marmousi_model,
    wavelet
)
from ADFWI.view import animate_inversion_process, plot_initial_and_inverted, plot_misfit
from ADFWI.fwi.misfit import Misfit_global_correlation
from ADFWI.fwi.regularization import regularization_TV_2order
from ADFWI.dip import DIP_CNN, DIP_ElasticFWI, DIP_ElasticModel

import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
if __name__ == "__main__":
    project_path = str(SCRIPT_DIR / "data")
    os.makedirs(os.path.join(project_path,"model"), exist_ok=True)
    os.makedirs(os.path.join(project_path,"waveform"), exist_ok=True)
    os.makedirs(os.path.join(project_path,"survey"), exist_ok=True)
    os.makedirs(os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32"), exist_ok=True)
    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "npu:0"
    dtype  = torch.float32
    backend = ADFWI.set_backend(device, dtype=dtype)
    ox, oz = 0, 0             # Origin coordinates for x and z directions
    nz, nx = 68, 200          # Grid dimensions in z and x directions
    dx, dz = 45, 45           # Grid spacing in x and z directions
    nt, dt = 2500, 0.003      # Time steps and time interval
    nabc = 50                 # Thickness of the absorbing boundary layer
    f0 = 5                    # Initial frequency in Hz
    free_surface = True       # Enable free surface boundary condition
    
    #------------------------------------------------------
    #                   Velocity Model
    #------------------------------------------------------
    # Load the Marmousi model dataset from the specified directory.
    marmousi_model = load_marmousi_model(in_dir=str(next(parent for parent in SCRIPT_DIR.parents if parent.name == "examples") / "datasets" / "marmousi2_source"))

    # Resample the Marmousi model for the defined coordinates
    x = np.linspace(5000, 5000 + dx * nx, nx)
    z = np.linspace(500, 500+dz * nz, nz)
    vel_model = resample_marmousi_model(x, z, marmousi_model)

    vp_true  = vel_model['vp'].T
    vs_true  = vel_model['vs'].T
    rho_true = vel_model['rho'].T

    smooth_model= get_smooth_marmousi_model(vel_model,gaussian_kernel=4,mask_extra_detph=0)
    vp_init     = smooth_model['vp'].T
    vs_init     = smooth_model['vs'].T
    rho_init    = smooth_model['rho'].T

    # -----------------------------------
    #     Define DIP model
    # -----------------------------------
    model_shape  = [nz,nx]
    DIP_model_vp = DIP_CNN(model_shape,in_channels=[32,32],vmin=vp_true.min()/1000 ,vmax=vp_true.max()/1000 )
    DIP_model_vs = DIP_CNN(model_shape,in_channels=[32,32],vmin=vs_true.min()/1000,vmax=vs_true.max()/1000)
    DIP_model_rho= DIP_CNN(model_shape,in_channels=[32,32],vmin=rho_true.min()/1000,vmax=rho_true.max()/1000)
    DIP_model_vp.to(device)
    DIP_model_vs.to(device)
    DIP_model_rho.to(device)

    # -----------------------------------
    #     Pretrain DIP model
    # -----------------------------------
    pretrain        = True
    load_pretrained = False

    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model_vp.load_state_dict(torch.load(os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/DIP_model_vp_pretrained.pt")))
        else:
            lr          = 0.0005
            iteration   = 10000
            step_size   = 1000
            gamma       = 0.5
            optimizer = torch.optim.Adam(DIP_model_vp.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            vp_init = numpy2tensor(vp_init).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                vp_nn = DIP_model_vp()
                loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            torch.save(DIP_model_vp.state_dict(),os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/DIP_model_vp_pretrained.pt"))

    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model_vs.load_state_dict(torch.load(os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/DIP_model_vs_pretrained.pt")))
        else:
            lr          = 0.0005
            iteration   = 10000
            step_size   = 1000
            gamma       = 0.5
            optimizer = torch.optim.Adam(DIP_model_vs.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            vs_init = numpy2tensor(vs_init).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                vs_nn = DIP_model_vs()
                loss = torch.sqrt(torch.sum((vs_nn - vs_init)**2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            torch.save(DIP_model_vs.state_dict(),os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/DIP_model_vs_pretrained.pt"))
    
    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model_rho.load_state_dict(torch.load(os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/DIP_model_rho_pretrained.pt")))
        else:
            lr          = 0.0005
            iteration   = 10000
            step_size   = 1000
            gamma       = 0.5
            optimizer = torch.optim.Adam(DIP_model_rho.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            rho_init = numpy2tensor(rho_init).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                rho_nn = DIP_model_rho()
                loss = torch.sqrt(torch.sum((rho_nn - rho_init)**2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            torch.save(DIP_model_rho.state_dict(),os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/DIP_model_rho_pretrained.pt"))
    
    # -----------------------------------
    #     velocity model for FWI
    # -----------------------------------
    model = DIP_ElasticModel(ox,oz,nx,nz,dx,dz,
                        DIP_model_vp=DIP_model_vp,
                        DIP_model_vs=DIP_model_vs,
                        DIP_model_rho=DIP_model_rho,
                        vp_init=vp_init,
                        vs_init=vs_init,
                        rho_init=rho_init,
                        vp_bound =[vp_true.min(),vp_true.max()],
                        vs_bound =[vs_true.min(),vs_true.max()],
                        rho_bound=[rho_true.min(),rho_true.max()],
                        free_surface=free_surface,
                        abc_type="PML",abc_jerjan_alpha=0.007,nabc=nabc,
                        auto_update_rho=False, auto_update_vp=False
                        )
    print(model.__repr__())
    model.save(os.path.join(project_path,"model/init_model.npz"))
    
    #------------------------------------------------------
    #                   Source And Receiver
    #------------------------------------------------------
    # source    
    src_z = np.array([2 for i in range(2, nx-1, 5)])  # Z-coordinates for sources
    src_x = np.array([i for i in range(2, nx-1, 5)])  # X-coordinates for sources
    src_t,src_v = wavelet(nt,dt,f0,amp0=1)
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0) #Integrate
    source = Source(nt=nt,dt=dt,f0=f0)
    for i in range(len(src_x)):
        source.add_source(src_x=src_x[i],src_z=src_z[i],src_wavelet=src_v,src_type="mt",src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
    source.plot_wavelet(save_path=os.path.join(project_path,"survey/wavelets.png"),show=False)

    # receiver
    rcv_z = np.array([2 for i in range(0, nx, 1)])  # Z-coordinates for receivers
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
    F = ElasticPropagator(model,survey)
    
    # load data
    d_obs = SeismicData(survey)
    d_obs.load(os.path.join(project_path,"waveform/obs_data.npz"))
    print(d_obs.__repr__())
    
    # optimizer
    iteration   =   300
    optimizer   =   torch.optim.Adam(model.parameters(), lr = 0.0001)
    scheduler   =   torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.75,last_epoch=-1)

    # Setup misfit function
    from ADFWI.fwi.misfit import Misfit_global_correlation
    from ADFWI.fwi.regularization import regularization_TV_2order
    loss_fn = Misfit_global_correlation(dt=1)
    regularization_fn = regularization_TV_2order(nx,nz,dx,dz,step_size=50,gamma=0.9)

    # gradient processor
    grad_mask              = np.ones((nz,nx))
    gradient_processor_vp  = GradProcessor(grad_mask=grad_mask,forw_illumination=False,norm_grad=True,grad_smooth=0)
    gradient_processor_vs  = GradProcessor(grad_mask=grad_mask,forw_illumination=False,norm_grad=True,grad_smooth=0)
    gradient_processor_rho = GradProcessor(grad_mask=grad_mask,forw_illumination=False,norm_grad=True,grad_smooth=0)
    gradient_processor = [gradient_processor_vp,gradient_processor_vs,gradient_processor_rho]

    # gradient processor
    fwi = DIP_ElasticFWI(propagator=F,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        regularization_fn=regularization_fn,
                        regularization_weights_x=[0,0,0,0,0,0],
                        regularization_weights_z=[0,0,0,0,0,0],
                        obs_data=d_obs,
                        gradient_processor=gradient_processor,
                        waveform_normalize=True,
                        cache_result=True,
                        save_fig_epoch=10,
                        save_fig_path=os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32"),
                        inversion_component=["vx","vz"]
                        )

    fwi.forward(iteration=iteration,fd_order=4,
                    batch_size=None,checkpoint_segments=7,
                    start_iter=0)
    
    # Retrieve the inversion results: updated velocity and loss values.
    iter_vp     = fwi.iter_vp
    iter_vs     = fwi.iter_vs
    iter_rho    = fwi.iter_rho
    iter_loss   = fwi.iter_loss

    np.savez(os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/iter_vs.npz"),data=np.array(iter_vs))
    np.savez(os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/iter_rho.npz"),data=np.array(iter_rho))
    np.savez(os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/iter_loss.npz"),data=np.array(iter_loss))
    torch.save(model.DIP_model_vp.state_dict(),os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/DIP_model_vp.pt"))
    torch.save(model.DIP_model_vs.state_dict(),os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/DIP_model_vs.pt"))
    torch.save(model.DIP_model_rho.state_dict(),os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/DIP_model_rho.pt"))
    #------------------------------------------------------
    #            Visualize the Inversion Results
    #------------------------------------------------------
    from ADFWI.view.inverted_loss_model import plot_misfit,plot_initial_and_inverted,animate_inversion_process
    
    # misfit
    plot_misfit(iter_loss = iter_loss, save_path=os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/misfit.png"),show=False)
    
    # inverted results
    vp_init = vp_init.cpu().detach().numpy() if torch.is_tensor(vp_init) else vp_init
    vs_init = vs_init.cpu().detach().numpy() if torch.is_tensor(vs_init) else vs_init
    rho_init = rho_init.cpu().detach().numpy() if torch.is_tensor(rho_init) else rho_init
    plot_initial_and_inverted(vp_init=vp_init,iter_vp=iter_vp,save_path=os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/inverted_vp.png"),show=False)
    plot_initial_and_inverted(vp_init=vs_init,iter_vp=iter_vs,save_path=os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/inverted_vs.png"),show=False)
    plot_initial_and_inverted(vp_init=rho_init,iter_vp=iter_rho,save_path=os.path.join(project_path,f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/inverted_rho.png"),show=False)
    
    # inversion animation
    animate_inversion_process(iter_vp=iter_vp,vmin=vp_true.min(),vmax=vp_true.max(),save_path=os.path.join(project_path, f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/inversion_vp.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_vs,vmin=vs_true.min(),vmax=vs_true.max(),save_path=os.path.join(project_path, f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/inversion_vs.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_rho,vmin=rho_true.min(),vmax=rho_true.max(),save_path=os.path.join(project_path, f"no-gradient-smooth/inversion-vp_vs_rho-CNN-2x32/inversion_rho.gif"),fps=10)