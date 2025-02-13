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
from ADFWI.dip import *
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    project_path = "./data/"
    layer_num = 4
    if not os.path.exists(os.path.join(project_path,"model")):
        os.makedirs(os.path.join(project_path,"model"))
    if not os.path.exists(os.path.join(project_path,"waveform")):
        os.makedirs(os.path.join(project_path,"waveform"))
    if not os.path.exists(os.path.join(project_path,"survey")):
        os.makedirs(os.path.join(project_path,"survey"))
    if not os.path.exists(os.path.join(project_path,f"inversion-{layer_num}layer-64-32-16-16")):
        os.makedirs(os.path.join(project_path,f"inversion-{layer_num}layer-64-32-16-16"))

    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "cuda:2"
    dtype  = torch.float32
    ox,oz  = 0,0
    nz,nx  = 88,200
    dx,dz  = 40, 40
    nt,dt  = 1600, 0.003
    nabc   = 30
    f0     = 5
    free_surface = True
    
    #------------------------------------------------------
    #                   Velocity Model
    #------------------------------------------------------
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

    # -----------------------------------
    #     Define DIP model
    # -----------------------------------
    model_shape = [nz,nx]
    DIP_model = DIP_CNN(model_shape,in_channels=[64,32,16,16],vmin=vp_true.min()/1000,vmax=vp_true.max()/1000,device=device)
    DIP_model.to(device)

    # -----------------------------------
    #     Pretrain DIP model
    # -----------------------------------
    pretrain        = True
    load_pretrained = False
    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model.load_state_dict(torch.load(os.path.join(project_path,f"inversion-{layer_num}layer-64-32-16-16/DIP_model_pretrained.pt")))
        else:
            lr          = 0.005
            iteration   = 10000
            step_size   = 1000
            gamma       = 0.5
            optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            vp_init = numpy2tensor(vp_init,dtype=dtype).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                vp_nn = DIP_model()
                loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            torch.save(DIP_model.state_dict(),os.path.join(project_path,f"inversion-{layer_num}layer-64-32-16-16/DIP_model_pretrained.pt"))
    
    # -----------------------------------
    #     velocity model for FWI
    # -----------------------------------
    water_layer_mask = np.zeros((vp_init.shape[0],vp_init.shape[1]))
    water_layer_mask[:10,:] = 1
    model = DIP_AcousticModel(ox,oz,nx,nz,dx,dz,
                            DIP_model_vp=DIP_model,
                            DIP_model_rho=None,
                            vp_init=vp_init,rho_init=rho_init,
                            auto_update_rho=True,auto_update_vp=False,
                            water_layer_mask=water_layer_mask,
                            free_surface=free_surface,
                            abc_type="PML",abc_jerjan_alpha=0.007,
                            nabc=nabc,
                            device=device,dtype=dtype)
    print(model.__repr__())
    model.save(os.path.join(project_path,"model/init_model.npz"))
    
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
    optimizer   =   torch.optim.Adam(model.parameters(), lr = 0.001)
    scheduler   =   torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.75,last_epoch=-1)

    # Setup misfit function
    from ADFWI.fwi.misfit import Misfit_global_correlation
    loss_fn = Misfit_global_correlation(dt=1)

    # gradient processor
    grad_mask = np.ones((vp_init.shape[0],vp_init.shape[1]))
    grad_mask[:10,:] = 0
    gradient_processor = GradProcessor(grad_mask=grad_mask)

    # gradient processor
    fwi = DIP_AcousticFWI(propagator=F,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        obs_data=d_obs,
                        gradient_processor=gradient_processor,
                        waveform_normalize=True,
                        cache_result=True,
                        save_fig_epoch=50,
                        save_fig_path=os.path.join(project_path,f"inversion-{layer_num}layer-64-32-16-16")
                        )

    fwi.forward(iteration=iteration,batch_size=None,checkpoint_segments=1)
    
    iter_vp     = fwi.iter_vp
    iter_loss   = fwi.iter_loss 
    np.savez(os.path.join(project_path,f"inversion-{layer_num}layer-64-32-16-16/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,f"inversion-{layer_num}layer-64-32-16-16/iter_loss.npz"),data=np.array(iter_loss))
    torch.save(model.DIP_model_vp.state_dict(),os.path.join(project_path,f"inversion-{layer_num}layer-64-32-16-16/DIP_model.pt"))
    
    #------------------------------------------------------
    #            Visualize the Inversion Results
    #------------------------------------------------------
    from ADFWI.view.inverted_loss_model import plot_misfit,plot_initial_and_inverted,animate_inversion_process
    
    # misfit
    plot_misfit(iter_loss = iter_loss, save_path=os.path.join(project_path,f"inversion-{layer_num}layer-64-32-16-16/misfit.png"),show=False)
    
    # inverted results
    vp_init = vp_init.cpu().detach().numpy() if torch.is_tensor(vp_init) else vp_init
    plot_initial_and_inverted(vp_init=vp_init,iter_vp=iter_vp,save_path=os.path.join(project_path,f"inversion-{layer_num}layer-64-32-16-16/inverted_res.png"),show=False)
    
    # inversion animation
    animate_inversion_process(iter_vp=iter_vp,vmin=vp_true.min(),vmax=vp_true.max(),save_path=os.path.join(project_path, f"inversion-{layer_num}layer-64-32-16-16/inversion_process.gif"),fps=10)