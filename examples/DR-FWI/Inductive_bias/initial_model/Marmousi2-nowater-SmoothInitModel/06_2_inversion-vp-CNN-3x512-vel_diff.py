import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")
from scipy import integrate
import sys
import os
sys.path.append("/home/bingxing2/ailab/scxlab0055/project/04_Inversion/ADFWI-github")
from ADFWI.propagator  import *
from ADFWI.model       import *
from ADFWI.view        import *
from ADFWI.utils       import *
from ADFWI.survey      import *
from ADFWI.fwi         import *
from ADFWI.dip         import *
from tqdm import tqdm
torch.cuda.set_device(3)
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    project_path = "./data-shot=40_smooth=8/"
    if not os.path.exists(os.path.join(project_path,"model")):
        os.makedirs(os.path.join(project_path,"model"))
    if not os.path.exists(os.path.join(project_path,"waveform")):
        os.makedirs(os.path.join(project_path,"waveform"))
    if not os.path.exists(os.path.join(project_path,"survey")):
        os.makedirs(os.path.join(project_path,"survey"))
    if not os.path.exists(os.path.join(project_path,"inversion-vp-dv-CNN-3x512")):
        os.makedirs(os.path.join(project_path,"inversion-vp-dv-CNN-3x512"))

    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "cuda:3"
    dtype  = torch.float32
    ox, oz = 0, 0             # Origin coordinates for x and z directions
    nz, nx = 76, 200          # Grid dimensions in z and x directions
    dx, dz = 40, 40           # Grid spacing in x and z directions
    nt, dt = 2500, 0.003      # Time steps and time interval
    nabc = 30                 # Thickness of the absorbing boundary layer
    f0 = 5                    # Initial frequency in Hz
    free_surface = True       # Enable free surface boundary condition
    
    # Load the Marmousi model dataset from the specified directory.
    marmousi_model = load_marmousi_model(in_dir="/home/bingxing2/ailab/scxlab0055/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source")

    # Create coordinate arrays for x and z based on the grid size.
    x = np.linspace(5000, 5000 + dx * nx, nx)
    z = np.linspace(500, 500+dz * nz, nz)
    true_model   = resample_marmousi_model(x, z, marmousi_model)
    smooth_model = get_smooth_marmousi_model(true_model, gaussian_kernel=8,mask_extra_detph=0,rcv_depth=0)

    # Initialize primary wave velocity (vp) and density (rho) for the model.
    vp_init  = smooth_model['vp'].T   # Transpose to match dimensions
    rho_init = smooth_model['rho'].T  # Calculate density based on vp

    # Extract true model properties for comparison.
    vp_true  = true_model['vp'].T   # Transpose for consistency
    rho_true = true_model['rho'].T  # Calculate true density
    
    # -----------------------------------
    #     Define DIP model
    # -----------------------------------
    model_shape = [nz,nx]
    DIP_model_vp = DIP_CNN(model_shape,in_channels=[512,512,512],vmin=-1.5,vmax=1.5,device=device)
    DIP_model_vp.to(device)

    model = DIP_AcousticModel(ox,oz,nx,nz,dx,dz,
                        DIP_model_vp=DIP_model_vp,
                        DIP_model_rho=None,
                        reparameterization_strategy='vel_diff',
                        vp_init=vp_init,rho_init=rho_init,
                        vp_bound =[vp_true.min(),vp_true.max()],
                        rho_bound=[rho_true.min(),rho_true.max()],
                        free_surface=free_surface,
                        abc_type="PML",abc_jerjan_alpha=0.007,nabc=nabc,
                        auto_update_rho=True, auto_update_vp=False,
                        device=device,dtype=dtype)
    
    model.save(os.path.join(project_path,"model/init_model.npz"))
    print(model.__repr__())
        
    model._plot_vp_rho(figsize=(12,5),wspace=0.15,cbar_pad_fraction=0.01,cmap='coolwarm',save_path=os.path.join(project_path,"model/init_vp_rho.png"),show=False)

    #------------------------------------------------------
    #                   Source And Receiver
    #------------------------------------------------------
    # Source
    src_z = np.array([1 for i in range(2, nx-1, 5)]) # Z-coordinates for sources
    src_x = np.array([i for i in range(2, nx-1, 5)])  # X-coordinates for sources
    src_t, src_v = wavelet(nt, dt, f0, amp0=1)  # Create time and wavelet amplitude
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0)  # Integrate wavelet to get velocity
    source = Source(nt=nt, dt=dt, f0=f0)  # Initialize source object
    for i in range(len(src_x)):
        source.add_source(src_x=src_x[i], src_z=src_z[i], src_wavelet=src_v, src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))

    # receiver
    rcv_z = np.array([1 for i in range(0, nx, 1)])  # Z-coordinates for receivers
    rcv_x = np.array([j for j in range(0, nx, 1)])  # X-coordinates for receivers
    receiver = Receiver(nt=nt, dt=dt)  # Initialize receiver object
    for i in range(len(rcv_x)):
        receiver.add_receiver(rcv_x=rcv_x[i], rcv_z=rcv_z[i], rcv_type="pr")
        
    # survey
    survey = Survey(source=source, receiver=receiver)
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
    iteration   =   501
    optimizer   =   torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler   =   torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.75,last_epoch=-1)
        
    # Setup misfit function
    from ADFWI.fwi.misfit import Misfit_global_correlation
    from ADFWI.fwi.regularization import regularization_TV_2order
    loss_fn = Misfit_global_correlation(dt=1)
    regularization_fn = regularization_TV_2order(nx,nz,dx,dz,step_size=50,gamma=0.9,device=device,dtype=dtype)

    # gradient processor
    grad_mask = np.ones((vp_init.shape[0],vp_init.shape[1]))
    gradient_processor = GradProcessor(grad_mask=grad_mask,norm_grad=False,forw_illumination=False)

    # gradient processor
    fwi = DIP_AcousticFWI(propagator=F,
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
                        save_fig_epoch=100,
                        save_fig_path=os.path.join(project_path,f"inversion-vp-dv-CNN-3x512")
                        )

    fwi.forward(iteration=iteration,batch_size=None,checkpoint_segments=2)
    
    iter_vp     = fwi.iter_vp
    iter_rho    = fwi.iter_rho
    iter_loss   = fwi.iter_loss 
    np.savez(os.path.join(project_path,f"inversion-vp-dv-CNN-3x512/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,f"inversion-vp-dv-CNN-3x512/iter_rho.npz"),data=np.array(iter_rho))
    np.savez(os.path.join(project_path,f"inversion-vp-dv-CNN-3x512/iter_loss.npz"),data=np.array(iter_loss))
    torch.save(model.DIP_model_vp.state_dict(),os.path.join(project_path,f"inversion-vp-dv-CNN-3x512/DIP_model.pt"))

    #------------------------------------------------------
    #            Visualize the Inversion Results
    #------------------------------------------------------
    from ADFWI.view.inverted_loss_model import plot_misfit,plot_initial_and_inverted,animate_inversion_process
    
    # misfit
    plot_misfit(iter_loss = iter_loss, save_path=os.path.join(project_path,f"inversion-vp-dv-CNN-3x512/misfit.png"),show=False)
    
    # inverted results
    vp_init = vp_init.cpu().detach().numpy() if torch.is_tensor(vp_init) else vp_init
    plot_initial_and_inverted(vp_init=vp_init,iter_vp=iter_vp,save_path=os.path.join(project_path,f"inversion-vp-dv-CNN-3x512/inverted_vp.png"),show=False)
    
    # inversion animation
    animate_inversion_process(iter_vp=iter_vp,vmin=vp_true.min(),vmax=vp_true.max(),save_path=os.path.join(project_path, f"inversion-vp-dv-CNN-3x512/inversion_vp.gif"),fps=10)