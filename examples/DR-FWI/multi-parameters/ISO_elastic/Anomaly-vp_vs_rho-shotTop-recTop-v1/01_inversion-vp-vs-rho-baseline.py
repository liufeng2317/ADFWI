import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
matplotlib.use("agg")
from scipy import integrate

import ADFWI
from ADFWI.model import IsotropicElasticModel
from ADFWI.propagator import ElasticPropagator, GradProcessor
from ADFWI.survey import Receiver, SeismicData, Source, Survey
from ADFWI.utils import wavelet
from ADFWI.view import animate_inversion_process, plot_initial_and_inverted, plot_misfit
from ADFWI.fwi import ElasticFWI
from ADFWI.fwi.misfit import Misfit_global_correlation
from ADFWI.fwi.regularization import regularization_TV_2order

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
    os.makedirs(os.path.join(project_path,f"inversion-vp_vs_rho-baseline"), exist_ok=True)
    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "npu:0"         # Specify the GPU device
    dtype = torch.float32     # Set data type to 32-bit floating point
    backend = ADFWI.set_backend(device, dtype=dtype)
    ox, oz = 0, 0             # Origin coordinates for x and z directions
    nz, nx = 100, 200         # Grid dimensions in z and x directions
    dx, dz = 30, 30           # Grid spacing in x and z directions
    nt, dt = 2500, 0.0025     # Time steps and time interval
    nabc = 30                 # Thickness of the absorbing boundary layer
    f0 = 5                    # Initial frequency in Hz
    free_surface = True       # Enable free surface boundary condition

    #------------------------------------------------------
    #                   Velocity Model
    #------------------------------------------------------
    def create_anomaly_model(nz, nx):
        """
        Create a 2-layer velocity model with multiple different-shaped anomalies in the first layer.
        """
        # Initialize first layer
        vp = np.ones((nz, nx)) * 1500
        vs = np.ones((nz, nx)) * 866
        rho = np.ones((nz, nx)) * 1000
        
        # Define second layer properties
        vp[80:, :]  = 3000
        vs[80:, :]  = 1732
        rho[80:, :] = 2600
        
        vp_init = vp.copy()
        vs_init = vs.copy()
        rho_init = rho.copy()
        
        # Define anomaly properties
        vp_anom, vs_anom, rho_anom = 2000, 1155, 1600
        
        # Add multiple circular anomalies in vp
        circle_positions = [(30, 30), (70, 50), (100, 30), (140, 50), (180, 35)]
        circle_radius = 10
        for cx, cy in circle_positions:
            for i in range(-circle_radius, circle_radius + 1):
                for j in range(-circle_radius, circle_radius + 1):
                    if i**2 + j**2 <= circle_radius**2:
                        if 0 <= cy + i < nz and 0 <= cx + j < nx:
                            vp[cy + i, cx + j] = vp_anom
        
        # Add multiple square anomalies in vs
        square_positions  = [(20, 25), (50, 50), (80, 30), (120, 50), (170, 40)]
        square_size = 20
        for x, y in square_positions:
            vs[y:y+square_size, x:x+square_size] = vs_anom
        
        # Add multiple triangular anomalies in rho
        triangle_positions = [(25, 25), (60, 35), (110, 40), (150, 25), (180, 50)]
        tri_size = 15
        for tri_x, tri_y in triangle_positions:
            for i in range(tri_size):
                rho[tri_y + i, tri_x - i:tri_x + i + 1] = rho_anom
        
        return vp, vs, rho, vp_init,vs_init,rho_init

    vp_true,vs_true,rho_true,vp_init,vs_init,rho_init = create_anomaly_model(nz, nx)

    # processing the water layer
    model = IsotropicElasticModel(
                    ox,oz,nx,nz,dx,dz,
                    vp_init,vs_init,rho_init,
                    vp_bound =[vp_true.min(),vp_true.max()],
                    vs_bound =[vs_true.min(),vs_true.max()],
                    rho_bound=[rho_true.min(),rho_true.max()],
                    vp_grad = True, vs_grad = True, rho_grad=True,
                    auto_update_rho=False, auto_update_vp=False,
                    free_surface=free_surface,
                    abc_type="PML",abc_jerjan_alpha=0.007,nabc=nabc
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
    # Initialize the wave propagator using the specified model and survey configuration
    F = ElasticPropagator(model,survey)

    # load data
    d_obs = SeismicData(survey)
    d_obs.load(os.path.join(project_path,"waveform/obs_data.npz"))
    print(d_obs.__repr__())
        
    # optimizer
    iteration   =   1000
    optimizer   =   torch.optim.Adam(model.parameters(), lr = 5)
    # optimizer   =   torch.optim.SGD(model.parameters(), lr = 0.01)
    scheduler   =   torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.75,last_epoch=-1)

    # Setup misfit function
    from ADFWI.fwi.misfit import Misfit_global_correlation
    from ADFWI.fwi.regularization import regularization_TV_2order
    loss_fn = Misfit_global_correlation(dt=1)
    regularization_fn = regularization_TV_2order(nx,nz,dx,dz,step_size=50,gamma=0.9)

    # gradient processor
    grad_mask             = np.ones((nz,nx))
    gradient_processor_vp = GradProcessor(grad_mask=grad_mask,forw_illumination=False,norm_grad=True,grad_smooth=0)
    gradient_processor_vs = GradProcessor(grad_mask=grad_mask,forw_illumination=False,norm_grad=True,grad_smooth=0)
    gradient_processor_rho= GradProcessor(grad_mask=grad_mask,forw_illumination=False,norm_grad=True,grad_smooth=0)
    gradient_processor = [gradient_processor_vp,gradient_processor_vs,gradient_processor_rho]

    # Initialize the acoustic full waveform inversion (FWI) object.
    fwi = ElasticFWI(propagator=F,
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
                        save_fig_path=os.path.join(project_path,f"inversion-vp_vs_rho-baseline"),
                        inversion_component=["vx","vz"]
                        )

    # Run the forward modeling for the specified number of iterations.
    fwi.forward(iteration=iteration,fd_order=4,
                        batch_size=None,checkpoint_segments=10,
                        start_iter=0)
    
    # Retrieve the inversion results: updated velocity and loss values.
    iter_vp     = fwi.iter_vp
    iter_vs     = fwi.iter_vs
    iter_rho    = fwi.iter_rho
    iter_loss   = fwi.iter_loss
    
    # Save the iteration results to files for later analysis.
    np.savez(os.path.join(project_path,"inversion-vp_vs_rho-baseline/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,"inversion-vp_vs_rho-baseline/iter_vs.npz"),data=np.array(iter_vs))
    np.savez(os.path.join(project_path,"inversion-vp_vs_rho-baseline/iter_rho.npz"),data=np.array(iter_rho))
    np.savez(os.path.join(project_path,"inversion-vp_vs_rho-baseline/iter_loss.npz"),data=np.array(iter_loss))

    #------------------------------------------------------
    #            Visualize the Inversion Results
    #------------------------------------------------------
    from ADFWI.view.inverted_loss_model import plot_misfit,plot_initial_and_inverted,animate_inversion_process
    
    # misfit
    plot_misfit(iter_loss = iter_loss, save_path=os.path.join(project_path,f"inversion-vp_vs_rho-baseline/misfit.png"),show=False)
    
    # inverted results
    vp_init = vp_init.cpu().detach().numpy() if torch.is_tensor(vp_init) else vp_init
    vs_init = vs_init.cpu().detach().numpy() if torch.is_tensor(vs_init) else vs_init
    rho_init = rho_init.cpu().detach().numpy() if torch.is_tensor(rho_init) else rho_init
    plot_initial_and_inverted(vp_init=vp_init,iter_vp=iter_vp,save_path=os.path.join(project_path,f"inversion-vp_vs_rho-baseline/inverted_vp.png"),show=False)
    plot_initial_and_inverted(vp_init=vs_init,iter_vp=iter_vs,save_path=os.path.join(project_path,f"inversion-vp_vs_rho-baseline/inverted_vs.png"),show=False)
    plot_initial_and_inverted(vp_init=rho_init,iter_vp=iter_rho,save_path=os.path.join(project_path,f"inversion-vp_vs_rho-baseline/inverted_rho.png"),show=False)
    
    # inversion animation
    animate_inversion_process(iter_vp=iter_vp,vmin=vp_true.min(),vmax=vp_true.max(),save_path=os.path.join(project_path, f"inversion-vp_vs_rho-baseline/inversion_vp.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_vs,vmin=vs_true.min(),vmax=vs_true.max(),save_path=os.path.join(project_path, f"inversion-vp_vs_rho-baseline/inversion_vs.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_rho,vmin=rho_true.min(),vmax=rho_true.max(),save_path=os.path.join(project_path, f"inversion-vp_vs_rho-baseline/inversion_rho.gif"),fps=10)