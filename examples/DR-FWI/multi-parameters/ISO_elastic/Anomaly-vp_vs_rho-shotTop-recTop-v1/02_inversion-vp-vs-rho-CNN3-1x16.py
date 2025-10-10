import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")
from scipy import integrate
import sys
import os
sys.path.append("../../../../../")
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
torch.cuda.set_device(2)

if __name__ == "__main__":
    project_path = "./data/"
    if not os.path.exists(os.path.join(project_path,"model")):
        os.makedirs(os.path.join(project_path,"model"))
    if not os.path.exists(os.path.join(project_path,"waveform")):
        os.makedirs(os.path.join(project_path,"waveform"))
    if not os.path.exists(os.path.join(project_path,"survey")):
        os.makedirs(os.path.join(project_path,"survey"))
    if not os.path.exists(os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16")):
        os.makedirs(os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16"))
    #------------------------------------------------------
    #                   Basic Parameters
    #------------------------------------------------------
    device = "cuda:2"
    dtype = torch.float32     # Set data type to 32-bit floating point
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



    # -----------------------------------
    #     Define DIP model
    # -----------------------------------
    from ADFWI.dip.model_multi.CNN import CNNs as DIP_CNN
    model_shape = [nz,nx]
    DIP_model  = DIP_CNN(model_shape,
                        random_state_num=100,
                        in_channels=[16],
                        out_channels_number=3,
                        vmins=[vp_true.min()/1000,vs_true.min()/1000,rho_true.min()/1000] ,
                        vmaxs=[vp_true.max()/1000,vs_true.max()/1000,rho_true.max()/1000],
                        units=[1000,1000,1000],device=device)
    DIP_model.to(device)

    # -----------------------------------
    #     Pretrain DIP model
    # -----------------------------------
    pretrain        = True
    load_pretrained = False

    if pretrain:
        if load_pretrained:
            # load the model parameters
            DIP_model.load_state_dict(torch.load(os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/DIP_model_vp_vs_rho_pretrained.pt")))
        else:
            lr          = 0.0001
            iteration   = 5000
            step_size   = 1000
            gamma       = 0.5
            optimizer = torch.optim.Adam(DIP_model.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            vp_init = numpy2tensor(vp_init,dtype=dtype).to(device)
            vs_init = numpy2tensor(vs_init,dtype=dtype).to(device)
            rho_init = numpy2tensor(rho_init,dtype=dtype).to(device)
            pbar = tqdm(range(iteration+1))
            for i in pbar:  
                vp_nn,vs_nn,rho_nn = DIP_model()
                loss = torch.sqrt(torch.sum((vp_nn - vp_init)**2)) + torch.sqrt(torch.sum((vs_nn - vs_init)**2))+ torch.sqrt(torch.sum((rho_nn - rho_init)**2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_description(f'Pretrain Iter:{i}, Misfit:{loss.cpu().detach().numpy()}')
            torch.save(DIP_model.state_dict(),os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/DIP_model_vp_vs_rho_pretrained.pt"))
    
    # -----------------------------------
    #     velocity model for FWI
    # -----------------------------------
    from ADFWI.dip.dip_elastic_model_vp_vs_rho import DIP_ElasticModel_vp_vs_rho
    model = DIP_ElasticModel_vp_vs_rho(ox,oz,nx,nz,dx,dz,
                            DIP_model=DIP_model,
                            vp_init=vp_init,
                            vs_init=vs_init,
                            rho_init=rho_init,
                            vp_bound =[vp_true.min(),vp_true.max()],
                            vs_bound =[vs_true.min(),vs_true.max()],
                            rho_bound=[rho_true.min(),rho_true.max()],
                            free_surface=free_surface,
                            abc_type="PML",abc_jerjan_alpha=0.007,nabc=nabc,
                            auto_update_rho=False, auto_update_vp=False,
                            device=device,dtype=dtype)
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
    F = ElasticPropagator(model,survey,device=device)
    
    # load data
    d_obs = SeismicData(survey)
    d_obs.load(os.path.join(project_path,"waveform/obs_data.npz"))
    print(d_obs.__repr__())
    
    # optimizer
    iteration   =   1000
    optimizer   =   torch.optim.Adam(model.parameters(), lr = 0.001)
    # optimizer   =   torch.optim.SGD(model.parameters(), lr = 1e-4)
    scheduler   =   torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.75,last_epoch=-1)

    # Setup misfit function
    from ADFWI.fwi.misfit import Misfit_global_correlation
    from ADFWI.fwi.regularization import regularization_TV_2order
    loss_fn = Misfit_global_correlation(dt=1)
    regularization_fn = regularization_TV_2order(nx,nz,dx,dz,step_size=50,gamma=0.9,device=device,dtype=dtype)

    # gradient processor
    grad_mask             = np.ones((nz,nx))
    gradient_processor_vp = GradProcessor(grad_mask=grad_mask,forw_illumination=False,norm_grad=False,grad_smooth=0)
    gradient_processor_vs = GradProcessor(grad_mask=grad_mask,forw_illumination=False,norm_grad=False,grad_smooth=0)
    gradient_processor_rho= GradProcessor(grad_mask=grad_mask,forw_illumination=False,norm_grad=False,grad_smooth=0)
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
                        save_fig_path=os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16"),
                        inversion_component=["vx","vz"],
                        )

    fwi.forward(iteration=iteration,fd_order=4,
                    batch_size=None,checkpoint_segments=11,
                    start_iter=0)
    
    # Retrieve the inversion results: updated velocity and loss values.
    iter_vp     = fwi.iter_vp
    iter_vs     = fwi.iter_vs
    iter_rho    = fwi.iter_rho
    iter_loss   = fwi.iter_loss

    np.savez(os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/iter_vp.npz"),data=np.array(iter_vp))
    np.savez(os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/iter_vs.npz"),data=np.array(iter_vs))
    np.savez(os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/iter_rho.npz"),data=np.array(iter_rho))
    np.savez(os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/iter_loss.npz"),data=np.array(iter_loss))
    torch.save(model.DIP_model.state_dict(),os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/DIP_model_vp_vs_rho.pt"))

    #------------------------------------------------------
    #            Visualize the Inversion Results
    #------------------------------------------------------
    from ADFWI.view.inverted_loss_model import plot_misfit,plot_initial_and_inverted,animate_inversion_process
    
    # misfit
    plot_misfit(iter_loss = iter_loss, save_path=os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/misfit.png"),show=False)
    
    # inverted results
    vp_init = vp_init.cpu().detach().numpy() if torch.is_tensor(vp_init) else vp_init
    vs_init = vs_init.cpu().detach().numpy() if torch.is_tensor(vs_init) else vs_init
    rho_init = rho_init.cpu().detach().numpy() if torch.is_tensor(rho_init) else rho_init
    plot_initial_and_inverted(vp_init=vp_init,iter_vp=iter_vp,save_path=os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/inverted_vp.png"),show=False)
    plot_initial_and_inverted(vp_init=vs_init,iter_vp=iter_vs,save_path=os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/inverted_vs.png"),show=False)
    plot_initial_and_inverted(vp_init=rho_init,iter_vp=iter_rho,save_path=os.path.join(project_path,f"inversion-vp_vs_rho-CNN3-1x16/inverted_rho.png"),show=False)
    
    # inversion animation
    animate_inversion_process(iter_vp=iter_vp,vmin=vp_true.min(),vmax=vp_true.max(),save_path=os.path.join(project_path, f"inversion-vp_vs_rho-CNN3-1x16/inversion_vp.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_vs,vmin=vs_true.min(),vmax=vs_true.max(),save_path=os.path.join(project_path, f"inversion-vp_vs_rho-CNN3-1x16/inversion_vs.gif"),fps=10)
    animate_inversion_process(iter_vp=iter_rho,vmin=rho_true.min(),vmax=rho_true.max(),save_path=os.path.join(project_path, f"inversion-vp_vs_rho-CNN3-1x16/inversion_rho.gif"),fps=10)