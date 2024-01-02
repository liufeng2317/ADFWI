import datetime
import numpy as np
import os 
import json
from scipy import integrate
import torch 
from tqdm import tqdm
import torch.distributed as dist
from TorchInversion.gradient_precond import grad_precond
from TorchInversion.utils import source_wavelet,dictToObj,set_damp,list2numpy,numpy2tensor,summary_NN_structure
from TorchInversion.plots import plot_inversion_iter
from TorchInversion.propagators import Acoustic_Simulation
from TorchInversion.propagators_NN import Acoustic_Simulation_NN
from TorchInversion.propagators_RTM import Acoustic_Simulation_RTM
from TorchInversion.optimizer import Optimization
from TorchInversion.logger import *
import warnings
warnings.filterwarnings("ignore")

def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def is_main_process() -> bool:
    return get_rank() == 0


class TorchInversion():
    def __init__(self,save_path,device="cuda:0",inversion=False):
        if os.path.exists(save_path):
            print(f"WARNING:The Path has exists: {save_path}")
        dtstr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if inversion:
            dtstr = dtstr+"_inv"
        else:
            dtstr = dtstr+"_for"
        self.dtstr = dtstr
        self.save_path = save_path
        self.device = device
        self.initialize()
        self.configs = {}
        # logging
        log_dir = os.path.join(save_path,f"log/{dtstr}")
        logger.set_logdir(log_dir)
        logger.set_logger("logs")
        
        if is_main_process():
            logger.info(f"Save To: {save_path}")
            logger.info(f"device: {device}")
            logger.info(f"pid: {os.getpid()}")
        
    ###########################################
    #       1. initialize the project
    ###########################################
    def initialize(self):
        # forward 
        if not os.path.exists(os.path.join(self.save_path,"model/True")):
            os.makedirs(os.path.join(self.save_path,"model/True"))

        if not os.path.exists(os.path.join(self.save_path,"obs/figure")):
            os.makedirs(os.path.join(self.save_path,"obs/figure"))

        if not os.path.exists(os.path.join(self.save_path,"syn/figure")):
            os.makedirs(os.path.join(self.save_path,"syn/figure"))
        
        if not os.path.exists(os.path.join(self.save_path,f"log/{self.dtstr}")):
            os.makedirs(os.path.join(self.save_path,f"log/{self.dtstr}"))
        
        # inversion
        if not os.path.exists(os.path.join(self.save_path,"model/Initial")):
            os.makedirs(os.path.join(self.save_path,"model/Initial"))
            
        if not os.path.exists(os.path.join(self.save_path,"inv/model")):
            os.makedirs(os.path.join(self.save_path,"inv/model"))

        if not os.path.exists(os.path.join(self.save_path,"inv/grad")):
            os.makedirs(os.path.join(self.save_path,"inv/grad"))
    
    ###########################################
    #       2. initialize the project
    ###########################################
    def modelConfig(self,nx,ny,dx,dy,pml,fs,nt,dt,vmin,vmax):
        """model's configuration"""
        nx_pml = nx+2*pml
        ny_pml = ny+2*pml
        model_config = {
            "nx":nx,"ny":ny,
            "dx":dx,"dy":dy,
            "nt":nt,"dt":dt,
            "pml":pml,"fs":fs,
            "nx_pml":nx_pml,"ny_pml":ny_pml,
            "vmax":vmax,"vmin":vmin
        }
        model_config = dictToObj(model_config)
        return model_config
    
    def source(self,f0,src_x,src_y,model_config):
        """initialize the source"""
        nt = model_config.nt
        dt = model_config.dt
        pml = model_config.pml
        
        # free surface
        mask =  (src_x==pml)
        src_x[mask] = src_x[mask] + 1
        
        # wavelet
        src = source_wavelet(nt, dt, f0, 'Ricker') #Ricker wavelet
        stf_val = integrate.cumtrapz(src, axis=-1, initial=0) #Integrate
        stf_t = np.arange(nt)*dt
        
        acoustic_src = {
            "f0":f0,
            "src_x":src_x,"src_y":src_y,"src_n":len(src_x),
            "stf_val":stf_val,"stf_t":stf_t
        }
        acoustic_src = dictToObj(acoustic_src)
        
        return acoustic_src
        
    def receiver(self,rcv_x,rcv_y):
        """initialize the receiver"""
        acoustic_rcv = {
            'rcv_x':rcv_x,
            'rcv_y':rcv_y,
            'rcv_n':len(rcv_x)
        }
        acoustic_rcv = dictToObj(acoustic_rcv)
        return acoustic_rcv
    
    def vel_model(self,v,rho,model_config):
        """initialize the velocity model
            e.g. For forward is the true model
            e.g. For inversion is the initial model
        """
        vmax = model_config.vmax
        nx_pml = model_config.nx_pml;    ny_pml = model_config.ny_pml
        pml = model_config.pml;          dx = model_config.dx
        
        # damping
        damp_global = set_damp(vmax,nx_pml,ny_pml,pml,dx)
        # velocity model
        vel_model ={
            "v"             :v,
            "rho"           :rho,
            "damp_global"   :damp_global
        }
        vel_model= dictToObj(vel_model)
        return vel_model
    
    def optimConfig(self,lr,iteration,step_size,gamma,optim_method="Adam",device="cpu"):
        """optmization parametes for inversion"""
        optimizer_param = {
            "lr"            :lr,                     
            "iteration"     :iteration,
            "step_size"     :step_size,       
            "gamma"         :gamma,
            "optim_method"  :optim_method, 
        }
        optimizer_param = dictToObj(optimizer_param)
        return optimizer_param
    
    def saveConfig(self,model_config,source,receiver,vel_model,optimizer={},inversion=False):
        """save the model parameters"""
        class NdarrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        configs = {
            "model_config"  :model_config,
            "source"        :source,
            "receiver"      :receiver,
            "vel_model"     :vel_model,
            "optimizer"     :optimizer
        }
        configs = dictToObj(configs)
        if inversion:
            with open(os.path.join(self.save_path,"config_inv.json"),'w') as f:
                json.dump(configs,f,cls=NdarrayEncoder)
        else:
            with open(os.path.join(self.save_path,"config_for.json"),'w') as f:
                json.dump(configs,f,cls=NdarrayEncoder)
        
        self.configs = configs
        
        if is_main_process():
            lines = ["\n ================ Model settings ================"]
            for key,value in model_config.items():
                lines.append((f"\n{key}\t:\t {value}"))
            lines = "".join(lines)
            logger.info(lines)
            lines = ["\n ================ Source settings ================"]
            for key,value in source.items():
                if isinstance(value,list) or isinstance(value,np.ndarray):
                    value = list(map(lambda x:str(x),value))
                    value = " ".join(value)
                if key in ["f0","src_x","src_y"]:
                    lines.append((f"\n{key}\t:\t {value}"))
            lines = "".join(lines)
            logger.info(lines)
            lines = ["\n =============== Receiver settings ==============="]
            for key,value in receiver.items():
                if isinstance(value,list) or isinstance(value,np.ndarray):
                    value = list(map(lambda x:str(x),value))
                    value = " ".join(value)
                if key in ["rcv_x","rcv_y"]:
                    lines.append((f"\n{key}\t:\t {value}"))
            lines = "".join(lines)
            logger.info(lines)
        return configs
    
    def loadConfig(self,param_path=""):
        """get the parameter from json file"""
        if not param_path=="":
            param = json.load(open(self.param_path,"r"))
        elif os.path.exists(os.path.join(self.save_path,"config_for.json")):
            file_path = os.path.join(self.save_path,"config_for.json")
            param = json.load(open(file_path))
        
        param = dictToObj(param)
        model_config = param.model_config
        # source
        source = param.source
        source.src_x = list2numpy(source.src_x)
        source.src_y = list2numpy(source.src_y)
        source.stf_val = list2numpy(source.stf_val)
        source.stf_t = list2numpy(source.stf_t)
        # receiver
        receiver = param.receiver
        receiver.rcv_x = list2numpy(receiver.rcv_x)
        receiver.rcv_y = list2numpy(receiver.rcv_y)
        # velocity model
        vel_model = param.vel_model
        vel_model.v = list2numpy(vel_model.v)
        vel_model.rho = list2numpy(vel_model.rho)
        vel_model.damp_global = list2numpy(vel_model.damp_global)
        
        self.configs["settings"] = model_config
        self.configs["source"] = source
        self.configs["receiver"] = receiver
        self.configs["vel_model"] = vel_model
        
        return model_config,source,receiver,vel_model
    
    ###########################################
    #    3. forward simulation
    ###########################################
    def forwrd(self,normalize=True):
        model_config = self.configs.model_config
        vel_model    = self.configs.vel_model
        source       = self.configs.source
        receiver     = self.configs.receiver     
        # simulation settings
        device = self.device

        # input model
        v = numpy2tensor(vel_model.v).to(device)
        rho = numpy2tensor(vel_model.rho).to(device)
        
        acoustic_sim = Acoustic_Simulation(model_config,vel_model,source,receiver,
                                            v=v,rho=rho,
                                            device=device)
        
        # forwrd simulation and normalize or not
        csg,_ = acoustic_sim.forward()
        if normalize:
            csg = csg/(torch.max(torch.abs(csg),axis=1,keepdim=True).values)
        
        if device == 'cpu':
            csg = csg.detach().numpy()
        else:
            csg = csg.cpu().detach().numpy()
        np.savez(os.path.join(self.save_path,"obs/obs.npz"),obs_data = csg)
        return csg
    
    ###########################################
    #    4. Inversion use AD
    ###########################################
    def inversion(self,csg_obs,normalize=True,grad_precondition=None,optim_method="adam"):
        model_config = self.configs.model_config
        vel_model    = self.configs.vel_model
        source       = self.configs.source
        receiver     = self.configs.receiver
        optim_config = self.configs.optimizer

        # optimizer
        lr = optim_config.lr
        iteration = optim_config.iteration
        step_size = optim_config.step_size
        gamma = optim_config.gamma
        device = self.device
        
        # input model
        v = numpy2tensor(vel_model.v).to(device)
        v.requires_grad = True
        rho = numpy2tensor(vel_model.rho).to(device)
        
        # observed data
        csg_obs = numpy2tensor(csg_obs).to(device)
        
        # simulation
        acoustic_sim = Acoustic_Simulation(model_config,vel_model,source,receiver,
                                            v=v,rho=rho,obs_data=csg_obs,device=device)

        # optimizer
        if optim_method.lower() == "adam":
            print("Using Adam to Iter the inversion")
            optimizer = torch.optim.Adam(acoustic_sim.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
            # iterative inversion
            loss_lists,inv_v = self.inversion_step_Adam(iteration,csg_obs,optimizer,scheduler,acoustic_sim,grad_precondition,normalize)
        else:
            print(f"Using {optim_method} to Iter the inversion")
            optimizer = Optimization(niter_max=iteration,conv=1e-8,method='SD',debug=False)
            optimizer.nls_max = 30
            optimizer.alpha = lr
            scheduler = None
            # iterative inversion
            loss_lists,inv_v = self.inversion_step_SD(iteration,csg_obs,optimizer,scheduler,acoustic_sim,grad_precondition,normalize)
        return loss_lists,inv_v
    
    def inversion_step_SD(self,iteration,csg_obs,optimizer,scheduler,acoustic_sim,grad_precondition,normalize=True):
        # model parameters
        model_config = self.configs.model_config
        grad_precondition = dictToObj(grad_precondition)
        dt = model_config.dt
        vmin,vmax = model_config.vmin,model_config.vmax
        nx,ny = model_config.nx,model_config.ny
        # grad precondition settings
        if grad_precondition == None:
            grad_precondition = {"grad_mute":0,"grad_smooth":0,"marine_or_land":"land",}
        
        def inversion_temp(acoustic_sim):
            csg,forw = acoustic_sim.forward()
            if normalize:
                csg = csg/(torch.max(torch.abs(csg),axis=1,keepdim=True).values)
            # the L2 misfit
            rsd = csg-csg_obs
            loss = torch.sum(torch.sqrt(torch.sum(rsd*rsd*dt,axis=1)))
            loss.backward()
            # grads precondition
            with torch.no_grad():
                v = acoustic_sim.v.cpu().detach().numpy()
                grads = acoustic_sim.v.grad.cpu().detach().numpy()
                forw = forw.cpu().detach().numpy()
                grads = grad_precond(model_config,grads,forw,
                                    grad_mute=grad_precondition.grad_mute,
                                    grad_smooth=grad_precondition.grad_smooth,
                                    marine_or_land=grad_precondition.marine_or_land)
                grads = numpy2tensor(grads).to(self.device)
                acoustic_sim.v.grad =  grads
            
            v = acoustic_sim.v.cpu().detach().numpy()
            grads = acoustic_sim.v.grad.cpu().detach().numpy()
            loss_np = loss.cpu().detach().numpy()
            return v,grads,loss_np
        
        loss_lists,inv_v,inv_grad = [],[],[]
        # iterative inversion
        pbar = tqdm(range(iteration))
        for i in pbar:
            # calculate the gradient
            if i > 0:
                v = numpy2tensor(v).to(self.device)
                v = torch.nn.Parameter(v)
                acoustic_sim.v = v
            v,grads,loss = inversion_temp(acoustic_sim)
            if i ==0:
                grad_pre = np.zeros_like(grads)
            # optimizer the velocity
            v = optimizer.iterate(v,loss,grads,grad_pre)
            v = np.clip(v,a_min=vmin,a_max=vmax)
            v = v.reshape(nx,ny)
            if(optimizer.FLAG == 'GRAD'):
                # update the gradient
                grad_pre = grads
            
            # save the temp result
            inv_grad.append(grads)
            inv_v.append(v)
            loss_lists.append(loss)
            
            # plot the temp result
            plot_inversion_iter(i,v,grads,save_path=self.save_path)
            pbar.set_description("Iter:{},Loss:{:.4}".format(i,loss))

        inv_v = np.array(inv_v)
        inv_grad = np.array(inv_grad)
        loss_lists = np.array(loss_lists) 

        np.savez(os.path.join(self.save_path,"inv/inv_v.npz"),data=inv_v)
        np.savetxt(os.path.join(self.save_path,"inv/loss.txt"),loss_lists)
        np.savez(os.path.join(self.save_path,"inv/inv_grad.npz"),data=inv_grad)
        return loss_lists,inv_v[-1]

    
    def inversion_step_Adam(self,iteration,csg_obs,optimizer,scheduler,acoustic_sim,grad_precondition,normalize=True):
        # grad precondition settings
        model_config = self.configs.model_config
        if grad_precondition == None:
            grad_precondition = {"grad_mute":0,"grad_smooth":0,"marine_or_land":"land",}
        grad_precondition = dictToObj(grad_precondition)
        # iterative inversion
        dt = model_config.dt
        loss_lists,inv_v,inv_grad = [],[],[]
        pbar = tqdm(range(iteration))
        for i in pbar:
            csg,forw = acoustic_sim.forward()
            if normalize:
                csg = csg/(torch.max(torch.abs(csg),axis=1,keepdim=True).values)
            # the L2 misfit
            rsd = csg-csg_obs
            loss = torch.sum(torch.sqrt(torch.sum(rsd*rsd*dt,axis=1)))
            optimizer.zero_grad()
            loss.backward()
            # grads precondition
            with torch.no_grad():
                v = acoustic_sim.v.cpu().detach().numpy()
                grads = acoustic_sim.v.grad.cpu().detach().numpy()
                forw = forw.cpu().detach().numpy()
                grads = grad_precond(model_config,grads,forw,
                                    grad_mute=grad_precondition.grad_mute,
                                    grad_smooth=grad_precondition.grad_smooth,
                                    marine_or_land=grad_precondition.marine_or_land)
                grads = numpy2tensor(grads).to(self.device)
                acoustic_sim.v.grad =  grads

            # optimizer step
            optimizer.step()
            scheduler.step()
            
            for para in acoustic_sim.parameters():
                para.data.clamp_(model_config.vmin,model_config.vmax)
            
            v = acoustic_sim.v.cpu().detach().numpy()
            grads = acoustic_sim.v.grad.cpu().detach().numpy()
            loss_np = loss.cpu().detach().numpy()
            
            # save the temp result
            inv_grad.append(grads)
            inv_v.append(v)
            loss_lists.append(loss_np)
            
            # plot the temp result
            plot_inversion_iter(i,v,grads,save_path=self.save_path)
            pbar.set_description("Iter:{},Loss:{:.4}".format(i,loss_np))

        inv_v = np.array(inv_v)
        inv_grad = np.array(inv_grad)
        loss_lists = np.array(loss_lists) 

        np.savez(os.path.join(self.save_path,"inv/inv_v.npz"),data=inv_v)
        np.savetxt(os.path.join(self.save_path,"inv/loss.txt"),loss_lists)
        np.savez(os.path.join(self.save_path,"inv/inv_grad.npz"),data=inv_grad)
        return loss_lists,inv_v[-1]

    ###########################################
    #    5. Inversion use AD and NN
    ###########################################
    def inversion_NN(self,csg_obs,normalize=True,NN_model="CNN",generator="v",pretrain_param=None):
        model_config = self.configs.model_config
        vel_model    = self.configs.vel_model
        source       = self.configs.source
        receiver     = self.configs.receiver
        optim_config = self.configs.optimizer

        # optimizer
        lr = optim_config.lr
        iteration = optim_config.iteration
        step_size = optim_config.step_size
        gamma = optim_config.gamma
        device = self.device
        
        # observed data
        csg_obs = numpy2tensor(csg_obs).to(device)
        
        # simulation
        acoustic_sim = Acoustic_Simulation_NN(model_config,vel_model,source,receiver,
                                            obs_data=csg_obs,device=device,
                                            NN_model=NN_model,generator=generator)

        # model structure
        summary_NN_structure(acoustic_sim.model,device=device)
        
        # pretrain
        if generator == "v" and pretrain_param != None:
            pretrain_param = dictToObj(pretrain_param)
            print("="*20+"\t"+f"Pretrain {NN_model}"+"\t"+"="*20)
            if os.path.exists(os.path.join(self.save_path,"inv/model.pt")):
                acoustic_sim.model.load_state_dict(torch.load(os.path.join(self.save_path,"inv/model.pt")))
            else:
                acoustic_sim.preTrainModel(pretrain_param=pretrain_param)
                torch.save(self.model.state_dict(),os.path.join(self.save_path,"inv/model.pt"))
            
        # optimizer
        optimizer = torch.optim.Adam(acoustic_sim.parameters(),lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
                
        # iterative inversion
        print("="*20+"\t"+f"Train with {NN_model}"+"\t"+"="*20)
        loss_lists,inv_v = self.inversion_step_NN(iteration,csg_obs,optimizer,scheduler,acoustic_sim,normalize)
        
        return loss_lists,inv_v
    
    def inversion_step_NN(self,iteration,csg_obs,optimizer,scheduler,acoustic_sim,normalize=True):
        # grad precondition settings
        model_config = self.configs.model_config
        vel_model = self.configs.vel_model
        dt = model_config.dt
        vmin = model_config.vmin; vmax = model_config.vmax
        
        loss_lists,inv_v,inv_grad = [],[],[]
        
        # iterative inversion
        pbar = tqdm(range(iteration))
        for i in pbar:
            csg,forw = acoustic_sim.forward()
            if normalize:
                csg = csg/(torch.max(torch.abs(csg),axis=1,keepdim=True).values)
            # the L2 misfit
            rsd = csg-csg_obs
            loss = torch.sum(torch.sqrt(torch.sum(rsd*rsd*dt,axis=1)))
            
            optimizer.zero_grad()
            loss.backward()
            
            # output initial model
            if acoustic_sim.generator == "dv":
                dv = acoustic_sim.get_v(acoustic_sim.model()).cpu().detach().numpy()
                dv[:12] = 0
                v = vel_model.v + dv
                v = np.clip(v,a_min=vmin,a_max=vmax)
            elif acoustic_sim.generator == "v":
                v = acoustic_sim.get_v(acoustic_sim.model()).cpu().detach().numpy()
                v[:12] = vel_model.v[:12]
                v = np.clip(v,a_min=vmin,a_max=vmax)
            
            grads = v - vel_model.v
            loss_np = loss.cpu().detach().numpy()
            
            # save the temp result
            inv_v.append(v)
            inv_grad.append(grads)
            loss_lists.append(loss_np)
            
            # plot the temp result
            plot_inversion_iter(i,v,grads,save_path=self.save_path)
            
            # optimizer step
            optimizer.step()
            scheduler.step()
            
            pbar.set_description("Iter:{},Loss:{:.4}".format(i,loss_np))

        inv_v = np.array(inv_v)
        inv_grad = np.array(inv_grad)
        loss_lists = np.array(loss_lists) 

        np.savez(os.path.join(self.save_path,"inv/inv_v.npz"),data=inv_v)
        np.savetxt(os.path.join(self.save_path,"inv/loss.txt"),loss_lists)
        np.savez(os.path.join(self.save_path,"inv/inv_grad.npz"),data=inv_grad)
        return loss_lists,inv_v[-1]

    ###########################################
    #    5. Inversion use RTM data
    ###########################################
    def inversion_RTM(self,csg_obs,RTM_img,normalize=True,NN_model="CNN",generator="v",pretrain_param=None):
        model_config = self.configs.model_config
        vel_model    = self.configs.vel_model
        source       = self.configs.source
        receiver     = self.configs.receiver
        optim_config = self.configs.optimizer

        # optimizer
        lr = optim_config.lr
        iteration = optim_config.iteration
        step_size = optim_config.step_size
        gamma = optim_config.gamma
        device = self.device
        
        # observed data
        csg_obs = numpy2tensor(csg_obs).to(device)
        RTM_img = numpy2tensor(RTM_img).to(device)
        
        # simulation
        acoustic_sim = Acoustic_Simulation_RTM(model_config,vel_model,source,receiver,
                                            obs_data=csg_obs,device=device,
                                            NN_model=NN_model,generator=generator)

        # model structure
        summary_NN_structure(acoustic_sim.model,device=device)
        
        # pretrain
        if generator == "v" and pretrain_param != None:
            pretrain_param = dictToObj(pretrain_param)
            print("="*20+"\t"+f"Pretrain {NN_model}"+"\t"+"="*20)
            if os.path.exists(os.path.join(self.save_path,"inv/model.pt")):
                acoustic_sim.model.load_state_dict(torch.load(os.path.join(self.save_path,"inv/model.pt")))
            else:
                acoustic_sim.preTrainModel(pretrain_param=pretrain_param)
                torch.save(acoustic_sim.model.state_dict(),os.path.join(self.save_path,"inv/model.pt"))

        # optimizer
        optimizer = torch.optim.Adam(acoustic_sim.parameters(),lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
                
        # iterative inversion
        print("="*20+"\t"+f"Train with {NN_model}"+"\t"+"="*20)
        loss_lists,inv_v = self.inversion_step_RTM(iteration,csg_obs,RTM_img,optimizer,scheduler,acoustic_sim)
        
        return loss_lists,inv_v
    
    def inversion_step_RTM(self,iteration,csg_obs,RTM_img,optimizer,scheduler,acoustic_sim):
        # grad precondition settings
        model_config = self.configs.model_config
        vel_model = self.configs.vel_model
        dt = model_config.dt
        vmin = model_config.vmin; vmax = model_config.vmax
        loss_lists,inv_v,inv_grad = [],[],[]
        
        # iterative inversion
        pbar = tqdm(range(iteration))
        for i in pbar:
            if i<=500:
                temp_RTM_img = RTM_img
                pretrain = False
            else:
                temp_RTM_img = []
                pretrain = True
            csg,forw = acoustic_sim.forward(RTM_img=temp_RTM_img,pretrain=pretrain)
            # the L2 misfit
            rsd = csg-csg_obs
            loss = torch.sum(torch.sqrt(torch.sum(rsd*rsd*dt,axis=1)))
            
            optimizer.zero_grad()
            loss.backward()
            
            # output initial model
            with torch.no_grad():
                if acoustic_sim.generator == "dv":
                    dv = acoustic_sim.get_v(acoustic_sim.model(RTM_img=temp_RTM_img,pretrain=pretrain)).cpu().detach().numpy()
                    dv[:10] = 0
                    v = vel_model.v + dv
                    v = np.clip(v,a_min=vmin,a_max=vmax)
                elif acoustic_sim.generator == "v":
                    v = acoustic_sim.get_v(acoustic_sim.model(RTM_img=temp_RTM_img,pretrain=pretrain)).cpu().detach().numpy()
                    v[:10] = vel_model.v[:10]
                    v = np.clip(v,a_min=vmin,a_max=vmax)
            
            grads = v - vel_model.v
            loss_np = loss.cpu().detach().numpy()
            
            # save the temp result
            inv_v.append(v)
            inv_grad.append(grads)
            loss_lists.append(loss_np)
            
            # plot the temp result
            plot_inversion_iter(i,v,grads,save_path=self.save_path)
            
            # optimizer step
            optimizer.step()
            scheduler.step()
            
            pbar.set_description("Iter:{},Loss:{:.4}".format(i,loss_np))

        inv_v = np.array(inv_v)
        inv_grad = np.array(inv_grad)
        loss_lists = np.array(loss_lists) 

        np.savez(os.path.join(self.save_path,"inv/inv_v.npz"),data=inv_v)
        np.savetxt(os.path.join(self.save_path,"inv/loss.txt"),loss_lists)
        np.savez(os.path.join(self.save_path,"inv/inv_grad.npz"),data=inv_grad)
        return loss_lists,inv_v[-1]