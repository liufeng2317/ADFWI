# Isotropic Elastic FWI Tests

## 1. Isotropic Elastic Forward Modeling

### 1.1 Packages Import

We begin by importing the libraries that provide functionality for numerical operations, plotting, and file management:

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import integrate
import sys
import os

from ADFWI.propagator  import *
from ADFWI.model       import *
from ADFWI.view        import *
from ADFWI.utils       import *
from ADFWI.survey      import *
```

Next, we set up the project path and ensure the necessary directories are created to store results from the modeling and inversion process:

```python
project_path = "./data" # Path to save intermediate results
if not os.path.exists(os.path.join(project_path,"model")):
    os.makedirs(os.path.join(project_path,"model"))
if not os.path.exists(os.path.join(project_path,"waveform")):
    os.makedirs(os.path.join(project_path,"waveform"))
if not os.path.exists(os.path.join(project_path,"survey")):
    os.makedirs(os.path.join(project_path,"survey"))
```

---

### 1.2 Basic Parameters Definition

Next to define key parameters for the simulation, including grid resolution, time steps, and wave source characteristics.

```python
device = "cuda:0"         # Specify the GPU device
dtype = torch.float32     # Set data type to 32-bit floating point
ox, oz = 0, 0             # Origin coordinates for x and z directions
nz, nx = 78, 200          # Grid dimensions in z and x directions
dx, dz = 45, 45           # Grid spacing in x and z directions
nt, dt = 2500, 0.003      # Time steps and time interval
nabc = 50                 # Thickness of the absorbing boundary layer
f0 = 5                    # Initial frequency in Hz
free_surface = True       # Enable free surface boundary condition
```

- **`device`**: Specifies which device (`CPU` or `GPU`) will be used for the computation.
- **`dtype`**: The data type for the tensors (e.g. `torch.float32`) 
- **`ox, oz`**: The origin of the grid in both the x and z directions.
- **`nz, nx`**: The number of grid points in the vertical (z) and horizontal (x) directions. 
- **`dx, dz`**: The grid spacing in the x and z directions.
- **`nt, dt`**: The number of time steps (`nt`) and the time interval between each step (`dt`).
- **`nabc`**: The thickness of the absorbing boundary layer in the model.
- **`f0`**: The initial frequency of the source wave.
- **`free_surface`**: A boolean flag indicating whether a free surface boundary condition is applied.

---

### 1.3 Velocity Model Definition

In this section, we load and prepare a predefined velocity model (e.g., Marmousi2 model) using the `ADFWI.model.IsotropicElasticModel` class.

```python
# Load the Marmousi model dataset
marmousi_model = load_marmousi_model(in_dir="./examples/datasets/marmousi2_source")
# Resample the Marmousi model for the defined coordinates
x = np.linspace(5000, 5000 + dx * nx, nx)
z = np.linspace(0,dz * nz, nz)
vel_model = resample_marmousi_model(x, z, marmousi_model)
# Extract and transpose the primary wave velocity (vp)
vp = vel_model['vp'].T
vs = vel_model['vs'].T
# Calculate density (rho) based on velocity
rho = np.power(vp, 0.25) * 310
# Initialize the IsotropicElasticModel model with parameters and properties
model = IsotropicElasticModel(
                    ox,oz,nx,nz,dx,dz,
                    vp,vs,rho,
                    vp_grad = False,vs_grad = False, rho_grad=False,
                    free_surface=free_surface,
                    abc_type="PML",abc_jerjan_alpha=0.007,
                    auto_update_rho=False,
                    auto_update_vp =False,
                    nabc=nabc,
                    device=device,dtype=dtype)
```

After initializing the model, it is saved for future use:
```python
# Save the model to a file
model.save(os.path.join(project_path, "model/true_model.npz"))
# Print the model representation
print(model.__repr__())
# Plot the primary model
model._plot_vp_vs_rho(figsize=(12,5),wspace=0.2,cbar_pad_fraction=0.18,cbar_height=0.04,cmap='coolwarm')
```
This model setup uses the Marmousi2 dataset, resamples it to match the grid, computes the density, and saves the model for subsequent simulations.

**Note: We provide utility functions to automatically download and load several predefined models. These models can be loaded using functions from `ADFWI.utils.velocityDemo`, including**:
- **Layer model**
- **Anomaly model**
- **Marmousi2 model**
- **Overthrust model**
- **Foothill model**
- **Hess model**
- **Valhall model**

---

### 1.4 Observed System Definition

**The observed system is defined using grid points (not physical distances) where the sources and receivers are located.** This section also sets up the seismic wave sources and receivers and creates a **Survey** object that connects both for simulation.

#### (1) Seismic Source Definition:

We define the positions of the seismic sources and generate the corresponding wavelets. Sources are added using a Source object:

```python
# Define source positions in the model
src_z = np.array([2 for i in range(2,nx-2,5)]) 
src_x = np.array([i for i in range(2,nx-2,5)])

# Generate wavelet for the source
src_t, src_v = wavelet(nt, dt, f0, amp0=1)  # Create time and wavelet amplitude
src_v = integrate.cumtrapz(src_v, axis=-1, initial=0)  # Integrate wavelet to get velocity

source = Source(nt=nt, dt=dt, f0=f0)  # Initialize source object

# Method 1: Add multiple sources at once (commented out)
# source.add_sources(src_x=src_x, src_z=src_z, src_wavelet=src_v, src_type='mt', src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))

# Method 2: Loop through each source position to add them individually
for i in range(len(src_x)):
    source.add_source(src_x=src_x[i], src_z=src_z[i], src_wavelet=src_v, src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))

# Plot the wavelet used in the source
source.plot_wavelet(save_path=os.path.join(project_path, "survey/wavelets.png"))
```
* `src_x` and `src_z`: Define the source positions.
* `wavelet()`: Creates the seismic wavelet.
* `integrate.cumtrapz()`: Converts the amplitude wavelet into velocity by integrating it.
* `source.add_source()`: Adds each source to the model.

#### (2) Seismic Receiver Definition:

Next, we define the receiver positions and initialize the Receiver object:

```python
# Define receiver positions in the model
rcv_z = np.array([2   for i in range(0,nx,1)])
rcv_x = np.array([j   for j in range(0,nx,1)])

receiver = Receiver(nt=nt, dt=dt)  # Initialize receiver object

# Method 1: Add all receivers at once (commented out)
# receiver.add_receivers(rcv_x=rcv_x, rcv_z=rcv_z, rcv_type='pr')

# Method 2: Loop through each receiver position to add them individually
for i in range(len(rcv_x)):
    receiver.add_receiver(rcv_x=rcv_x[i], rcv_z=rcv_z[i], rcv_type="pr")
```
The receivers are placed at specific grid points (along the x-axis), ready to record the seismic waves generated by the sources.
- **Note: The receiver type is currently not in use, but it will be utilized in future updates.**

#### (3) Creating the Survey Object:

Finally, the **Survey** object is created by associating the sources and receivers. We can print a summary of the **Survey** object and plot it to verify its configuration

```python
# Create a survey object using the defined source and receiver
survey = Survey(source=source, receiver=receiver)

# Print a representation of the survey object to check its configuration
print(survey.__repr__())

# Plot the survey configuration over the velocity model
survey.plot(model.vp, cmap='coolwarm')
```

---

### 1.5 Define the Propagator & Forward Modeling

In this step, we initialize the wave propagator and perform the forward modeling to simulate the wave propagation through the defined model. The propagator takes into account the velocity model, survey setup (source and receiver positions), and relevant physical properties for the isotropic elastic waves.

#### (1) Initialize the Wave Propagator

We start by initializing the `ElasticPropagator`, which is responsible for simulating wave propagation using the provided model and survey configuration:

```python
# Initialize the wave propagator using the specified model and survey configuration
F = ElasticPropagator(model,survey,device=device)
```

Here, `model` contains the velocity and density information, while `survey` holds the source and receiver positions. The `device` specifies the hardware (e.g., GPU or CPU) for running the simulation.

#### (2) Forward Propagation

Next, we perform the forward propagation to simulate how the isotropic elastic waves travel through the model and record the waveforms. The `forward()` method of the propagator performs the simulation and returns the recorded wavefields:

```python
# Perform the forward propagation to record waveforms (fd_order: 4, 6, 8, 10)
record_waveform = F.forward(fd_order=4)
```

The `record_waveform` object contains the recorded data during the forward propagation. We extract the relevant wavefields from the recorded data:

```python
# Extract recorded pressure wavefield and particle velocities
rcv_txx = record_waveform["txx"]
rcv_tzz = record_waveform["tzz"]
rcv_txz = record_waveform["txz"]
rcv_vx  = record_waveform["vx"]
rcv_vz  = record_waveform["vz"]
```

Additionally, we can extract the forward wavefields, which represent the evolution of wavefields during propagation:

```python
# Extract forward wavefields for analysis
forward_wavefield_txx = record_waveform["forward_wavefield_txx"]
forward_wavefield_tzz = record_waveform["forward_wavefield_tzz"]
forward_wavefield_txz = record_waveform["forward_wavefield_txz"]
forward_wavefield_vx  = record_waveform["forward_wavefield_vx"]
forward_wavefield_vz  = record_waveform["forward_wavefield_vz"]
```

#### (3) Save the Modeling Results

Finally, we save the recorded waveforms to disk for later use, such as inversion or analysis. We create a `SeismicData` object to store the observed data from the survey and record the waveforms into it:

```python
# Create a SeismicData object to store observed data from the survey
d_obs = SeismicData(survey)

# Record the waveform data into the SeismicData object
d_obs.record_data(record_waveform)

# Save the recorded data to a specified file
d_obs.save(os.path.join(project_path, "waveform/obs_data.npz"))
```

This step ensures that all the forward modeling results are safely stored and ready for further analysis or inversion tasks.

----

## 2. Isotropic Elastic Inversion

### 2.1 Packages Import
The imported packages are same to the forward modeling

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
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

project_path = "./data/"
if not os.path.exists(os.path.join(project_path,"model")):
    os.makedirs(os.path.join(project_path,"model"))
if not os.path.exists(os.path.join(project_path,"waveform")):
    os.makedirs(os.path.join(project_path,"waveform"))
if not os.path.exists(os.path.join(project_path,"survey")):
    os.makedirs(os.path.join(project_path,"survey"))
if not os.path.exists(os.path.join(project_path,"inversion")):
    os.makedirs(os.path.join(project_path,"inversion"))
```

### 2.2 Basic Parameters Definition
The Basic parameters is the same to the forward modeling

```python
device = "cuda:0"         # Specify the GPU device
dtype = torch.float32     # Set data type to 32-bit floating point
ox, oz = 0, 0             # Origin coordinates for x and z directions
nz, nx = 78, 200          # Grid dimensions in z and x directions
dx, dz = 45, 45           # Grid spacing in x and z directions
nt, dt = 2500, 0.003      # Time steps and time interval
nabc = 50                 # Thickness of the absorbing boundary layer
f0 = 5                    # Initial frequency in Hz
free_surface = True       # Enable free surface boundary condition
```

----

### 2.3 Velocity Model Definition
For inversion, you need to define the intiial velocity model. In this case, we use the smoothed true model as the initial model.

```python
# Load the Marmousi model dataset from the specified directory.
marmousi_model = load_marmousi_model(in_dir="./examples/datasets/marmousi2_source")

# Create coordinate arrays for x and z based on the grid size.
x = np.linspace(5000, 5000 + dx * nx, nx)
z = np.linspace(0, dz * nz, nz)
true_model   = resample_marmousi_model(x, z, marmousi_model)
vp_true   = vel_model['vp'].T
vs_true   = vel_model['vs'].T
rho_true  = np.power(vp_true, 0.25) * 310

# Initialize primary wave velocity (vp) and density (rho) for the model.
smooth_model= get_smooth_marmousi_model(vel_model,gaussian_kernel=4,mask_extra_detph=10)
vp_init     = smooth_model['vp'].T
vs_init     = smooth_model['vs'].T
rho_init    = np.power(vp_init, 0.25) * 310
vp_init[:10]= vp_true[:10] 
vs_init[:10]= vs_init[:10]
rho_init[:10]=rho_true[:10] 
```

An `IsotropicElasticModel` is initialized with the given parameters. It's important to note that the key parameter `vp_grad=True` and `vs_grad=True` indicates that the inversion process will update the velocity model (`vp` and `vs`). These parameter are crucial in full waveform inversion (FWI) since it allows the model to learn and adjust the velocity profile (`vp` and `vs`) during the inversion. On the other hand,  `rho_grad=False` ensures that the density model (`rho`) remains fixed during the inversion, and is not updated by the optimization process. If defined the `water_layer_mask`, the masked model will not update along with the inversion process.

```python
water_layer_mask = np.zeros((nz,nx))
water_layer_mask[:10] = 1

# processing the water layer
model = IsotropicElasticModel(
                ox,oz,nx,nz,dx,dz,
                vp_init,vs_init,rho_init,
                vp_bound =[vp_true.min(),vp_true.max()],
                vs_bound =[vs_true.min(),vs_true.max()],
                vp_grad = True,vs_grad = True, rho_grad=False,
                auto_update_rho=False,auto_update_vp=False,
                free_surface=free_surface,
                abc_type="PML",abc_jerjan_alpha=0.007,nabc=nabc,
                water_layer_mask=water_layer_mask,
                device=device,dtype=dtype)

# Save the initialized model to a file for later use.
model.save(os.path.join(project_path, "model/init_model.npz"))

# Print the model's representation for verification.
print(model.__repr__())

model._plot_vp_vs_rho(figsize=(12,5),wspace=0.2,cbar_pad_fraction=0.18,cbar_height=0.04,cmap='coolwarm',show=True)
```

----

### 2.4 Observed System Definition
The definition of observed system is tha same to the forward modeling process.

**Source**:

```python
# Define source positions in the model
src_z = np.array([2 for i in range(2,nx-2,5)]) 
src_x = np.array([i for i in range(2,nx-2,5)])

# Generate wavelet for the source
src_t, src_v = wavelet(nt, dt, f0, amp0=1)  # Create time and wavelet amplitude
src_v = integrate.cumtrapz(src_v, axis=-1, initial=0)  # Integrate wavelet to get velocity

source = Source(nt=nt, dt=dt, f0=f0)  # Initialize source object

# Method 1: Add multiple sources at once (commented out)
# source.add_sources(src_x=src_x, src_z=src_z, src_wavelet=src_v, src_type='mt', src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))

# Method 2: Loop through each source position to add them individually
for i in range(len(src_x)):
    source.add_source(src_x=src_x[i], src_z=src_z[i], src_wavelet=src_v, src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))

# Plot the wavelet used in the source
source.plot_wavelet(save_path=os.path.join(project_path, "survey/wavelets.png"))
```

**Receiver**:
```python
# Define receiver positions in the model
rcv_z = np.array([2   for i in range(0,nx,1)])
rcv_x = np.array([j   for j in range(0,nx,1)])

receiver = Receiver(nt=nt, dt=dt)  # Initialize receiver object

# Method 1: Add all receivers at once (commented out)
# receiver.add_receivers(rcv_x=rcv_x, rcv_z=rcv_z, rcv_type='pr')

# Method 2: Loop through each receiver position to add them individually
for i in range(len(rcv_x)):
    receiver.add_receiver(rcv_x=rcv_x[i], rcv_z=rcv_z[i], rcv_type="pr")
```

**Survey**：
```python
# Create a survey object using the defined source and receiver
survey = Survey(source=source, receiver=receiver)

# Print a representation of the survey object to check its configuration
print(survey.__repr__())
```

### 2.5 Define the Propagator & Inversion

In this step, we set up the forward propagation and inversion process to simulate how the Isotropic elastic waves travel through the model, record the waveforms, and then use these recorded waveforms for inversion.

#### (1) Forward Propagation
To simulate the isotropic elastic wave propagation, we initialize the wave propagator using the previously defined model and survey configuration. The propagator will compute how waves travel through the model and interact with the boundaries.
```python
# Initialize the wave propagator using the specified model and survey configuration
F = ElasticPropagator(model,survey,device=device)
```

#### (2) Inversion setup 
For inversion, a series of functions and configurations need to be defined, including misfit functions, optimization functions, and gradient processing. These components guide the inversion process, helping the model update the velocity profile based on the observed data.

##### a. Misfit Function
The misfit function is used to quantify the difference between the observed and predicted data. 

```python
# Import the L2 misfit function for waveform inversion.
from ADFWI.fwi.misfit import Misfit_waveform_L2, Misfit_global_correlation

# Configure the loss function (misfit) for the inversion.
loss_fn = Misfit_waveform_L2(dt=1)
# Alternatively, you can use global correlation as a misfit function.
# loss_fn = Misfit_global_correlation(dt=1)
```

##### b. Optimizer

An optimizer is required to adjust the model parameters (`vp` in this case) during the inversion. We use the Adam optimizer for its efficiency in handling large parameter spaces and its adaptive learning rate. The learning rate is set initially to 10 but will be adjusted later through the scheduler.

```python
# Initialize the optimizer (Adam) for model parameters with a learning rate.
optimizer = torch.optim.Adam(model.parameters(), lr=10)
```

##### c. Learning Rate Scheduler
A scheduler is used to modify the learning rate during the training process. Here, a StepLR scheduler is employed, which reduces the learning rate by a factor of 0.75 every 100 steps to help refine the model as it converges.

```python
# Set up a learning rate scheduler to adjust the learning rate over time.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75, last_epoch=-1)
```

##### d. Gradient Processing
Gradient processing involves manipulating the gradients during the backpropagation step. In this case, a gradient mask is applied to the velocity model (`vp`), more operation can be found at `ADFWI.propagator.gradient_process.GradProcessor`

```python
# gradient processor
grad_mask = np.ones_like(vp_init)
grad_mask[:10] = 0
gradient_processor_vp = GradProcessor(grad_mask=grad_mask,forw_illumination=False)
gradient_processor_vs = GradProcessor(grad_mask=grad_mask,forw_illumination=False,grad_mute=10,grad_smooth=2,marine_or_land='marine')
gradient_processor = [gradient_processor_vp,gradient_processor_vs]
```

##### e. Load Observed Data
We load the observed data (measured waveforms) that the model will compare against during the inversion. The data is loaded into a `SeismicData` object, which holds the observed waveforms and makes them available for the inversion process.

```python
# Create an instance of SeismicData using the survey object.
d_obs = SeismicData(survey)
d_obs.load(os.path.join(project_path, "waveform/obs_data.npz"))
print(d_obs.__repr__())
```

#### (3) Elastic Full Waveform Inversion Loop
Once all the necessary components are defined, we initialize the `ElasticFWI` object, which orchestrates the full waveform inversion process. This object takes in the propagator, model, optimizer, loss function, observed data, gradient processor, and other settings.

```python
# Initialize the elastic full waveform inversion (FWI) object.
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
                    save_fig_epoch=1,
                    save_fig_path=os.path.join(project_path,"inversion"),
                    inversion_component=["vx","vz"]
                    )
```

The inversion process is initiated by running the forward method of the FWI object. This method takes several parameters such as the number of iterations, batch size and checkpoints. Notably, the **batch size** and **checkpoitns** are used for balancing the inversion efficiency and memory usage.

```python
iteration = 300
# Run the forward modeling for the specified number of iterations.
fwi.forward(iteration=iteration,fd_order=4,
                    batch_size=None,checkpoint_segments=4,
                    start_iter=0)

# Retrieve the inversion results: updated velocity and loss values.
iter_vp     = fwi.iter_vp
iter_vs     = fwi.iter_vs
iter_rho    = fwi.iter_rho
iter_loss   = fwi.iter_loss

# Save the iteration results to files for later analysis.
np.savez(os.path.join(project_path,"inversion/iter_vp.npz"),data=np.array(iter_vp))
np.savez(os.path.join(project_path,"inversion/iter_vs.npz"),data=np.array(iter_vs))
np.savez(os.path.join(project_path,"inversion/iter_rho.npz"),data=np.array(iter_rho))
np.savez(os.path.join(project_path,"inversion/iter_loss.npz"),data=np.array(iter_loss))
```