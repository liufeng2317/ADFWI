# Wave Equations

In the context of **ADFWI**, accurate wave propagation modeling is essential for the inversion process. These wave equations can be categorized into two main types: **acoustic wave equations** and **elastic wave equations**. Both types play crucial roles in accurately modeling wave propagation in various subsurface environments.


## 1. Acoustic Wave Equation

The acoustic wave equation describes the propagation of pressure waves in a compressible medium. It is a fundamental equation in seismic imaging and FWI.

### 1.1 **Mathematical Formulation**

The acoustic wave equation in its standard second-order form is:

```{math}
\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u
```

where:
- {math}` u(x, t) ` is the pressure or displacement field,
- {math}` c ` is the wave velocity,
- {math}` \nabla^2 ` is the Laplacian operator,
- {math}` t ` is time, and {math}` x ` represents spatial coordinates.

### 1.2 **First-Order Formulation**

In numerical implementations, the wave equation is often rewritten as a first-order hyperbolic system:

```{math}
\frac{\partial v_i}{\partial t} = \frac{1}{\rho} \frac{\partial p}{\partial x_i}
```
```{math}
\frac{\partial p}{\partial t} = \kappa \nabla \cdot \mathbf{v} + f
```

where:
- {math}` v_i ` is the particle velocity in the {math}` i `-th direction,
- {math}` p ` is the acoustic pressure,
- {math}` \rho ` is the medium density,
- {math}` \kappa = \rho c^2 ` is the bulk modulus,
- {math}` f ` is an external source term.

### 1.3 **Derivation**

The first-order wave equation is derived from:

1. **Momentum Conservation (Newton’s Second Law):**
   ```{math}
   \rho \frac{\partial v_i}{\partial t} = -\frac{\partial p}{\partial x_i}
   ```
   which describes how pressure gradients drive velocity changes.

2. **Mass Conservation (Continuity Equation):**
   ```{math}
   \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0
   ```
   Under small perturbations, we approximate:
   ```{math}
   \frac{\partial p}{\partial t} = \kappa \nabla \cdot \mathbf{v} + f
   ```
   which relates pressure changes to velocity divergence.

### 1.4 **Numerical Implementation**

In our numerical implementation, we employ the **staggered grid finite difference method** to accurately solve the acoustic wave equation. This approach improves numerical stability and accuracy by placing pressure and velocity components at staggered locations in the computational grid.

The staggered grid method discretizes the wave equation using a spatially offset grid configuration:
- Pressure (`p`) is defined at the cell center.
- Velocity components (`u`, `w`) are defined at the edges of the grid cells.
- The first-order spatial derivatives are computed using finite difference approximations, reducing numerical dispersion and improving accuracy.

To approximate the spatial derivatives, we use **fourth-order finite difference coefficients**:
- Primary coefficients:
  - {math}` c_1 = \frac{9}{8} `
  - {math}` c_2 = -\frac{1}{24} `
- These coefficients ensure a balance between accuracy and computational efficiency.

We utilize an explicit **second-order time-stepping** scheme based on the velocity-pressure formulation:
- The pressure field is updated using the velocity divergence.
- The velocity fields are updated using the pressure gradient.
- A Perfectly Matched Layer (PML) is incorporated to absorb outgoing waves and prevent reflections at boundaries.

### 1.5 **Numerical Dispersion**

Numerical dispersion arises in wave simulations due to the discretization of the wave equation in space and time. It occurs when the numerical solution does not accurately capture the true phase velocity of the waves, leading to distortions in wave propagation. This issue is particularly noticeable when using lower-order finite difference schemes or when the grid resolution is too coarse to adequately represent the wave features, especially for high-frequency waves.

Several strategies can help reduce numerical dispersion:
- **Higher-order spatial discretization**, such as fourth-order to tenth-order finite differences, improves accuracy in approximating spatial derivatives.
- **Increasing grid resolution** captures finer wave details, especially for high-frequency waves, although this increases computational cost.
- **Higher-order time-stepping methods** can further reduce dispersion, though at an increased computational cost.
- **Perfectly Matched Layers (PML)** can be applied at boundaries to absorb outgoing waves and prevent reflections, improving the accuracy of the simulation and minimizing boundary-related dispersion.

The **Courant-Friedrichs-Lewy (CFL) condition** establishes the relationship between grid resolution and time step size to ensure stability and accuracy. It is expressed as:

```{math}
\frac{c \Delta t}{\Delta x} \leq C
```

where:
- {math}` c ` is the wave speed,
- {math}` \Delta t ` is the time step size,
- {math}` \Delta x ` is the spatial grid spacing,
- {math}` C ` is a constant (typically less than or equal to 1 for stability).

By carefully choosing the discretization scheme, grid resolution, time step, and boundary conditions, numerical dispersion can be minimized, leading to more accurate and stable wave propagation simulations.

### 1.6 **Reference**
* **Acoustic Wave Equation**
  * [1] Alford, R., Kelly, K., & Boore, D. M. (1974). *Accuracy of finite-difference modeling of the acoustic wave equation*. Geophysics, 39 (6), 834–842.
  * [2] Schuster, G. T. (2017). *Seismic inversion*. Society of Exploration Geophysicists. doi:10.1190/1.9781560803423

* **Numerical Dispersion**
  * [1] Levander, A.R., 1988. *Fourth‐order finite‐difference P-SV seismograms*. GEOPHYSICS, 53, 1425–1436.
  * [2] Moczo, P., Robertsson, J.O.A. \& Eisner, L., 2007. *The finite-difference time-domain method for modeling of seismic wave propagation*. in Advances in Geophysics, Vol. 48, pp. 421–516, Elsevier.
  * [3] Virieux, J., 1986. *P-SV wave propagation in heterogeneous media: Velocity‐stress finite‐difference method*. GEOPHYSICS, 51, 889–901.


## 2. Elastic Wave Equation

The elastic wave equation governs the propagation of stress and strain in a solid elastic medium.

### 2.1 **Mathematical Formulation**

The elastic wave equation in its second-order form is:

```{math}
\rho \frac{\partial^2 u_i}{\partial t^2} = \frac{\partial \tau_{ij}}{\partial x_j} + f_i
```
where:
- {math}` u_i(x, t) ` is the displacement field in the {math}`i`-th direction,
- {math}` \rho ` is the density of the medium,
- {math}` \tau_{ij} ` is the stress tensor, 
- {math}` f_i ` is the external force density,
- {math}` t ` is time, and {math}` x ` represents spatial coordinates.

### 2.2 **First-Order Formulation**

To facilitate numerical solutions, the elastic wave equation is often reformulated into a first-order system:

```{math}
\rho \frac{\partial v_i}{\partial t} = \frac{\partial \tau_{ij}}{\partial x_j} + f_i,
```
```{math}
\tau_{ij} = C_{ijkl} \epsilon_{kl},
```

where:
- {math}` v_i ` is the velocity in the {math}`i`-th direction,
- {math}` \tau_{ij} ` is the stress tensor,
- {math}` C_{ijkl} ` is the elasticity tensor (which relates stress and strain), 
- {math}` \epsilon_{kl} ` is the strain tensor, defined as:
  
```{math}
\epsilon_{kl} = \frac{1}{2} \left( \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right)
```

#### **Elastic Moduli {math}` C_{ij} ` and Thomsen Parameters: ISO vs. TI Media**

In elastic media, the stiffness is represented by the fourth-order tensor {math}` C_{ijkl} `. By exploiting symmetry, we can express it in a simplified 6×6 matrix form (Voigt notation), {math}` C_{ij} `. Below are the simplified forms and parameter correspondences for isotropic (ISO) and transversely isotropic (TI) media (including both VTI and HTI). 

The Lame’s parameters ({math}` \lambda ` and {math}` \mu `) are preferred since these two are directly related to the P- and S-wave velocities. It makes more sense to choose the parameters with a physical interpretation, and thereby, the anisotropy parameters are generally expressed using the Thomsen’s parameters {math}`\epsilon, \gamma, \delta` and two velocities {math}`\alpha, \beta` to characterize weak anisotropy in exploration seismology

(1) **Isotropic elastic media**

For isotropic media, the elastic behavior is fully characterized by the Lamé constants {math}` \lambda ` and {math}` \mu `. The stiffness matrix is given by:

```{math}
C_{ij} (\text{ISO}) =
\begin{bmatrix}
\lambda + 2\mu & \lambda & \lambda & 0 & 0 & 0 \\
\lambda & \lambda + 2\mu & \lambda & 0 & 0 & 0 \\
\lambda & \lambda & \lambda + 2\mu & 0 & 0 & 0 \\
0 & 0 & 0 & \mu & 0 & 0 \\
0 & 0 & 0 & 0 & \mu & 0 \\
0 & 0 & 0 & 0 & 0 & \mu
\end{bmatrix}
```

In this case, the Thomsen parameters, which describe anisotropy, are all zero: {math}` \epsilon = 0 `, {math}` \gamma = 0 `, {math}` \delta = 0 `. The P-wave and S-wave velocities are expressed as:
- {math}` \alpha = \sqrt{\frac{\lambda + 2\mu}{\rho}} `
- {math}` \beta = \sqrt{\frac{\mu}{\rho}} `

(2) **Transversely Isotropic (TI) Media (VTI/HTI)**

For TI media, five independent parameters ({math}` C_{11}, C_{13}, C_{33}, C_{44}, C_{66}`, and {math}`C_{12} = C_{11} - 2C_{66}`) are needed. For a vertically transversely isotropic (VTI) medium, the stiffness matrix takes the form:

```{math}
C_{ij} (\text{VTI}) =
\begin{bmatrix}
C_{11} & C_{12} & C_{13} & 0 & 0 & 0 \\
C_{12} & C_{11} & C_{13} & 0 & 0 & 0 \\
C_{13} & C_{13} & C_{33} & 0 & 0 & 0 \\
0 & 0 & 0 & C_{44} & 0 & 0 \\
0 & 0 & 0 & 0 & C_{44} & 0 \\
0 & 0 & 0 & 0 & 0 & C_{66}
\end{bmatrix}
```

Thomsen’s parameters provide an intuitive description of weak anisotropy. For a VTI medium, they are defined as follows:

- **{math}` \alpha `:** (Reference P-wave velocity):Typically related to the vertical P-wave modulus:
  ```{math}
  \alpha = \sqrt{\frac{C_{33}}{\rho}}
  ```

- **{math}` β `:** (Reference S-wave velocity): Related to the vertical S-wave modulus:
  ```{math}
  \beta = \sqrt{\frac{C_{44}}{\rho}}
  ```

- **{math}` ε `:** (P-wave anisotropy): Measures the fractional difference between horizontal and vertical P-wave stiffness:
  ```{math}
  \epsilon = \frac{C_{11} - C_{33}}{2 C_{33}}
  ```

- **{math}` γ `:** (S-wave anisotropy): Quantifies the difference between the two shear moduli:
  ```{math}
  \gamma = \frac{C_{66} - C_{44}}{2 C_{44}}
  ```

- **{math}` δ `:** (Influences P-wave velocity variation with angle):
  ```{math}
  \delta = \frac{(C_{13} + C_{44})^2 - (C_{33} - C_{44})^2}{2C_{33}(C_{33} - C_{44})}
  ```

For a horizontally transversely isotropic (HTI) medium, the form of {math}` C_{ij} ` is similar but the roles of the vertical and horizontal directions are interchanged. Consequently, the definitions of {math}` \alpha `, {math}` β `, and the Thomsen parameters are adapted to refer to the appropriate (vertical) reference directions.



### 2.3 **Derivation**

The first-order elastic wave equation can be derived from the following physical principles:

1. **Momentum Conservation (Newton's Second Law):**

```{math}
\rho \frac{\partial v_i}{\partial t} = \frac{\partial \tau_{ij}}{\partial x_j}
```
This equation describes how stress gradients influence the velocity field.

2. **Stress-Strain Relationship:**
The stress tensor {math}`\tau_{ij}` is related to the strain tensor {math}`\epsilon_{kl}` via the elasticity tensor {math}`C_{ijkl}`:

```{math}
\tau_{ij} = C_{ijkl} \epsilon_{kl}
```


3. **Strain Rate (Time Derivative of Strain):**
The strain rate {math}`\frac{\partial \epsilon_{kl}}{\partial t}` is computed from the velocities:

```{math}
\frac{\partial \epsilon_{kl}}{\partial t} = \frac{1}{2} \left( \frac{\partial v_k}{\partial x_l} + \frac{\partial v_l}{\partial x_k} \right)
```


### 2.4 **Numerical Implementation**

In our numerical implementation, we utilize the **staggered grid finite difference method** to solve the elastic wave equation. This method ensures high accuracy and stability by placing velocity and stress components at staggered locations in the computational grid.

We provide a range of spatial discretization schemes, from **4th-order to 10th-order** finite difference approximations, allowing for a flexible trade-off between accuracy and computational cost. The choice of higher-order schemes improves the precision of the spatial derivatives and **reduces numerical dispersion**, leading to more accurate wave propagation simulations.


The discretization of the elastic wave equations using the staggered grid method is as follows:
- **Velocity components** ({math}`v_x, v_z`) are defined at the edges of the grid cells.
- **Stress components** ({math}`\tau_{xx}, \tau_{xz}, \tau_{zz}, \tau_{zx}`) are defined at the cell centers.
- The **first-order spatial derivatives** are approximated using finite difference schemes of up to **tenth-order**, depending on the desired level of accuracy.

For the time-stepping scheme, we employ a **second-order explicit time-stepping method**, which provides a balance of computational efficiency and accuracy. The time evolution of the velocity and stress fields is updated sequentially:
- **Velocity fields** are updated based on the pressure gradient.
- **Stress fields** are updated using the velocity divergence.

### 2.5 **Reference**
* **Elastic Wave Equation**
  * [1] Virieux, J. (1986). *P-sv wave propagation in heterogeneous media: Velocity-stress finite-difference method*. Geophysics, 51 (4), 889–901. doi: 10.1190/1.1442147
  * [2] Levander, A. R. (1988). *Fourth-order finite-difference p-sv seismograms*. Geophysics, 53 (11), 1425–1436. doi: 10.1190/1.1442422
  * [3] Li, L., Tan, J., Zhang, D., Malkoti, A., Abakumov, I., & Xie, Y. (2021). Fdwave3d: *A matlab solver for the 3d anisotropic wave equation using the finite-difference method*. Computational Geosciences, 25 (5), 1565–1578. doi: 10.1007/s10596-021-10060-3