# Regularization Methods

The objective function for full waveform inversion (FWI) typically combines a data misfit term with a regularization term to stabilize the inversion process and incorporate prior information about the model. This can be formulated as:

```{math}
\mathbf{m}^* = \underset{\mathbf{m}}{\arg\min} \left( \mathcal{J}(\mathbf{m}) + \alpha \mathcal{R}(\mathbf{m}) \right),
```
where {math}`\mathcal{J}(\mathbf{m})` represents the data misfit term, and {math}`\mathcal{R}(\mathbf{m})` is the regularization term. The regularization parameter {math}`\alpha` controls the trade-off between data fidelity and model regularization. The choice of {math}`\mathcal{R}(\mathbf{m})` depends on the desired properties of the recovered model and can take various forms.

****

## 1. Tikhonov Regularization

Tikhonov regularization is a commonly used method to stabilize inverse problems by adding smoothness constraints to the model parameters. It is often applied in FWI to suppress high-frequency artifacts and enforce geological continuity.

### 1.1 Introduction

#### **(1) First-Order Tikhonov Regularization**  
First-order Tikhonov regularization penalizes the magnitude of the first-order differences (gradients) of the model parameters, enforcing smooth transitions between neighboring grid points:

```{math}
\mathcal{R}_{\text{Tikh}1}(\mathbf{m}) = \alpha_x \sum_{i=1}^{n_x-1} \sum_{j=1}^{n_z} (m_{i+1,j} - m_{i,j})^2 + \alpha_z \sum_{i=1}^{n_x} \sum_{j=1}^{n_z-1} (m_{i,j+1} - m_{i,j})^2
```

This regularization is commonly used to reduce high-frequency noise while preserving overall model trends.

#### **(2) Second-Order Tikhonov Regularization**  
Second-order Tikhonov regularization penalizes the magnitude of the second-order differences (curvature) of the model parameters:

```{math}
\mathcal{R}_{\text{Tikh}2}(\mathbf{m}) = \alpha_x \sum_{i=1}^{n_x-2} \sum_{j=1}^{n_z} (m_{i+2,j} - 2m_{i+1,j} + m_{i,j})^2 + \alpha_z \sum_{i=1}^{n_x} \sum_{j=1}^{n_z-2} (m_{i,j+2} - 2m_{i,j+1} + m_{i,j})^2
```

This regularization enforces smoothness by suppressing second-order variations, leading to a more continuous and geologically reasonable model with minimal artificial oscillations.

### 1.2 Pros and Cons

**Advantages:**  
- **Stabilization of Ill-posed Problems:** Helps to regularize the inversion process and prevent overfitting to noise.  
- **Enforcement of Smoothness:** Encourages geologically plausible models by reducing sharp variations in the model parameters.  
- **Flexibility:** Can be applied in different orders (e.g., first-order and second-order) to control different aspects of smoothness.  

**Disadvantages:**  
- **Loss of Resolution:** Excessive smoothing can remove meaningful small-scale structures, reducing inversion accuracy.  
- **Bias in Model Recovery:** May bias the solution toward overly smooth models that do not capture sharp geological boundaries.  
- **Tuning of Regularization Parameters:** The selection of optimal regularization coefficients ({math}`\alpha_x, \alpha_z`) requires careful calibration.  

### 1.3 **Comparison Between First-Order and Second-Order Tikhonov Regularization**
| Regularization Type | Effect | Typical Use Case |
|---------------------|--------|------------------|
| First-Order ({math}`\mathcal{R}_{\text{Tikh}1}`) | Penalizes sharp gradients, reducing noise and enforcing smoothness. | Suitable for suppressing high-frequency noise while preserving large-scale structures. |
| Second-Order ({math}`\mathcal{R}_{\text{Tikh}2}`) | Penalizes curvature, leading to a smoother and more continuous model. | Used when geological continuity is a priority, minimizing artificial oscillations. |

The choice between first-order and second-order Tikhonov regularization depends on the desired balance between resolution and smoothness in the inversion process.

****

## 2. Total Variation (TV) Regularization  

Total Variation (TV) regularization is widely used in inverse problems to preserve sharp boundaries while reducing noise. Unlike Tikhonov regularization, which enforces smoothness, TV regularization promotes piecewise-constant solutions, making it particularly useful in applications where geological structures contain sharp contrasts.  

### 2.1 Introduction  

#### **(1) First-Order TV Regularization**  
First-order total variation (TV1) regularization penalizes the absolute differences between neighboring model parameters:  

```{math}
\mathcal{R}_{\text{TV1}}(\mathbf{m}) = \alpha_x \sum_{i=1}^{n_x-1} \sum_{j=1}^{n_z} |m_{i+1,j} - m_{i,j}| + \alpha_z \sum_{i=1}^{n_x} \sum_{j=1}^{n_z-1} |m_{i,j+1} - m_{i,j}|
```

This formulation enforces sparsity in the model gradients, allowing for sharp transitions between regions with different properties, making it well-suited for geological structures with clear discontinuities, such as faults and layering.  

#### **(2) Second-Order TV Regularization**  
Second-order total variation (TV2) regularization penalizes the absolute second-order differences (curvature) in the model parameters:  

```{math}
\mathcal{R}_{\text{TV2}}(\mathbf{m}) = \alpha_x \sum_{i=1}^{n_x-2} \sum_{j=1}^{n_z} |m_{i+2,j} - 2m_{i+1,j} + m_{i,j}| + \alpha_z \sum_{i=1}^{n_x} \sum_{j=1}^{n_z-2} |m_{i,j+2} - 2m_{i,j+1} + m_{i,j}|
```

By penalizing curvature, TV2 regularization reduces high-frequency noise while maintaining significant structural variations in the model, making it particularly useful when smoothness is required without losing important geological features.  

### 2.2 Pros and Cons  

**Advantages:**  
- **Edge Preservation:** TV1 regularization maintains sharp boundaries in the model, preventing excessive smoothing of significant transitions.  
- **Robustness to Noise:** TV regularization is effective in reducing small-scale noise without blurring geological structures.  
- **Adaptive Smoothing:** Unlike Tikhonov regularization, which uniformly smooths the model, TV regularization selectively smooths regions without strong discontinuities.  

**Disadvantages:**  
- **Non-Differentiability:** The absolute value function in TV regularization makes gradient-based optimization more challenging and often requires specialized numerical methods.  
- **Blocky Artifacts:** TV1 regularization can produce staircasing effects, where the model appears as piecewise-constant regions rather than smooth transitions.  
- **Computational Cost:** Solving TV-regularized inverse problems can be computationally demanding due to the need for iterative solvers and total variation minimization techniques.  

### 2.3 **Comparison Between First-Order and Second-Order TV Regularization**  
| Regularization Type | Effect | Typical Use Case |  
|---------------------|--------|------------------|  
| First-Order ({math}`\mathcal{R}_{\text{TV1}}`) | Preserves sharp edges by penalizing first-order differences. | Suitable for models with abrupt changes, such as faulted geological formations. |  
| Second-Order ({math}`\mathcal{R}_{\text{TV2}}`) | Reduces noise while maintaining large-scale variations. | Useful when geological continuity is desired without excessive smoothing. |  

The selection between first-order and second-order TV regularization depends on the need for sharp boundaries versus smooth transitions in the inversion process.
