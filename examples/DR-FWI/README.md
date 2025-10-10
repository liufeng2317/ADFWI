
---

## DR-FWI: Deep Reparameterization for Full Waveform Inversion

:newspaper: **Latest Update (April 2025)**  
Our new preprint **Deep Reparameterization for Full Waveform Inversion: Architecture Benchmarking, Robust Inversion, and Multiphysics Extension** is now available on [arXiv](https://arxiv.org/abs/2504.17375).

---

## 📰 Introduction

**DR-FWI** introduces a deep learning-based reparameterization framework for Full Waveform Inversion (FWI), replacing direct parameter optimization with neural representations. By encoding subsurface structures into neural networks, DR-FWI enables more flexible, robust, and accurate inversion. Key contributions include:

### 1. 🔧 Architecture Benchmarking and Reference Integration
We systematically benchmark multiple neural architectures (CNN, MLP, U-Net) and strategies for integrating reference velocity models:
- **Shallow CNNs** outperform deeper variants and traditional FWI, capturing geological complexity more effectively.
- **Stepwise embedding** of reference models consistently improves inversion accuracy over direct model superposition.
- These insights guide practical architecture design and reference incorporation in real-world scenarios.

### 2. 🛡️ Robust Inversion under Sparse and Noisy Conditions
DR-FWI remains highly effective even under challenging conditions:
- Works well with extremely **sparse data** (e.g., 10 sources × 20 receivers).
- Maintains performance under **strong noise** (e.g., Gaussian noise with 6× standard deviation).
- Significantly outperforms conventional FWI in accuracy and stability, enabling applications in data-limited and cost-sensitive environments.

### 3. 🔁 Multiphysics Joint Inversion via Adaptive Backbone–Branch Network
We propose a unified **backbone–branch** network for multiparameter inversion:
- A **shared backbone** captures common structural features.
- Separate **branches** learn parameter-specific details (e.g., $v_p$, $v_s$, $\rho$), with normalization tailored to each.
- Demonstrated on synthetic and Marmousi2 models, this architecture significantly reduces parameter crosstalk, offering a scalable solution for multiphysics joint inversion.

### 4. ⚙️ Mechanistic Insights into Deep Reparameterization

We reveal the underlying mechanism that explains why deep reparameterization enhances FWI stability and accuracy. Through spectral dynamics analysis, DR-FWI is shown to impose implicit frequency regularization, consistent with the “spectral bias” observed in deep neural networks.
- **Progressive frequency learning**: Neural representations first reconstruct low-wavenumber (large-scale) structures and gradually recover high-wavenumber details, forming a natural low-to-high frequency learning sequence.
- **Implicit regularization**: This spectral bias acts as a built-in regularizer, preventing convergence to local minima and improving robustness against noise and poor initial models.
- **Architecture-dependent dynamics**: CNNs and U-Nets exhibit stronger low-frequency preference than MLPs, leading to smoother and more geologically consistent inversion results.

These findings bridge empirical observations with theoretical understanding, demonstrating that deep reparameterization functions as an implicit hierarchical spectral regularizer in FWI.

---

## 📊 Visual Overview

<div align="center">
  <img src="./Figures/Figure1_Physical_and_Data_Driven_FWI.png" width="600"/>
  <p><b>Figure 1:</b> Comparison of data-driven (e.g., generative and supervised learning) and physics-driven (e.g., traditional FWI, PINNs, IFWI) methods.</p>

  <img src="./Figures/Figure2_Deep_Reparameterization_Network_Workflow.png" width="600"/>
  <p><b>Figure 2:</b> DR-FWI workflow: replacing direct parameter optimization with deep reparameterization.</p>

  <img src="./Figures/Figure3_MultiParameter_Network.png" width="600"/>
  <p><b>Figure 3:</b> "Backbone–branch" architecture for joint inversion of multiple physical parameters.</p>

  <img src="./Figures/Figure10_FBC.png" width="400">
  <p><b>Figure 10:</b> Frequency-Band Correspondence (FBC) analysis for conventional FWI and DR-FWI with different reparameterization networks.</p>
</div>

---

## 📦 Code Availability

The code and testing examples will be released following peer review.  
For early access, please contact the corresponding author:  
📧 **liufeng2317@sjtu.edu.cn**