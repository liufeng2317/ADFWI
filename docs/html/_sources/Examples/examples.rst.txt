Examples
========

We provide a series of practical examples implemented in Jupyter notebooks, which can be found in our GitHub repository:

👉 `ADFWI GitHub Repository <https://github.com/liufeng2317/ADFWI>`_

These examples cover a range of forward and inversion modeling scenarios, helping users understand and implement acoustic and elastic full waveform inversion (FWI) using the `ADFWI` framework.

Basic Examples
--------------

The following fundamental examples introduce acoustic and isotropic elastic FWI:

.. toctree::
   :maxdepth: 1

   acoustic_fwi_tests
   elastic_fwi_tests

Advanced Examples
-----------------

Beyond the basic examples, we provide more advanced case studies that explore different modeling strategies, inversion techniques, and optimization methods in FWI.

Model-Based Tests
~~~~~~~~~~~~~~~~~

- **Acoustic Model Tests**: Forward modeling and inversion using widely used velocity models:
  
  - **Marmousi2 Model**: A complex geological benchmark model.
  - **Foothill Model**: Characterizing geological structures with faults and folds.
  - **SEAM-I Model**: A realistic subsurface model with intricate stratigraphy.
  - **Overthrust Model**: Simulating conditions with overthrust faulting.
  - **Anomaly Model**: A synthetic test model featuring localized anomalies.

- **Isotropic Elastic Model Tests**: Wave propagation and inversion in isotropic elastic media.
  
  - **Marmousi2 Model**
  - **Anomaly Model**

- **VTI Elastic Model Tests**: Full waveform inversion in transversely isotropic (VTI) media, illustrating anisotropic wave propagation effects.

Algorithmic Tests
~~~~~~~~~~~~~~~~~

- **Misfit Function Comparisons**: Evaluating different misfit measures, such as:

  - **L2 Norm**
  - **L1 Norm**
  - **T-distribution (StudentT)**
  - **Envelope**
  - **Global Correlation**
  - **Differentiable Dynamic Time Warping (soft-DTW)**
  - **Wasserstein distance with Sinkhorn divergence**
  - **Hybrid Strategies**

- **Regularization Techniques**: Investigating methods to stabilize inversion, including:

  - **Total Variation (TV)**
  - **Tikhonov Regularization**

- **Optimization Strategies**: Comparing various optimization techniques, such as:

  - **Stochastic Gradient Descent (SGD)**
  - **Average Stochastic Gradient Descent (ASGD)**
  - **Adaptive Gradient Algorithm (Adagrad)**
  - **Root Mean Square Propagation (RMSProp)**
  - **Adaptive Moment Estimation (Adam)**
  - **Adam with Weight Decay (AdamW)**
  - **Nesterov-accelerated Adam (NAdam)**
  - **Rectified Adam (RAdam)**
  - **L-BFGS**


- **Multi-Source Encoding**: Enhancing efficiency through multi-source encoding strategies.

Advanced Inversion Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Deep Reparameterization**: Leveraging deep learning-based parameterization to improve inversion stability and generalization.

- **Multi-Scale Strategies**: Implementing hierarchical inversion approaches, including:
  - **Frequency Continuation**
  - **Sequential Inversion with Progressive Refinement**

- **Real-World Case Studies**: Applying ADFWI to real-world seismic datasets, demonstrating the framework’s effectiveness in practical scenarios.

Each case study provides insights into the advantages and limitations of different FWI methodologies. Users are encouraged to explore these examples and customize them for their own research needs.

🚀 **Start exploring today and unleash the power of ADFWI for your seismic inversion tasks!**