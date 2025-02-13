<!--
 * @Author: LiuFeng(USTC) : 
   * liufeng2317@mail.ustc.edu.cn
   * liufeng1@pjlab.org.cn
 * @Date: 2023-07-03 11:16:43
 * @LastEditors: LiuFeng
 * @LastEditTime: 2024-01-02 13:16:52
 * @FilePath: /ADFWI/README.md
 * @Description: 
 * Copyright (c) 2024 by liufeng2317 email: liufeng1@pjlab.org.cn, All Rights Reserved.
-->

<div align="center">
  <img height="150" src="./docs/Md_img/ADFWI-logo.png" />
</div>
<br/>
<div align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=liufeng2317.liufeng2317&" />
  <img src="https://img.shields.io/github/stars/liufeng2317/ADFWI" />
  <img src="https://img.shields.io/github/forks/liufeng2317/ADFWI" />
</div>
<div align="center">
  <img src="https://img.shields.io/github/license/liufeng2317/ADFWI" />
  <img src="https://img.shields.io/github/repo-size/liufeng2317/ADFWI"/>
  <img src="https://img.shields.io/github/last-commit/liufeng2317/ADFWI"/>
  <img src="https://img.shields.io/badge/lauguage-python-%233572A5"/>
  <img src="https://img.shields.io/badge/lauguage-jupyter-%23F37626"/>
  <img src="https://img.shields.io/badge/lauguage-shell-%2389E051"/>
</div>
<h1 align="center">Automatic Differentiation-Based Full Waveform Inversion</h1>

- [👩‍💻 Introduction](#-introduction)
- [⚡️ Installation](#️-installation)
- [👾 Examples](#-examples)
  - [1. Gradient Comparation between AD \& Central Difference](#1-gradient-comparation-between-ad--central-difference)
  - [2. Iso-acoustic Model Tests](#2-iso-acoustic-model-tests)
  - [3. Iso-elastic \& VTI-elastic Model Tests](#3-iso-elastic--vti-elastic-model-tests)
  - [4. Misfits Tests](#4-misfits-tests)
  - [5. Optimizer Tests](#5-optimizer-tests)
  - [6. Regularization Methods](#6-regularization-methods)
  - [7. Multi-Scale Strategy in FWI](#7-multi-scale-strategy-in-fwi)
  - [8. Deep Reparameterization](#8-deep-reparameterization)
  - [9. Uncertainty Estimation Using Deep Neural Networks (DNNs)](#9-uncertainty-estimation-using-deep-neural-networks-dnns)
- [📝 Special Features](#-special-features)
- [⚖️ LICENSE](#️-license)
- [🗓️ To-Do List](#️-to-do-list)
- [🔰 Contact](#-contact)

---

## 👩‍💻 Introduction
&emsp;&emsp;**ADFWI** is an open-source framework for high-resolution subsurface parameter estimation by minimizing discrepancies between observed and simulated waveform. Utilizing automatic differentiation (AD), ADFWI **simplifies the derivation and implementation of full waveform inversion (FWI)**, enhancing the design and evaluation of methodologies. It supports wave propagation in various media, including isotropic acoustic, isotropic elastic, and both vertical transverse isotropy (VTI) and horizontal transverse isotropy (HTI) medias.

&emsp;&emsp;In addition, **ADFWI** provides a comprehensive collection of **Objective functions**, **regularization techniques**, **optimization algorithms**, and **deep neural networks**. This rich set of tools facilitates researchers in conducting experiments and comparisons, enabling them to explore innovative approaches and refine their methodologies effectively.

![ADFWI](./docs/Md_img/Figure1-AISWIT-Workflow.png)
![NNFWI](./docs/Md_img/Figure2_DIP_process.png)

---

## ⚡️ Installation

To install ADFWI, please follow these steps:

1. **Ensure Prerequisites**  
   - **Python 3.8+**: [Python Downloads](https://www.python.org/downloads/).

2. **Create a Virtual Environment (Optional)**
   It is recommended to create a virtual environment using `conda`:
   ```bash
   conda create --name adfwi-env python=3.8
   conda activate adfwi-env
   ```

3. **Install Required Packages**
- Method 1: **Clone the github Repository**
  This method provides the latest version, which may be more suitable for your research:
    ```bash
    git clone https://github.com/liufeng2317/ADFWI.git
    cd ADFWI
    ```
    Then, install the necessary packages:
    ```bash
    pip install -r requirements.txt
    ```
- Method 2: Install via pip
  Alternatively, you can directly install ADFWI from PyPI:
  ```bash
    pip install ADFWI-Torch
  ```
---

## 👾 Examples

Details can be found at the codebook.

### 1. Gradient Comparation between AD & Central Difference
A comparative analysis of gradient calculations obtained through AD and the central difference method.

<div style="text-align: center;">
<table>
    <tr>
        <th style="text-align: center;">Test Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Acoustic (vp)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/gradient_checking/Acoustic-Marmousi2/03_1_compare_gradient.ipynb">Codes</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <!-- <details>
                <summary>Gradient Comparation</summary> -->
                <img src="./examples/gradient_checking/Acoustic-Marmousi2/data/Figure_S1_acoustic_Gradient_Cmp.png" alt="Marmousi2GradientCmp" style="max-width: 120px; height: auto; max-height: 120px;" />
            <!-- </details> -->
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Elastic (vp/vs/rho)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/gradient_checking/Elastic-Marmousi2/03_1_compare_gradient.ipynb">Codes</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Gradient Comparation</summary>
                <img src="./examples/gradient_checking/Elastic-Marmousi2/data/Elastic_vp_grad.png" alt="Marmousi2GradientCmp" style="max-width: 120px; height: auto; max-height: 120px;" />
                <img src="./examples/gradient_checking/Elastic-Marmousi2/data/Elastic_vs_grad.png" alt="Marmousi2GradientCmp" style="max-width: 120px; height: auto; max-height: 120px;" />
                <img src="./examples/gradient_checking/Elastic-Marmousi2/data/Elastic_rho_grad.png" alt="Marmousi2GradientCmp" style="max-width: 120px; height: auto; max-height: 120px;" />
            </details>
        </td>
    </tr>
</table>
</div>

### 2. Iso-acoustic Model Tests

<div style="text-align: center;">
<table>
    <tr>
        <th style="text-align: center;">Test Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Example's Path</th>
        <th style="text-align: center;">Example's Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Marmousi2  (low resulotion)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;"><a href="./examples/acoustic/01-model-test/01-Marmousi2/02_inversion.ipynb">Example-Marmousi2 (low)</a></td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/01-model-test/01-Marmousi2/data/inversion/inversion_process.gif" alt="Marmousi2" style="max-width: 300px; height: auto;"/>
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Marmousi2 (high resulotion)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;"><a href="./examples/acoustic/01-model-test/06-LargeScaleTest-Marmousi2/02_inversion.ipynb">Example-Marmousi2 (high)</a></td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/01-model-test/06-LargeScaleTest-Marmousi2/data/inversion/inversion_process.gif" alt="Marmousi2" style="max-width: 300px; height: auto;"/>
            </details>
        </td>
    </tr>
        <tr>
        <td style="text-align: center; vertical-align: middle;">Marmousi2 (vp and rho)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;"><a href="./examples/acoustic/01-model-test/08-Marmousi2-vp-rho/02_inversion.ipynb">Example-vp & rho</a></td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/01-model-test/08-Marmousi2-vp-rho/data/inversion/inversion_process.gif" alt="Anomaly" style="max-width: 300px; height: auto;"/>
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">FootHill (low resulotion)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;"><a href="./examples/acoustic/01-model-test/02-FootHill/02_inversion.ipynb">Example-FootHill (low)</a></td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/01-model-test/02-FootHill/data/inversion/inversion_process.gif" alt="FootHill" style="max-width: 300px; height: auto;"/>
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">FootHill (high resulotion)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;"><a href="./examples/acoustic/01-model-test/07-LargeScaleTest-FootHill/02_inversion.ipynb">Example-FootHill (high)</a></td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/01-model-test/07-LargeScaleTest-FootHill/data/inversion/inversion_process.gif" alt="FootHill" style="max-width: 300px; height: auto;"/>
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">SEAM-I</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;"><a href="./examples/acoustic/01-model-test/03-SEAM-I/02_inversion.ipynb">Example-SEAM-I</a></td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/01-model-test/03-SEAM-I/data/inversion/inversion_process.gif" alt="SEAM-I" style="max-width: 300px; height: auto;"/>
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Overthrust-offshore</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;"><a href="./examples/acoustic/01-model-test/04-Overthrust-offshore/02_inversion.ipynb">Example-Overthrust-offshore</a></td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/01-model-test/04-Overthrust-offshore/data/inversion/inversion_process.gif" alt="Overthrust-offshore" style="max-width: 300px; height: auto;"/>
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Anomaly</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;"><a href="./examples/acoustic/01-model-test/05-Anomaly/02_inversion.ipynb">Example-Anomaly</a></td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/01-model-test/05-Anomaly/data/inversion/inversion_process.gif" alt="Anomaly" style="max-width: 300px; height: auto;"/>
            </details>
        </td>
    </tr>
</table>
</div>

****

### 3. Iso-elastic & VTI-elastic Model Tests
<div style="text-align: center;">
<table>
    <tr>
        <th style="text-align: center;">Test Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Iso-elastic Anomaly</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/elastic/Iso-elastic-Anomaly/02_inversion.ipynb">Example-Anomaly</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/elastic/Iso-elastic-Anomaly/data/inversion-Adam/inversion_process.gif" alt="Marmousi2" style="max-height: 600px; width: auto; " />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Iso-elastic Marmousi2-1</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/elastic/Iso-elastic-Marmousi2-shotTop-recTop/02_inversion.ipynb">Shot & Rec on Surface </a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/elastic/Iso-elastic-Marmousi2-shotTop-recTop/data/inversion_tv2/inversion_process.gif" alt="Marmousi2" style="max-height: 600px; width: auto; " />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Iso-elastic Marmousi2-2</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/elastic/Iso-elastic-Marmousi2-shotWater-recWater/02_inversion.ipynb">Shot & Rec Underwater</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/elastic/Iso-elastic-Marmousi2-shotWater-recWater/data/inversion/inversion_process.gif" alt="Marmousi2" style="max-height: 600px; width: auto; " />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">VTI-elastic Anomaly-1</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/elastic/VTI-elastic-Anomaly-eps/02_inversion.py">Inv Epsilon</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/elastic/VTI-elastic-Anomaly-eps/data/inversion/inversion_process.gif" alt="Anomaly" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">VTI-elastic Anomaly-2</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/elastic/VTI-elastic-Anomaly-eps-delta/02_inversion.py">Inv Epsilon & Delta</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/elastic/VTI-elastic-Anomaly-eps-delta/data/inversion/inversion_process.gif" alt="Anomaly" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
</table>
</div>

****

### 4. Misfits Tests

We assess the convexity of different objective functions by simulating seismic records using shifted wavelets. The following table summarizes the results and provides examples for further exploration.

<table>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Convexity</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/00-Ricker-Test/01_misfit_wavelets-shift.ipynb">Example-Ricker Shift</a><br><br>
            <a href="./examples/acoustic/02-misfit-functions-test/00-Ricker-Test/02_misfit_wavelets-shift_and_Amplitude.ipynb">Example-Ricker Shift & vary Amplitude</a><br><br>
            <a href="./examples/acoustic/02-misfit-functions-test/00-Ricker-Test/03_misfit_wavelets-shift_and_f0.ipynb">Example-Ricker Shift & domain Frequency</a><br><br>
            <a href="./examples/acoustic/02-misfit-functions-test/00-Ricker-Test/04_misfit_wavelets-shift_and_Gaussian_noise.ipynb">Example-Ricker Shift & Gaussian noise</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <img src="./examples/acoustic/02-misfit-functions-test/00-Ricker-Test/Figures/misfit_shift.png" alt="Ricker-cmp" style="max-width: 300px; height: auto; " />
        </td>
    </tr>
</table>



It is important to note that we present the performance of various objective functions under **poorer initial models**. When using better initial model conditions, each objective function demonstrates improved performance. Relevant results can be found in [Better Initial Model](./examples/acoustic/02-misfit-functions-test/01-Marmousi2-Test/).


<table style="width: 100%; table-layout: fixed;">
    <tr>
        <td style="text-align: center; vertical-align: middle;">
            <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/TestWithDiffMisfit.png" 
                 alt="Optimized_Model" 
                 style="max-height: 150px; width: auto; max-height: 200px;" />
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/MisfitTest_SSIM.png" 
                 alt="Optimized_Model_misfits" 
                 style="max-height: 150px; width: auto; max-height: 200px;" />
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/MisfitTest_MAPE.png" 
                 alt="Optimized_Model_misfits" 
                 style="max-height: 150px; width: auto; max-height: 200px;" />
        </td>
    </tr>
</table>



<div style="text-align: center;">
<table>
    <tr>
        <th style="text-align: center;">Test Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">L1-norm</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/02_inversion_L1.py">Example-L1</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/inversion-L1/inversion_process.gif" alt="L1" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">L2-norm</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/02_inversion_L2.py">Example-L2</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/inversion-L2/inversion_process.gif" alt="L2" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">T-distribution (StudentT)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/02_inversion_StudentT.py">Example-StudentT</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/inversion-StudentT/inversion_process.gif" alt="StudentT" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Envelope</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/02_inversion_Envelope.py">Example-Envelope</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/inversion-Envelope/inversion_process.gif" alt="Envelope" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Global Correlation (GC)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/02_inversion_GC.py">Example-GC</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/inversion-GC/inversion_process.gif" alt="GC" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Soft Dynamic Time Warping (soft-DTW)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/02_inversion_SoftDTW.py">Example-SoftDTW</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/inversion-SoftDTW/inversion_process.gif" alt="SoftDTW" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Wasserstein Distance with Sinkhorn</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/02_inversion_WassersteinSinkhorn.py">Example-Wasserstein</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/inversion-Wasserstein_Sinkhorn/inversion_process.gif" alt="Wasserstein" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Hybrid Misfit: Envelope & Global Correlation (WECI)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/02_inversion_WECI.py">Example-WECI</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/inversion-WECI/inversion_process.gif" alt="WECI" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
</table>
</div>


The following misfit functions are still in beta version and are undergoing further development and validation. Their performance and reliability will be evaluated in future studies.

<div style="text-align: center;">
<table style="width: 100%;">
    <tr>
        <th style="text-align: center;">Misfit Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Travel Time</td>
        <td style="text-align: center; vertical-align: middle;">🛠️</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/02_inversion_TravelTime.py">Example-TravelTime</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/data/inversion-TravelTime/inversion_process.gif" alt="TravelTime" style="max-width: 300px; height: auto;" />
                <!-- 🖼️ Image under development -->
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Normalized Integration Method (NIM)</td>
        <td style="text-align: center; vertical-align: middle;">🛠️</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/02-misfit-functions-test/02-Marmousi2-Test2/02_inversion_NIM.py">Example-NIM</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <!-- <img src="#" alt="TravelTime" style="max-width: 300px; height: auto;" /> -->
                🖼️ Image under development
            </details>
        </td>
    </tr>
</table>
</div>

***

### 5. Optimizer Tests

The results presented below specifically characterize the impact of using the `L2-norm objective function` in conjunction with the Marmousi2 model. It is important to note that the effects of different optimization algorithms may vary significantly when applied to other objective functions or models. Consequently, the findings should be interpreted within this specific context, and further investigations are recommended to explore the performance of these algorithms across a broader range of scenarios.

<table style="width: 100%; table-layout: fixed;">
    <tr>
        <td style="text-align: center; vertical-align: middle;">
            <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/Optimizer_Model.png" 
                 alt="Optimized_Model" 
                 style="height: 100%; width: auto; max-height: 200px;" />
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/Data_and_Model_Misfits.png" 
                 alt="Optimized_Model_misfits" 
                 style="height: 100%; width: auto; max-height: 200px;" />
        </td>
    </tr>
</table>


<div style="text-align: center;">
<table>
    <tr>
        <th style="text-align: center;">Test Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Stochastic Gradient Descent (SGD)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/02_inversion_SGD.py">Example-SGD</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/inversion-SGD/inversion_process.gif" alt="SGD" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Average Stochastic Gradient Descent (ASGD)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/02_inversion_ASGD.py">Example-ASGD</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/inversion-ASGD/inversion_process.gif" alt="ASGD" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Root Mean Square Propagation (RMSProp)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/02_inversion_RMSProp.py">Example-RMSProp</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/inversion-RMSProp/inversion_process.gif" alt="RMSProp" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Adaptive Gradient Algorithm (Adagrad)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/02_inversion_Adagrad.py">Example-Adagrad</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/inversion-Adagrad/inversion_process.gif" alt="Adagrad" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Adaptive Moment Estimation (Adam)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/02_inversion_Adam.py">Example-Adam</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/inversion-Adam/inversion_process.gif" alt="Adam" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Adam with Weight Decay (AdamW)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/02_inversion_AdamW.py">Example-AdamW</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/inversion-AdamW/inversion_process.gif" alt="AdamW" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Nesterov-accelerated Adam (NAdam)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/02_inversion_NAdam.py">Example-NAdam</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/inversion-NAdam/inversion_process.gif" alt="NAdam" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Rectified Adam (RAdam)</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/02_inversion_RAdam.py">Example-RAdam</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/inversion-RAdam/inversion_process.gif" alt="RAdam" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
</table>
</div>

<div style="text-align: center;">
<table>
    <tr>
        <th style="text-align: center;">Test Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">L-BFGS</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/02_inversion_LBFGS.py">Example-LBFGS</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/03-optimizer-test/01-Marmousi2-Test/data/inversion-LBFGS/inversion_process.gif" alt="L-BFGS" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
</table>
</div>

### 6. Regularization Methods

<table style="width: 100%; table-layout: fixed;">
    <tr>
        <td style="text-align: center; vertical-align: middle;">
            <img src="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/data/Regularization_Model.png" 
                 alt="Optimized_Model" 
                 style="height: 100%; height: auto; max-height: 200px;" />
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <img src="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/data/Regularization_Data_and_Model_misfit.png" 
                 alt="Optimized_Model_misfits" 
                 style="height: 100%; height: auto; max-height: 200px;" />
        </td>
    </tr>
</table>


<div style="text-align: center;">
<table>
    <tr>
        <th style="text-align: center;">Test Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">no-regularization</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/02_inversion_no_regularization.py">no-regular</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/data/inversion-no_regularization/inversion_process.gif" alt="no-regular" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Tikhonov-1st Order</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/02_inversion_Tikhonov-1order.py">Example-Tikhonov1</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/data/inversion-Tikhonov1-order/inversion_process.gif" alt="Tikhonov1" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Tikhonov-2nd Order</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/02_inversion_Tikhonov-2order.py">Example-Tikhonov2</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/data/inversion-Tikhonov2-order/inversion_process.gif" alt="Tikhonov2" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Total Variation-1st Order</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/02_inversion_TV-1order.py">Example-TV1</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/data/inversion-TV1-order/inversion_process.gif" alt="TV1" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Total Variation-2nd Order</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/02_inversion_TV-2order.py">Example-TV2</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/acoustic/04-regularization-techniques-test/01-Marmousi2-Test/data/inversion-TV2-order/inversion_process.gif" alt="TV2" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
</table>
</div>

----

### 7. Multi-Scale Strategy in FWI
Multi-scale strategies play a critical role in FWI as they help to mitigate **non-linearity** issues and enhance convergence, especially for complex models. Multi-scale strategies are currently in development to further improve robustness and efficiency.

<div style="text-align: center;">
<table>
    <tr>
        <th style="text-align: center;">Test Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Iso-elastic Marmousi2</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/multi-scale/Iso-elastic-Marmousi2-multifreq/02_inversion.py">Multi-freq (2Hz,3Hz,5Hz)</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/multi-scale/Iso-elastic-Marmousi2-multifreq/data/inversion/inversion_process.gif" alt="Marmousi2" style="max-height: 600px; width: auto; " />
            </details>
        </td>
    </tr>
</table>
</div>

<div style="text-align: center;">
<table style="width: 100%;">
    <tr>
        <th style="text-align: center;">Multi-Scale Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Multi-Offsets</td>
        <td style="text-align: center; vertical-align: middle;">🛠️</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="#">on-going</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                🖼️ Image under development
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">Multi-scale in Time</td>
        <td style="text-align: center; vertical-align: middle;">🛠️</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="#">on-going</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                🖼️ Image under development
            </details>
        </td>
    </tr>
</table>
</div>

----

### 8. Deep Reparameterization

<div style="text-align: center;">
<table>
    <tr>
        <th style="text-align: center;">Test Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">no-regularization</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/dip/DIP-ADFWI/01_Multi-CNN/02_inversion_no_regularization.py">no-regular</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/dip/DIP-ADFWI/01_Multi-CNN/data/inversion-no_regularization/inversion_process.gif" alt="no-regular" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">2-Layer CNN</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/dip/DIP-ADFWI/01_Multi-CNN/02_inversion_2layer-4-32.py">Example-2LayerCNN</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/dip/DIP-ADFWI/01_Multi-CNN/data/inversion-2layer-4-32/inversion_process.gif" alt="2LayerCNN" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">3-Layer CNN</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/dip/DIP-ADFWI/01_Multi-CNN/02_inversion_3layer-16-32-16.py">Example-3LayerCNN</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/dip/DIP-ADFWI/01_Multi-CNN/data/inversion-3layer-16-32-16/inversion_process.gif" alt="3LayerCNN" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">3-Layer Unet</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/dip/DIP-ADFWI/02_Unet/02_inversion_3layer_64channels.py">Example-3LayerUNet</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/dip/DIP-ADFWI/02_Unet/data/inversion-3layer-64channels/inversion_process.gif" alt="3LayerUNet" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">4-Layer Unet</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/dip/DIP-ADFWI/02_Unet/02_inversion_4layer_64channels.py">Example-4LayerUNet</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <details>
                <summary>Inversion Process</summary>
                <img src="./examples/dip/DIP-ADFWI/02_Unet/data/inversion-4layer-64channels/inversion_process.gif" alt="4LayerUNet" style="max-width: 300px; height: auto;" />
            </details>
        </td>
    </tr>
</table>
</div>

### 9. Uncertainty Estimation Using Deep Neural Networks (DNNs)

We employ DNNs derived from the Deep Image Prior (DIP) test described earlier, to perform uncertainty estimation. The variable `p` represents the dropout ratio applied during both training and inference to evaluate uncertainty.

<div style="text-align: center;">
<table>
    <tr>
        <th style="text-align: center;">Test Name</th>
        <th style="text-align: center;">Status</th>
        <th style="text-align: center;">Path</th>
        <th style="text-align: center;">Figure</th>
    </tr>
    <tr>
        <td style="text-align: center; vertical-align: middle;">2LayerCNN-uncertainty</td>
        <td style="text-align: center; vertical-align: middle;">✅</td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="./examples/dip/DIP-ADFWI/01_Multi-CNN/04_uncertainty_assesment.ipynb">Codes</a>
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <img src="./examples/dip/DIP-ADFWI/01_Multi-CNN/data/inversion-2layer-4-32/Uncertainty_evaluate.png" alt="2LayerCNN-uncertainty" style="max-width: 300px; height: auto;" />
        </td>
    </tr>
</table>
</div>

---

## 📝 Special Features

- **Deep Neural Network Integration**
  - **DNNs Reparameterization**: DNNs reparameterize the Earth model, introducing learnable regularization to improve the inversion process.
  - **Uncertainty estimation using Dropout**: Applied to assess inversion uncertainty by randomly dropping units of neural network.
  - **Multiphysics Joint Inversion (on-going)**: Neural networks are used to fuse data from different physical fields, enabling joint inversion for a more comprehensive and accurate Earth model (**new research is comming soon**).

- **Resource Management**
  - **Mini-batch**: Splitting datasets into mini-batches prevents loading the entire into memory at once.
  - **Checkpointing**: Instead of storing all intermediate results, only a few checkpoints are saved. During backpropagation, missing steps are recomputed, reducing memory usage at the cost of extra computation. ==We emphasise that this approach does not significantly reduce the inversion efficiency while drastically reducing the memory usage.==
  - Multi-Source encoding: a technique in FWI that combines multiple seismic sources into a single simulation to improve computational efficiency. 
  - **boundary saving (on-going)**: methods are being developed to efficiently reduce memory usage by saving only the wavefield boundaries during forward propagation instead of the entire wavefield, allowing for their later use in backpropagation.

- **Acceleration Methods**
  - **matrix operation**: Incorporates shots as extra dimensions into the matrix operation of forward modling, e.g. `[t,wavefiled] -> [shot, t, wavefield]`.
  - **GPU Acceleration**: Multi-GPU version is comming soon.
  - **JIT(Just-in-Time)**: Speeds up code execution by compiling Python code into optimized machine code at runtime, improving performance without modifying the original codebase.
  - **Reconstruction Using Lower-Level Language (C++) (on-going)**

- **Robustness and Portability**
  - Each of the method has proposed a code for testing.

---

## ⚖️ LICENSE

The **Automatic Differentiation-Based Full Waveform Inversion (ADFWI)** framework is licensed under the [MIT License](https://opensource.org/licenses/MIT). This license allows you to:

- **Use**: You can use the software for personal, academic, or commercial purposes.
- **Modify**: You can modify the software to suit your needs.
- **Distribute**: You can distribute the original or modified software to others.
- **Private Use**: You can use the software privately without any restrictions.

---

## 🗓️ To-Do List

- <details>
    <summary><b>Memory Optimization through Boundary and Wavefield Reconstruction</b></summary>
    <b>Objective</b>: Implement a strategy to save boundaries and reconstruct wavefields to reduce memory usage.  

    <b>Explanation</b>: This approach focuses on saving only the wavefield boundaries during forward propagation and reconstructing the wavefields as needed. By doing so, it aims to minimize memory consumption while ensuring the accuracy of wave propagation simulations, particularly in large-scale models.
    
    <b>Related Work</b>: 
    - [1] P. Yang, J. Gao, and B. Wang, *RTM using Effective Boundary Saving: A Staggered Grid GPU Implementation*, Comput. Geosci., vol. 68, pp. 64–72, Jul. 2014. doi: [10.1016/j.cageo.2014.04.004](https://doi.org/10.1016/j.cageo.2014.04.004).
    - [2] Wang, S., Jiang, Y., Song, P., Tan, J., Liu, Z., & He, B., 2023. *Memory Optimization in RNN-Based Full Waveform Inversion Using Boundary Saving Wavefield Reconstruction*, IEEE Trans. Geosci. Remote Sens., 61, 1–12. doi: [10.1109/TGRS.2023.3317529](https://doi.org/10.1109/TGRS.2023.3317529).
    </details>

- <details>
    <summary><b>C++ / C-Based Forward Propagator</b></summary>
    <b>Objective</b>: Develop a forward wave propagation algorithm using C++ or C.  
    
    <b>Explanation</b>: Implementing the forward propagator in lower-level languages like C++ or C will significantly enhance computational performance, particularly for large-scale simulations. The aim is to leverage the improved memory management, faster execution, and more efficient parallel computing capabilities of these languages over Python-based implementations.
  </details>

- <details>
    <summary><b><del>Resource Optimization for Memory Efficiency (2024/12)</del></b></summary>
    <b>Objective</b>: Reduce memory consumption for improved resource utilization.  

    <b>Explanation</b>: The current computational framework may encounter memory bottlenecks, especially when processing large datasets. Optimizing memory usage by identifying redundant storage, streamlining data structures, and using efficient algorithms will help in scaling up the computations while maintaining or even enhancing performance. This task is critical for expanding the capacity of the system to handle larger and more complex datasets.
  </details>

- <details>
    <summary><b><del>Custom Input Data Management System (2025/1)</del></b></summary>
    <b>Objective</b>: Develop a tailored system for managing input data effectively.  
    
    <b>Explanation</b>: A customized data management framework is needed to better organize, preprocess, and handle input data efficiently. This may involve designing workflows for data formatting, conversion, and pre-processing steps, ensuring the consistency and integrity of input data. Such a system will provide flexibility in managing various input types and scales, and it will be crucial for maintaining control over data quality throughout the project lifecycle.
  </details>

- <details>
    <summary><b>Enhanced Gradient Processing</b></summary>
    <b>Objective</b>: Implement advanced techniques for gradient handling.  

    <b>Explanation</b>: Developing a sophisticated gradient processing strategy will improve the inversion results by ensuring that gradients are effectively utilized and interpreted. This may include techniques such as gradient clipping, adaptive learning rates, and noise reduction methods to enhance the stability and convergence of the optimization process, ultimately leading to more accurate inversion outcomes.
  </details>

- <details>
    <summary><b><del>Multi-Scale Inversion Strategies (2024/11)</del></b></summary>
    <b>Objective</b>: Introduce multi-scale approaches for improved inversion accuracy. 

    <b>Explanation</b>: Multi-scale inversion involves processing data at various scales to capture both large-scale trends and small-scale features effectively. Implementing this strategy will enhance the robustness of the inversion process, allowing for better resolution of subsurface structures. Techniques such as hierarchical modeling and wavelet analysis may be considered to achieve this goal, thus improving the overall quality of the inversion results.
  </details>

- <details>
    <summary><b><del>Real Data Testing (2D acoustic land datasets, 2025/1)</del></b></summary>
    <b>Objective</b>: Evaluate the performance and robustness of the developed methodologies using real-world datasets. (we have varified this framework with a 2D land datasets)  
    <b>Explanation</b>: Conducting tests with actual data is crucial for validating the effectiveness of the Status algorithms. This will involve the following steps:
    
    1. <b>Dataset Selection</b>: Identify relevant real-world datasets that reflect the complexities of the target applications. These datasets should include diverse scenarios and noise characteristics typical in field data.

    2. <b>Preprocessing</b>: Apply necessary preprocessing techniques to ensure data quality and consistency. This may include data normalization, filtering, and handling missing or corrupted values.

    3. <b>Implementation</b>: Utilize the developed algorithms on the selected datasets, monitoring their performance metrics such as accuracy, computational efficiency, and convergence behavior.

    4. <b>Comparison</b>: Compare the results obtained from the Status methods against established benchmarks or existing methodologies to assess improvements.

    5. <b>Analysis</b>: Analyze the outcomes to identify strengths and weaknesses, and document any discrepancies or unexpected behaviors. This analysis will help refine the algorithms and inform future iterations.

    6. <b>Reporting</b>: Summarize the findings in a comprehensive report, detailing the testing procedures, results, and any implications for future work.

    This actual data testing phase is essential for ensuring that the developed methodologies not only perform well in controlled environments but also translate effectively to real-world applications. It serves as a critical validation step before broader deployment and adoption.
  </details>

---
## 🔰 Contact

This project was developed by **Feng Liu** at the University of Science and Technology of China (USTC) and Shanghai Jiao Tong University (SJTU). For any inquiries, please contact Liu Feng via email at: [liufeng2317@sjtu.edu.cn](mailto:liufeng2317@sjtu.edu.cn) or [liufeng2317@mail.ustc.edu.cn](mailto:liufeng2317@mail.ustc.edu.cn).



The related paper, [Automatic Differentiation-based Full Waveform Inversion with Flexible Workflows](https://arxiv.org/abs/2412.00486), is available on Arxiv (https://arxiv.org/abs/2412.00486). If you find ADFWI useful, please consider citing the following reference:
```
Liu, F., Li, H., Zou, G., & Li, J.. Automatic Differentiation-based Full Waveform Inversion with Flexible Workflows[J]. arXiv preprint arXiv:2412.00486, 2024.
```

or/and the software

```bibtex
@software{ADFWI_LiuFeng_2024,
  author       = {Feng Liu, Haipeng Li, GuangYuan Zou and Junlun Li},
  title        = {ADFWI},
  month        = dec,
  year         = 2024,
  version      = {v1.1.2},
  doi          = {10.5281/zenodo.14261243},
  url          = {https://doi.org/10.5281/zenodo.14261243},
}
```