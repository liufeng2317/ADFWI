# Objective Functions

## 1. Waveform-based

### 1.1 L2-norm

The **L2-norm** objective function is a fundamental metric in FWI that quantifies the misfit between observed and synthetic data by minimizing the squared differences. Its quadratic nature provides a smooth and convex optimization landscape, facilitating gradient-based inversion.

```{math}
\mathcal{J}_{L2}(\mathbf{m}) = \frac{1}{2} \sum_{s} \sum_{g} \int_{0}^{T} ||d_{obs}(s, r, t) - d_{cal}(\mathbf{m}; s, r, t)||^2 dt,
```
where {math}`d_{obs}(s, r, t)` and {math}`d_{cal}(\mathbf{m}; s, r, t)` represent the observed and synthetic data, respectively, for each shot ( {math}`s`) and receiver ( {math}`r`);  {math}`t` denotes the recording time,  {math}`T` denotes the maximum recording time, and  {math}`\mathbf{m}` denotes the model parameters. 

#### **Advantages**:
* **Mathematically well-posed**: Differentiable and convex, enabling stable gradient-based optimization.
* **Computationally efficient**: Simple and widely implemented in inversion frameworks.

#### **Disadvantages**:
* **Sensitive to noise**: Outliers and high-amplitude errors can dominate the misfit.
* **Risk of overfitting**: Can lead to an excessive fit to noisy data, reducing model robustness.

****

### 1.2 L1-norm

The **L1-norm** objective function is an alternative to the L2-norm in fwi, measuring the misfit using absolute differences instead of squared differences. It is less sensitive to large errors, making it more robust to noise and outliers.

```{math}
\mathcal{J}_{L1}(\mathbf{m}) = \sum_{s} \sum_{g} \int_{0}^{T} |d_{obs}(s, r, t) - d_{cal}(\mathbf{m}; s, r, t)| dt,
```
where the definitions are the same as those for  {math}`\mathcal{J}_{L2}(\mathbf{m})`.

#### **Advantages**:
* **Robustness to noise**: Reduces the influence of large errors and outliers compared to the L2-norm.
* **Enhanced resolution**: Can better preserve sharp model contrasts in inversion results.

#### **Disadvantages**:
* **Non-differentiability at zero**: Requires special treatment in gradient-based optimization.
* **Potentially slower convergence**: May lead to optimization challenges due to the lack of a smooth gradient.

****

### 1.3 T-distribution (StudentT)

The **Student’s T-norm** objective function is designed to improve robustness against noise and outliers by modeling the data misfit using a heavy-tailed Student’s t-distribution. Unlike the L2-norm, it mitigates the influence of large residuals by adaptively weighting them based on the degrees of freedom parameter.

```{math}
\mathcal{J}_{StudentT}(\mathbf{m}) = \sum_{s} \sum_{g} \int_{0}^{T} \frac{n+1}{2} \log\left[1 + \frac{1}{n \sigma^2} ||d_{obs}(s, r, t) - d_{cal}(\mathbf{m}; s, r, t)||^2\right] dt,
```
where  {math}`\sigma` is the scaling parameter and  {math}`n` denotes the degrees of freedom of the T-distribution (Aravkin et al., 2011; Guo et al., 2023).

#### **Advantages**:
* **Robust to outliers**: Reduces the impact of extreme residuals compared to the L2-norm.
* **Adaptive weighting**: Provides a balance between L1- and L2-norm behaviors, adjusting based on noise characteristics.

#### **Disadvantages**:
* **Additional hyperparameters**: Requires tuning of  {math}`\sigma` and  {math}`n`, which may impact performance.
* **Increased computational cost**: More complex than standard L1- and L2-norms, requiring additional calculations.

#### **Reference**
* [1] Aravkin, A., Van Leeuwen, T., & Herrmann, F. (2011). Robust full-waveform inversion using the student’s t-distribution. In Seg technical program expanded abstracts 2011 (pp. 2669–2673). Society of Exploration Geophysicists. doi: 10.1190/1.3627747
* [2] Guo, K., Zong, Z., Yang, J., & Tan, Y. (2023). Parametric elastic full waveform inversion with convolutional neural network. Acta Geophysica, 72 (2), 673–687. doi: 10.1007/s11600-023-01123-3

****

## 2. Waveform-attributes based

### 2.1 Envelope

The **Envelope** objective function is based on the comparison of the envelope of the observed and synthetic data. It is particularly useful for capturing the amplitude variations of the seismic waves, as it operates on the envelope rather than the raw signal. The envelope is obtained by combining the original data with its Hilbert transform, which provides a smooth representation of the signal's amplitude.

```{math}
\mathcal{J}_{Envelope}(\mathbf{m}) = \sum_{s} \sum_{g} \int_{0}^{T} ||E_{obs}^p(s, r, t) - E_{cal}^p(\mathbf{m}; s, r, t)||^2 dt,
```
where {math}`E_{obs}(t) = \sqrt{d_{obs}^2(t) + \widetilde{d}_{obs}^2(t)}` is the envelope of {math}`d_{obs}(t)`, and {math}`\widetilde{d}_{obs}(t)` is the Hilbert transform of {math}`d_{obs}(t)`; {math}`E_{syn}(t)` is similar to {math}`E_{obs}(t)` but for synthetic data; {math}`p` represents an operation on {math}`E(t)`, such as absolute values ({math}`p=1`) or squares ({math}`p=2`) (Bozda{\u{g}} et al., 2011; Wu et al., 2014).

#### **Advantages**:
* **Captures amplitude variations**: Focuses on the amplitude characteristics of the seismic signals, which can improve the accuracy of inversion in cases where phase information is less critical.
* **Improved robustness to phase shifts**: By operating on the envelope, it is less sensitive to phase mismatches, which can be particularly beneficial in noisy data.

#### **Disadvantages**:
* **Loss of phase information**: The envelope approach sacrifices phase information, which may be crucial in some applications.
* **Computational complexity**: Requires the computation of the Hilbert transform, which can be computationally expensive for large datasets.

#### **Reference**
* Bozda˘g, E., Trampert, J., & Tromp, J. (2011). Misfit functions for full waveform inversion based on instantaneous phase and envelope measurements. Geophysical Journal International, 185 (2), 845–870. doi: 10.1111/j.1365-246X.2011.04970.x
* Wu, R.-S., Luo, J., & Wu, B. (2014). Seismic envelope inversion and modulation signal model. Geophysics, 79 (3), WA13–WA24.

****

### 2.2 Global Correlation

The **Global Correlation** objective function measures the similarity between the observed and synthetic data by evaluating the zero-lag cross-correlation between their normalized waveforms. This approach focuses on the overall correlation between the waveforms rather than their individual amplitudes or phases.

```{math}
\mathcal{J}_{GC}(\mathbf{m}) = \sum_{s} \sum_{r} \int_{0}^{T} \left[ 1 - \hat{d}_{obs}(s, r, t) \cdot \hat{d}_{cal}(\mathbf{m}; s, r, t) \right] dt,
```
where {math}`\hat{d}_{obs}(s, r, t) = d_{obs}(s, r, t)/\left\| d_{obs}(s, r, t) \right\|` and {math}`\hat{d}_{cal}(\mathbf{m}; s, r, t) = d_{cal}(\mathbf{m}; s, r, t) / \left\| d_{cal}(\mathbf{m}; s, r, t) \right\|`.

#### **Advantages**:
* **Focus on waveform shape**: This method emphasizes the overall shape of the seismic waves, which can be more informative in cases where absolute amplitudes may vary.
* **Robust to amplitude discrepancies**: By normalizing the waveforms, the method is less sensitive to large amplitude variations, such as those caused by noise.

#### **Disadvantages**:
* **Sensitivity to phase shifts**: Although the method focuses on waveform shape, it may still be sensitive to small phase mismatches, particularly for strongly dispersive data.
* **May neglect amplitude information**: Normalization may ignore important amplitude variations, which could be crucial in some inversion problems.

****

## 3. Data-alignment based

### 3.1 Differentiable Dynamic Time Warping (soft-DTW)

Dynamic time warping (DTW) is a method used to measure the similarity between two time sequences. Given two time series {math}`X = [x_1, x_2, \ldots, x_N]` and {math}`Y = [y_1, y_2, \ldots, y_M]`, the DTW distance {math}`D(X, Y)` is defined as:
```{math}
    D(X, Y) = \min_{W} \left( \sum_{(i, j) \in W} d(x_i, y_j) \right),
```
where {math}`W` represents a warping path from {math}`(1,1)` to{math}`(N,M)`, and {math}`d(x_i, y_j)` denotes the distance between points {math}`x_i` and{math}`y_j` (typically the Euclidean distance). The cumulative distance matrix is updated using:
```{math}
    D(i, j) = d(x_i, y_j) + \min \{ D(i-1, j), D(i, j-1), D(i-1, j-1) \}.
```

In FWI, we calculate the DTW distance between the observed data {math}`d_{obs}(s, r, t)` and synthetic data {math}`d_{cal}(\mathbf{m}; s, r, t)` for each source-receiver pair {math}`(s, r)`. The DTW-based objective function is then defined as (Ma \& Hale, 2013; Cuturi \& Blondel, 2017; Chen et al., 2022):
```{math}
    \mathcal{J}_{DTW}(\mathbf{m}) = \sum_{s} \sum_{r} D(d_{obs}(s, r, t), d_{cal}(\mathbf{m}; s, r, t)).
```
The details of the calculations are as follows:
* **Initialization:** Construct the cumulative distance matrix {math}`D` with size {math}`(N+1) \times (M+1)` and initialize all elements to infinity, except {math}`D(0, 0) = 0`.
* **Update cumulative distance matrix:** For each {math}`(i, j)`, update the matrix using:
```{math}
\begin{aligned}
    D(i, j) &= d(d_{obs}(s, r, t_i), d_{cal}(\mathbf{m}; s, r, t_j)) \\
            &+ \min \left\{ D(i-1, j), D(i, j-1), D(i-1, j-1) \right\}.
\end{aligned}
```
* **Find the optimal path:** The optimal path is traced back from {math}`(N, M)` to {math}`(1, 1)`, accumulating the distances along the path.

This DTW-based objective function can handle time shifts and distortions between the observed and simulated data, providing a more robust and flexible measurement of misfits compared to the traditional waveform-difference-based objective functions.

#### **Advantages**:
* **Differentiable and smooth**: Enables gradient-based optimization, making it suitable for deep learning.
* **Robust to time shifts**: Handles temporal misalignment better.

#### **Disadvantages**:
* **Computationally expensive**: More costly than standard DTW, especially for long sequences.
* **Parameter sensitivity**: Performance depends on tuning the smoothness parameter.

#### **Reference**
* Ma, Y., & Hale, D. (2013). Wave-equation reflection traveltime inversion with dynamic warping and full waveform inversion. Geophysics, 78 (6), R223–R233
* Cuturi, M., & Blondel, M. (2017). Soft-dtw: A differentiable loss function for time-series. In Proceedings of the 34th international conference on machine learning (Vol. 70, pp. 894–903). PMLR.
* Chen, F., Peter, D., & Ravasi, M. (2022). Cycle-skipping mitigation using misfit measurements based on differentiable dynamic time warping. Geophysics, 87 (4), R325-R335. doi: 10.1190/geo2021-0598.1


### 3.2 Wasserstein distance with sinkhorn divergence (Wasserstein-Sinkhorn)

The Wasserstein-Sinkhorn distance is a measurement used to quantify the difference between two probability distributions. It combines the Wasserstein distance with the Sinkhorn regularization to balance computational efficiency and stability. Consider two discrete probability distributions {math}`\mu` and {math}`\nu`:
```{math}
    \mu = \sum_{i=1}^{n} \mu_i \delta_{x_i}, \quad \nu = \sum_{j=1}^{m} \nu_j \delta_{y_j},
```
where {math}`\delta_{x_i}` and {math}`\delta_{y_j}` are Dirac functions at {math}`x_i` and {math}`y_j`, respectively, and {math}`\mu_i` and {math}`\nu_j` are the weights (probability masses) at these locations. The Wasserstein distance is defined as:
```{math}
    W(\mu, \nu) = \min_{\gamma \in \Gamma(\mu, \nu)} \sum_{i=1}^{n} \sum_{j=1}^{m} \gamma_{ij} d(x_i, y_j),
```
where {math}`\Gamma(\mu, \nu)` denotes the set of all joint probability distributions satisfying the marginal constraints:
```{math}
    \sum_{j=1}^{m} \gamma_{ij} = \mu_i, \quad \sum_{i=1}^{n} \gamma_{ij} = \nu_j.
```

Sinkhorn regularization introduces an entropy regularization term to make the computation more efficient. With a regularization parameter {math}`\lambda > 0`, the Sinkhorn distance is defined as:
```{math}
    W_{\lambda}(\mu, \nu) = \min_{\gamma \in \Gamma(\mu, \nu)} \sum_{i=1}^{n} \sum_{j=1}^{m} \gamma_{ij} d(x_i, y_j) + \frac{1}{\lambda} \sum_{i=1}^{n} \sum_{j=1}^{m} \gamma_{ij} (\log \gamma_{ij} - 1).
```

In FWI, we use the Wasserstein-Sinkhorn distance to measure the difference between observed data {math}`d_{obs}(s, r, t)` and synthetic {math}`d_{cal}(\mathbf{m}; s, r, t)` for each source-receiver pair {math}`(s, r)`. The objective function based on the Wasserstein-Sinkhorn distance is defined as (Engquist et al., 2016; M{\'e}tivier et al., 2016; Y. Yang et al., 2018; Chizat et a., 2020):
```{math}
    \mathcal{J}_{WS}(\mathbf{m}) = \sum_{s} \sum_{r} W_{\lambda}(d_{obs}(s, r, t), d_{cal}(\mathbf{m}; s, r, t)),
```
where {math}`W_{\lambda}(d_{obs}(s, r, t), d_{cal}(\mathbf{m}; s, r, t))` represents the Wasserstein-Sinkhorn distance between the observed and simulated data. This objective function provides a robust and flexible measurement of misfits.

#### **Advantages**:
* **Robust to distribution shifts**: Captures global differences between distributions, making it less sensitive to local misalignments.
* **Differentiable and scalable**: The Sinkhorn divergence enables efficient gradient-based optimization.

#### **Disadvantages**:
* **Computationally expensive**: Solving the optimal transport problem requires higher computational cost than L2-based methods.
* **Regularization tuning**: The Sinkhorn regularization parameter affects the balance between accuracy and smoothness, requiring careful selection.

#### **Reference**
* Engquist, B., Froese, B. D., & Yang, Y. (2016). Optimal transport for seismic full waveform inversion (No. arXiv:1602.01540). arXiv.
* M´etivier, L., Brossier, R., M´erigot, Q., Oudet, E., & Virieux, J. (2016). Measuring the misfit between seismograms using an optimal transport distance: Application to full waveform inversion. Geophysical Journal International , 205 (1), 345–377. doi: 10.1093/gji/ggw014
* Yang, Y., Engquist, B., Sun, J., & Hamfeldt, B. F. (2018). Application of optimal transport and the quadratic wasserstein metric to full-waveform inversion. Geophysics, 83 (1), R43-R62. doi: 10.1190/geo2016-0663.1
* Chizat, L., Roussillon, P., L´eger, F., Vialard, F.-X., & Peyr´e, G. (2020). Faster wasserstein distance estimation with the sinkhorn divergence. Advances in Neural Information Processing Systems, 33 , 2257–2269. doi: 10.48550/ARXIV.2006.08172

## 4. Hybrid Objective Functions

### 4.1 Weighted Envelope and Global Correlation
The weighted envelope correlation-based objective function (WEC) combines the advantages of the global correlation and the envelope objective functions (Song et al., 2023):
```{math}
    \mathcal{J}_{WEC}(\mathbf{m}) = w(i) \mathcal{J}_{GC}(\mathbf{m}) + (1 - w(i))\mathcal{J}_{Envelope}(\mathbf{m}),
```
where {math}`w(i)` denotes a weighting factor, with {math}`i` representing the iteration number of the inversion. We use the sigmoid function to define the weighting factor {math}`w(i)`:
```{math}
    w(i) = \frac{1}{1 + e^{-i - \frac{N}{2}}}, \quad \text{for } i = 1, 2, \ldots, N,
```
where {math}`N` is the number of iterations.

#### **Advantages**:
* **Dynamic balance**: Allows the contribution of the global correlation and envelope terms to be adjusted iteratively, improving convergence.
* **Combines strengths**: Leverages the robustness of the envelope for amplitude and the shape preservation of the global correlation.

#### **Disadvantages**:
* **Parameter tuning**: The choice of the weighting function and its parameters may require tuning for optimal performance.
* **Increased complexity**: The addition of a dynamic weighting factor introduces extra complexity compared to simpler objective functions.

#### **Reference**
* Song, C., Wang, Y., Richardson, A., & Liu, C. (2023). Weighted envelope correlation-based waveform inversion using automatic differentiation. IEEE Transactions on Geoscience and Remote Sensing, 61 , 1–11. doi: 10.1109/TGRS.2023.3300127