# Optimization Methods

## 1. First-Order Optimization Methods

### 1.1 Gradient Descent (GD)

#### (1) Introduction

Gradient Descent (GD) is one of the most fundamental optimization algorithms used to minimize a differentiable objective function. The algorithm iteratively updates the model parameters in the direction of the negative gradient of the function. By moving in the steepest descent direction, GD seeks to converge to the minimum of the objective function. This approach is commonly employed in tasks such as linear regression, logistic regression, and deep learning. Depending on the method of computing the gradients, GD can be classified into three types: Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-batch Gradient Descent.

1. **Gradient Calculation:**
   ```{math}
   g_t = \nabla_{\mathbf{m}} \mathcal{J}(\mathbf{m}_{t-1})
   ```
   Here, {math}`g_t` represents the gradient of the objective function {math}`\mathcal{J}` with respect to the model parameters {math}`\mathbf{m}_{t-1}` at time step {math}`t-1`.

2. **Parameter Update Rule:**
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - \eta g_t
   ```
   The model parameters {math}`\mathbf{m}_t` are updated by subtracting the gradient {math}`g_t` scaled by the learning rate {math}`\eta`, which controls the step size in the optimization process.

#### (2) Pros and Cons

**Advantages:**
- **Simplicity:** GD is straightforward to implement and understand. It is based on well-established principles from calculus and linear algebra.
- **Theoretical Guarantees:** In convex optimization problems, GD guarantees convergence to the global minimum, provided the learning rate is appropriately chosen.
- **Wide Applicability:** GD is broadly applicable across machine learning models, including linear regression, logistic regression, and neural networks.

**Disadvantages:**
- **Sensitive to Learning Rate:** If the learning rate {math}`\eta` is too large, the algorithm may overshoot the minimum, while a very small {math}`\eta` could lead to slow convergence or getting stuck in local minima.
- **Convergence Speed:** For large-scale datasets, GD can be slow because it computes the gradient using the entire dataset for each update (Batch Gradient Descent), which is computationally expensive.
- **Local Minima and Saddle Points:** In non-convex problems, such as neural networks, GD can converge to local minima or saddle points depending on the initial parameter values, which may not correspond to the global minimum.

----

### 1.2 Stochastic Gradient Descent (SGD)

#### (1) Introduction

Stochastic Gradient Descent (SGD) is a variant of Gradient Descent where the model parameters are updated using the gradient computed from a single data point (or a small batch) at each iteration, rather than using the entire dataset. This allows the algorithm to perform updates more frequently, making it computationally more efficient for large datasets. Although the updates are noisy due to the use of single data points, SGD often converges faster than Batch Gradient Descent, especially in large-scale problems.

In the context of FWI, each data corresponds to a source (shot), and the gradient is computed based on the mismatch between the observed and predicted wavefields at each receiver location.

1. **Gradient Calculation:**
   ```{math}
   g_t = \nabla_{\mathbf{m}} \mathcal{J}(\mathbf{m}_{t-1}; \text{s}_i)
   ```
   Here, {math}`g_t` represents the gradient of the objective function {math}`\mathcal{J}` with respect to the model parameters {math}`\mathbf{m}_{t-1}`, computed from a single shot {math}`(\text{s}_i)` at iteration {math}`t`. The data here refers to the wavefield measurements at the receiver positions corresponding to each shot.

2. **Parameter Update Rule:**
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - \eta g_t
   ```
   The model parameters {math}`\mathbf{m}_t` (typically a velocity model or other subsurface properties) are updated by subtracting the gradient {math}`g_t` scaled by the learning rate {math}`\eta`, which controls the step size in the direction of the negative gradient.

#### (2) Pros and Cons

**Advantages:**
- **Faster Updates:** Since the gradient is calculated from a single shot-receiver pair, SGD allows for faster updates, making it suitable for large-scale FWI tasks.
- **Memory Efficiency:** SGD processes one shot-receiver pair at a time, which is more memory-efficient compared to Batch Gradient Descent, especially when dealing with large datasets.
- **Potential for Better Generalization:** The noisy updates from SGD can help avoid local minima or saddle points, potentially leading to better generalization and more robust models.

**Disadvantages:**
- **Noisy Updates:** The updates are noisy due to the use of a single shot-receiver pair, which can cause fluctuations in the convergence path and make the training process less stable.
- **Convergence to Local Minima:** Similar to Batch Gradient Descent, SGD can still converge to a local minimum or a saddle point, especially in the highly non-linear objective function of FWI.
- **Tuning Learning Rate:** As with Batch Gradient Descent, the learning rate {math}`\eta` needs to be carefully tuned. A too-high learning rate can cause divergence, while a too-low learning rate can lead to slow convergence.

----

### 1.3 Average Stochastic Gradient Descent (ASGD)

#### (1) Introduction

Average Stochastic Gradient Descent (ASGD) is a variant of Stochastic Gradient Descent (SGD) that addresses the noisy updates by averaging the parameters over time. In standard SGD, the model parameters are updated with each gradient step based on a single data samples (or a small batch), leading to noisy and potentially unstable updates. ASGD improves upon this by maintaining an average of the parameters throughout the optimization process, which helps reduce the variance in the updates and leads to more stable convergence.

In Full Waveform Inversion (FWI), the gradients are still computed from single (or mini-batch) shot pairs, but the model parameters are averaged over multiple iterations, which helps in preventing overfitting and improving generalization.

1. **Gradient Calculation:**
   ```{math}
   g_t = \nabla_{\mathbf{m}} \mathcal{J}(\mathbf{m}_{t-1}; \text{s}_i)
   ```
   The gradient {math}`g_t` is computed based on the objective function {math}`\mathcal{J}`, which is typically the misfit between observed and predicted wavefields at a particular shot {math}`(\text{s}_i)` at iteration {math}`t`.

2. **Parameter Update Rule:**
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - \eta g_t
   ```
   The model parameters {math}`\mathbf{m}_t` are updated by subtracting the gradient {math}`g_t` scaled by the learning rate {math}`\eta`.

3. **Parameter Averaging:**
   ```{math}
   \hat{\mathbf{m}} = \frac{1}{T} \sum_{t=1}^T \mathbf{m}_t
   ```
   After {math}`T` iterations, the final parameters {math}`\hat{\mathbf{m}}` are computed as the average of the parameters across all iterations. This averaging helps smooth the updates and stabilize convergence.

#### (2) Pros and Cons

**Advantages:**
- **Reduced Variance in Updates:** By averaging the parameters over time, ASGD reduces the noisy updates typically seen in SGD, leading to smoother and more stable convergence.
- **Improved Generalization:** The averaging of parameters helps prevent overfitting, leading to better generalization on unseen data, which is particularly useful in FWI where the model may be highly sensitive to noise.
- **Faster Convergence:** The averaging process can speed up convergence by mitigating the effect of noisy gradients, especially in the early stages of optimization.

**Disadvantages:**
- **Slower Convergence in Later Stages:** While the averaging helps in early stages, it can slow down convergence towards the end of the training, as the parameters might become too smooth to make significant progress.
- **Requires More Memory:** Storing the history of parameter updates to compute the average can require more memory, especially when working with large models or datasets.
- **Tuning Averaging Window:** The parameter averaging process introduces an additional hyperparameter, which may need to be tuned to find the optimal trade-off between stability and speed.

----

## 2. Adaptive Gradient Optimization

### 2.1 Adaptive Gradient Algorithm (Adagrad)

#### (1) Introduction

The Adaptive Gradient Algorithm (Adagrad) optimizer adjusts the learning rate for each parameter by accumulating the sum of the squared gradients over time (Duchi et al., 2011). This allows the optimizer to scale the learning rate for each parameter individually, giving larger updates to parameters with smaller gradients and smaller updates to parameters with larger gradients. This makes Adagrad especially effective for sparse data and problems where different parameters have widely varying gradients.

1. **Gradient Calculation:**
   ```{math}
   g_t = \nabla_{\mathbf{m}} \mathcal{J}(\mathbf{m}_{t-1})
   ```
   where {math}`g_t` represents the gradient of the objective function {math}`\mathcal{J}` with respect to the model parameters {math}`\mathbf{m}_{t-1}` at time step {math}`t-1`.

2. **Accumulation of Squared Gradients:**
   ```{math}
   G_t = G_{t-1} + g_t^2
   ```
   This equation accumulates the sum of the squared gradients for the model parameters up to time step {math}`t`, where {math}`G_t` stores the historical gradient information.

3. **Parameter Update Rule:**
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t
   ```
   The model parameters {math}`\mathbf{m}_t` are updated by scaling the gradient {math}`g_t` with the inverse square root of the accumulated gradient sum {math}`G_t`, and normalizing it with the learning rate {math}`\eta`. The small constant {math}`\epsilon` is added for numerical stability to prevent division by zero.

#### (2) Pros and Cons

**Advantages:**
- **Per-Parameter Learning Rate:** Adagrad adjusts the learning rate for each parameter individually, making it effective for sparse gradients or data with varying magnitudes of gradients.
- **Good for Sparse Data:** Adagrad is especially useful when working with sparse data (e.g., in natural language processing or recommendation systems), where many parameters receive small gradients.
- **No Need for Learning Rate Scheduling:** Since the learning rate is adapted during the training process, it can eliminate the need for manually tuning the learning rate schedule.

**Disadvantages:**
- **Learning Rate Shrinking:** The accumulation of squared gradients results in a monotonically decreasing learning rate, which may cause the algorithm to stop updating parameters too early, particularly for parameters with smaller gradients.
- **Inefficiency in Long-Term Training:** Over time, the accumulated gradient sum {math}`G_t` can grow too large, which leads to overly small updates in the later stages of training, potentially slowing down convergence.
- **Sensitive to Hyperparameters:** Although Adagrad reduces the need for learning rate tuning, the choice of initial learning rate {math}`\eta` still requires careful consideration.

#### (3) Reference

[1] Duchi J, Hazan E, Singer Y. Adaptive subgradient methods for online learning and stochastic optimization[J]. Journal of machine learning research, 2011, 12(7).


----

### 2.2 Root Mean Square Propagation (RMSProp)

#### (1) Introduction

The Root Mean Square Propagation (RMSProp) optimizer is designed to adjust the learning rate based on the moving average of the squared gradients. This helps in adapting the learning rate to the scale of the gradients and stabilizes the optimization process, especially in the case of non-stationary objectives. RMSProp is widely used in deep learning for optimizing complex models and has been shown to outperform traditional gradient descent methods in certain scenarios (Graves, 2014).

1. **Gradient Calculation:**
   ```{math}
   g_t = \nabla_{\mathbf{m}} \mathcal{J}(\mathbf{m}_{t-1})
   ```
   where {math}`g_t` represents the gradient of the objective function {math}`\mathcal{J}` with respect to the model parameters {math}`\mathbf{m}_{t-1}` at time step {math}`t-1`.

2. **Exponential Moving Average of Squared Gradients:**
   ```{math}
   E[g_t^2] = \gamma E[g_{t-1}^2] + (1 - \gamma) g_t^2
   ```
   This equation computes the exponentially weighted moving average of the squared gradients. The decay rate {math}`\gamma` controls the influence of previous gradients. A higher {math}`\gamma` gives more weight to past gradients, while a lower {math}`\gamma` places more emphasis on recent gradients.

3. **Parameter Update Rule:**
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - \frac{\eta}{\sqrt{E[g_t^2] + \epsilon}} g_t
   ```
   The model parameters {math}`\mathbf{m}_t` are updated by adjusting the gradient {math}`g_t` scaled by the learning rate {math}`\eta` and normalized by the moving average of squared gradients. The small constant {math}`\epsilon` is added to avoid division by zero and ensure numerical stability.

#### (2) Pros and Cons

**Advantages:**
- **Adaptive Learning Rate:** RMSProp adapts the learning rate for each parameter, which can help with convergence in non-stationary problems where gradients can vary in scale.
- **Stable and Efficient:** By using the moving average of squared gradients, RMSProp avoids the issue of large gradient updates, leading to more stable training.
- **Efficient for Online Learning:** It is particularly effective in cases where the dataset is very large or when training involves streaming data, as it updates based on recent gradients without storing large amounts of past data.

**Disadvantages:**
- **Sensitive to Hyperparameters:** The choice of the decay rate {math}`\gamma` and the learning rate {math}`\eta` can significantly affect performance. Tuning these hyperparameters may require careful experimentation.
- **Not Ideal for Highly Noisy Gradients:** In some cases, RMSProp may over-smooth the updates, especially if {math}`\gamma` is set too high, leading to slower convergence in noisy environments.
- **Can Get Stuck in Local Minima:** Like other gradient-based methods, RMSProp may still suffer from getting stuck in local minima, though this is a general issue with many first-order optimization methods.

#### (3) Reference

[1] Graves A, Jaitly N. Towards end-to-end speech recognition with recurrent neural networks[C]//International conference on machine learning. PMLR, 2014: 1764-1772.

****

### 2.3 Adaptive Moment Estimation (Adam)

#### (1) Introduction

The Adaptive Moment Estimation (Adam) optimizer combines the advantages of both the Adaptive Gradient Algorithm (Adagrad) and Root Mean Square Propagation (RMSProp) by maintaining both the first moment (mean) and the second moment (uncentered variance) of the gradients. This allows Adam to adaptively adjust the learning rates for each parameter, improving performance and convergence speed. Adam has become one of the most popular optimizers in deep learning due to its efficiency and good generalization performance (Kingma & Ba, 2017).

1. **First Moment Estimate (Momentum):**
   ```{math}
   v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t
   ```
   The first moment estimate {math}`v_t` is the exponential moving average of the gradients {math}`g_t`. The decay rate {math}`\beta_1` controls the weight of past gradients in this moving average. Larger {math}`\beta_1` values give more weight to historical gradients, which introduces momentum to the optimization process.

2. **Second Moment Estimate (Variance):**
   ```{math}
   s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2
   ```
   The second moment estimate {math}`s_t` is the exponential moving average of the squared gradients. The decay rate {math}`\beta_2` controls the influence of past squared gradients, helping to scale the learning rate based on the variance of the gradients.

3. **Bias-Corrected First and Second Moments:**
   ```{math}
   \hat{v}_t = \frac{v_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}
   ```
   These are bias-corrected versions of {math}`v_t` and {math}`s_t`, which help to counteract the initial bias towards zero, especially in the early stages of training.

4. **Parameter Update Rule:**
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - \eta \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon}
   ```
   The model parameters {math}`\mathbf{m}_t` are updated using the bias-corrected first and second moment estimates. The learning rate {math}`\eta` is scaled by the ratio of the first moment estimate {math}`\hat{v}_t` and the square root of the second moment estimate {math}`\hat{s}_t`, with the small constant {math}`\epsilon` added for numerical stability.

#### (2) Pros and Cons

**Advantages:**
- **Adaptive Learning Rates:** Adam computes individual adaptive learning rates for each parameter, which helps to optimize complex models and handle problems with different gradient magnitudes across parameters.
- **Momentum and Variance Adaptation:** By incorporating both the first and second moments of the gradients, Adam can converge faster and more reliably than other optimizers like SGD or Adagrad, particularly in cases where gradients are sparse or noisy.
- **Bias Correction:** The bias-corrected moments help to overcome the problem of having overly small gradient estimates in the early stages of training, improving the stability and performance of the optimizer.

**Disadvantages:**
- **Sensitive to Hyperparameters:** The decay rates {math}`\beta_1` and {math}`\beta_2` and the learning rate {math}`\eta` require tuning, and improper values can degrade performance.
- **Memory Overhead:** Adam maintains both first and second moment estimates for each parameter, which increases memory consumption compared to simpler optimizers like SGD.
- **Potential Overfitting:** In some cases, the adaptive learning rates might lead to overfitting, especially with very large models or small datasets, as Adam may fit too closely to the training data.

#### (3) Reference

[1] Kingma D P. Adam: A method for stochastic optimization[J]. arXiv preprint arXiv:1412.6980, 2014.

----

### 2.4 Adam with Weight Decay (AdamW)

#### (1) Introduction

Adam with Weight Decay (AdamW) is a variant of the Adam optimizer that incorporates weight decay regularization directly into the parameter update rule. This modification helps to prevent overfitting by penalizing large weights, which is especially useful when training large models on complex datasets. The weight decay term {math}`\lambda` decays the model parameters in each update, and this decouples the weight decay from the gradient-based optimization process, leading to better performance in many scenarios (Loshchilov & Hutter, 2019).

1. **First Moment Estimate (Momentum):**
   ```{math}
   v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t
   ```
   The first moment estimate {math}`v_t` is the exponential moving average of the gradients {math}`g_t`. The decay rate {math}`\beta_1` determines the influence of past gradients on the current gradient estimate.

2. **Second Moment Estimate (Variance):**
   ```{math}
   s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2
   ```
   The second moment estimate {math}`s_t` is the exponential moving average of the squared gradients, with the decay rate {math}`\beta_2` controlling the weighting of past squared gradients.

3. **Bias-Corrected First and Second Moments:**
   ```{math}
   \hat{v}_t = \frac{v_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}
   ```
   These are the bias-corrected versions of {math}`v_t` and {math}`s_t`, compensating for their initial values being too small, especially in the early stages of training.

4. **Parameter Update Rule with Weight Decay:**
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - \eta \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon} + \lambda \eta \mathbf{m}_{t-1}
   ```
   The model parameters {math}`\mathbf{m}_t` are updated by scaling the bias-corrected first and second moment estimates, and the weight decay term {math}`\lambda \eta \mathbf{m}_{t-1}` adds a regularization penalty to prevent overfitting. Here, {math}`\lambda` is the weight decay coefficient, and {math}`\epsilon` ensures numerical stability.

#### (2) Pros and Cons

**Advantages:**
- **Improved Regularization:** AdamW decouples weight decay from the gradient-based updates, allowing for better control of regularization. This leads to more effective regularization compared to traditional methods that apply weight decay to the gradients.
- **Effective for Large Models:** The incorporation of weight decay into AdamW helps to prevent overfitting, making it especially useful for training large models on high-dimensional data.
- **Better Convergence:** By decoupling the weight decay from the optimization process, AdamW can improve convergence rates and avoid the issues associated with traditional weight decay methods.

**Disadvantages:**
- **Sensitive to Hyperparameters:** As with Adam, the hyperparameters {math}`\beta_1`, {math}`\beta_2`, and {math}`\eta` need to be carefully tuned to achieve optimal performance. Additionally, the weight decay coefficient {math}`\lambda` must also be selected appropriately.
- **Increased Memory Usage:** Like Adam, AdamW requires storing both first and second moment estimates for each parameter, which increases memory usage compared to simpler optimizers like SGD.
- **Potential Over-regularization:** If {math}`\lambda` is set too high, it may lead to underfitting, where the model is overly regularized and unable to learn the underlying patterns in the data.

#### (3) Reference

[1] Loshchilov I. Decoupled weight decay regularization[J]. arXiv preprint arXiv:1711.05101, 2017.

----

### 2.5 Nesterov-accelerated Adam (NAdam)

#### (1) Introduction

Nesterov-accelerated Adam (NAdam) combines the benefits of the Adam optimizer with Nesterov's accelerated gradient (NAG) momentum, which helps improve the convergence speed by anticipating the future gradient direction. This is particularly useful when dealing with complex, high-dimensional optimization problems. NAdam computes the gradient with respect to the "lookahead" gradient, enabling faster and more accurate convergence than Adam, especially in situations where the learning rate needs to be adjusted dynamically (Dozat, 2016).

1. **First Moment Estimate (Momentum):**
   ```{math}
   v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t
   ```
   The first moment estimate {math}`v_t` is the exponential moving average of the gradients {math}`g_t`. The decay rate {math}`\beta_1` determines how much weight is given to previous gradients in the momentum update.

2. **Second Moment Estimate (Variance):**
   ```{math}
   s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2
   ```
   The second moment estimate {math}`s_t` is the exponential moving average of the squared gradients, with the decay rate {math}`\beta_2` controlling the influence of past squared gradients.

3. **Bias-Corrected First and Second Moments:**
   ```{math}
   \hat{v}_t = \frac{v_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}
   ```
   These are the bias-corrected versions of {math}`v_t` and {math}`s_t`, compensating for their initial small values, particularly at the start of training.

4. **Nesterov-accelerated Parameter Update Rule:**
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - \frac{\eta}{\sqrt{\hat{s}_t} + \epsilon} \left( \beta_1 \hat{v}_t + \frac{1 - \beta_1}{1 - \beta_1^t} g_t \right)
   ```
   The model parameters {math}`\mathbf{m}_t` are updated using the combination of the bias-corrected momentum estimate {math}`\hat{v}_t` and the adjusted gradient {math}`g_t`, which is scaled by the lookahead term {math}`\frac{1 - \beta_1}{1 - \beta_1^t}`. This update rule applies the Nesterov momentum by taking a "lookahead" at the future gradient direction before updating the parameters.

#### (2) Pros and Cons

**Advantages:**
- **Improved Convergence:** The Nesterov momentum provides an "anticipation" of the future gradients, which helps the optimizer to converge faster than regular Adam, especially when training on large and complex datasets.
- **Momentum with a Lookahead:** By adding the Nesterov term, NAdam incorporates a more informed gradient update, which can improve the stability and efficiency of the optimization process.
- **Combines Strengths of Momentum and Adam:** NAdam effectively combines the adaptive learning rates of Adam with the faster convergence properties of Nesterov's momentum, making it a powerful optimization choice for deep learning tasks.

**Disadvantages:**
- **Computational Overhead:** The additional lookahead gradient computation introduces extra overhead compared to standard Adam, which may increase the training time, particularly for very large models.
- **Sensitive to Hyperparameters:** The effectiveness of NAdam is still dependent on proper tuning of the hyperparameters, including the learning rate {math}`\eta`, decay rates {math}`\beta_1` and {math}`\beta_2`, and the small constant {math}`\epsilon`.
- **May Not Always Improve Over Adam:** In some cases, particularly when the problem is not sensitive to momentum or the learning rate is well-tuned, NAdam may not provide significant improvements over Adam.

#### (3) Reference

[1] Dozat T. Incorporating nesterov momentum into adam[J]. 2016.

----

### 2.6 Rectified Adam (RAdam)

#### (1) Introduction

Rectified Adam (RAdam) introduces a rectification mechanism to the Adam optimizer in order to stabilize the variance of adaptive learning rates, particularly in the early stages of optimization. RAdam adjusts the learning rates during the initial steps, preventing the instability commonly observed with adaptive optimizers when training deep neural networks. By adding a correction term, it helps avoid the "early stage instability" that can arise when the first and second moment estimates are biased at the beginning of training (L. Liu et al., 2019).

1. **First Moment Estimate (Momentum):**
   ```{math}
   v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t
   ```
   The first moment estimate {math}`v_t` is the exponential moving average of the gradients {math}`g_t`, where {math}`\beta_1` controls the influence of past gradients on the momentum update.

2. **Second Moment Estimate (Variance):**
   ```{math}
   s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2
   ```
   The second moment estimate {math}`s_t` is the exponential moving average of the squared gradients, with {math}`\beta_2` determining the influence of past squared gradients.

3. **Bias-Corrected First and Second Moments:**
   ```{math}
   \hat{v}_t = \frac{v_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}
   ```
   The bias-corrected first and second moment estimates, {math}`\hat{v}_t` and {math}`\hat{s}_t`, compensate for their initial bias during the early training stages.

4. **Rectification Mechanism (for Early Stages):**
   ```{math}
   \rho_\infty = \frac{2}{1 - \beta_2} - 1, \quad \rho_t = \rho_\infty - \frac{2t\beta_2^t}{1 - \beta_2^t}
   ```
   The parameter {math}`\rho_\infty` represents the asymptotic value of the variance ratio, and {math}`\rho_t` adjusts for the changes in {math}`\rho_\infty` during the optimization process. This correction is introduced to stabilize the learning rates in the initial steps.

5. **Rectification Factor:**
   ```{math}
   r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_\infty}{(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}}
   ```
   The rectification factor {math}`r_t` helps stabilize the variance of the adaptive learning rates by adjusting for the bias in the second moment estimate, especially in the early stages of training.

6. **Parameter Update Rule:**
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - \eta \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon} r_t
   ```
   The model parameters {math}`\mathbf{m}_t` are updated using the bias-corrected first moment estimate {math}`\hat{v}_t`, the bias-corrected second moment estimate {math}`\hat{s}_t`, and the rectification factor {math}`r_t`, which improves stability in the early stages of training.

#### (2) Pros and Cons

**Advantages:**
- **Stabilized Early Training:** RAdam mitigates the instability issues during the early phases of training by using a rectification mechanism, ensuring smoother convergence in the initial iterations.
- **Improved Performance in Early Stages:** The adjustment of the learning rate during the early stages of training allows RAdam to benefit from faster convergence, especially when compared to Adam, which may experience instability when the moment estimates are biased.
- **Adaptive Learning Rates with Stability:** RAdam combines the benefits of adaptive learning rates, as in Adam, with an additional mechanism that stabilizes them in the beginning stages of training.

**Disadvantages:**
- **Complexity in Calculation:** The rectification factor adds additional complexity to the calculations compared to standard Adam, potentially introducing slight overhead in terms of computation.
- **Hyperparameter Sensitivity:** Like other adaptive optimizers, RAdam still requires careful tuning of hyperparameters such as the learning rate {math}`\eta`, the decay rates {math}`\beta_1` and {math}`\beta_2`, and the small constant {math}`\epsilon`.
- **Limited Improvement in Some Cases:** In scenarios where the optimizer already converges smoothly, such as in well-conditioned problems, the benefits of RAdam over Adam may be marginal.

#### (3) Reference

[1] Liu L, Jiang H, He P, et al. On the variance of the adaptive learning rate and beyond[J]. arXiv preprint arXiv:1908.03265, 2019.

----

## 3. Second-Order Optimization Methods

### 3.1 Quasi-Newton Methods: L-BFGS

#### (1) Introduction

The Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm is a popular quasi-Newton optimization method. It is a variant of the BFGS (Broyden–Fletcher–Goldfarb–Shanno) algorithm that uses limited memory to compute an approximation of the inverse Hessian matrix. L-BFGS is an iterative algorithm used for large-scale optimization problems where storing the full Hessian matrix is computationally expensive. By using only a small number of vectors to approximate the Hessian, L-BFGS is much more memory efficient while retaining the advantages of Newton-type methods. It is widely used in machine learning tasks where high-dimensional optimization is required, such as training neural networks or solving large linear systems.

1. **Gradient Calculation:**
   ```{math}
   g_t = \nabla_{\mathbf{m}} \mathcal{J}(\mathbf{m}_{t-1})
   ```
   where {math}`g_t` represents the gradient of the objective function {math}`\mathcal{J}` with respect to the model parameters {math}`\mathbf{m}_{t-1}` at time step {math}`t-1`.

2. **Update Rule with Quasi-Newton Approximation:**
   The L-BFGS algorithm computes an approximation to the inverse Hessian matrix using previous gradients and parameter updates:
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - \alpha_t H_t g_t
   ```
   Here, {math}`\alpha_t` is a step size determined by a line search, and {math}`H_t` is the approximation to the inverse Hessian matrix at time step {math}`t`, which is updated using the following formulas based on the gradient and parameter updates:
   ```{math}
   y_t = g_t - g_{t-1}, \quad s_t = \mathbf{m}_t - \mathbf{m}_{t-1}
   ```
   where {math}`y_t` is the difference in gradients, and {math}`s_t` is the difference in the parameters.

   The update rule for the inverse Hessian approximation in L-BFGS uses a recursive formula:
   ```{math}
   H_t = H_{t-1} + \frac{y_t y_t^T}{y_t^T s_t} - \frac{H_{t-1} s_t s_t^T H_{t-1}}{s_t^T H_{t-1} s_t}
   ```

#### (2) Pros and Cons

**Advantages:**
- **Memory Efficiency:** L-BFGS uses only a small number of vectors (instead of the full Hessian matrix), making it much more memory efficient compared to other Newton-type methods like BFGS.
- **Fast Convergence:** It typically converges faster than gradient-based methods like Gradient Descent due to the use of second-order approximations (approximating the Hessian matrix).
- **Applicability to Large-Scale Problems:** L-BFGS is well-suited for large-scale problems where computing and storing the full Hessian matrix is impractical.

**Disadvantages:**
- **Computational Overhead:** Although it is more memory-efficient, L-BFGS can still be computationally expensive for very large datasets due to the need for computing gradients and maintaining the history of parameter updates.
- **Requires Line Search:** The effectiveness of the method depends on performing a line search to determine the optimal step size {math}`\alpha_t`. The line search can be costly, especially for high-dimensional problems.
- **Sensitive to Hyperparameters:** Like many optimization methods, L-BFGS can be sensitive to the choice of hyperparameters, such as the initial step size and the number of stored vectors for approximating the Hessian.

****

### 3.2 Newton's Method

#### (1) Introduction

Newton's Method is an optimization algorithm that uses second-order derivatives to find the local minimum (or maximum) of a function. Unlike gradient descent methods, which use first-order derivatives (gradients), Newton's method utilizes the Hessian matrix, which contains second-order derivatives. This allows for faster convergence, especially when the function is well-behaved (i.e., the objective function is convex and the Hessian is positive definite).

In the context of optimization, the update rule in Newton's Method is:

1. **Gradient Calculation:**
   ```{math}
   g_t = \nabla_{\mathbf{m}} \mathcal{J}(\mathbf{m}_{t-1})
   ```
   where {math}` g_t ` is the gradient of the objective function {math}`\mathcal{J}` at time step {math}` t-1 `, with respect to the model parameters {math}`\mathbf{m}_{t-1}`.

2. **Hessian Matrix Calculation:**
   ```{math}
   H_t = \nabla^2_{\mathbf{m}} \mathcal{J}(\mathbf{m}_{t-1})
   ```
   where {math}` H_t ` is the Hessian matrix, which represents the second-order derivatives of the objective function {math}`\mathcal{J}` with respect to the model parameters {math}`\mathbf{m}_{t-1}`.

3. **Update Rule:**
   ```{math}
   \mathbf{m}_t = \mathbf{m}_{t-1} - H_t^{-1} g_t
   ```
   The model parameters {math}`\mathbf{m}_t` are updated by subtracting the product of the inverse of the Hessian matrix and the gradient {math}` g_t `.

#### (2) Computational Inefficiency

Although Newton's method can converge faster compared to first-order methods like gradient descent due to its use of second-order information, it suffers from significant computational inefficiency, especially when dealing with large-scale problems. This inefficiency arises from the following factors:

- **Hessian Calculation:** Computing the full Hessian matrix {math}` H_t ` is computationally expensive, as it requires calculating the second-order partial derivatives of the objective function. For high-dimensional problems, this becomes a bottleneck.
- **Matrix Inversion:** Inverting the Hessian matrix {math}` H_t ` at each iteration requires {math}` O(d^3) ` operations, where {math}` d ` is the number of model parameters. This can be prohibitively slow for large-scale problems, particularly in the context of Full Waveform Inversion (FWI) where the number of model parameters can be very large.

#### (3) ADFWI Framework Consideration

Due to the high computational cost associated with both Hessian matrix computation and inversion, **Newton's method is not implemented in the ADFWI framework**. While the method theoretically provides faster convergence, the computational overhead makes it unsuitable for real-time or large-scale inversion problems in the ADFWI framework. Instead, more efficient optimization methods, such as L-BFGS or gradient-based methods with adaptive learning rates (like Adam), are preferred to achieve a balance between computational efficiency and convergence speed.

#### (4) Pros and Cons

**Advantages:**
- **Faster Convergence:** When the objective function is smooth and the Hessian is positive definite, Newton's method can converge in fewer iterations compared to first-order methods like gradient descent.
- **Second-order Information:** By using second-order derivatives, Newton's method can account for the curvature of the objective function, making it more robust to ill-conditioned optimization problems.

**Disadvantages:**
- **High Computational Cost:** The most significant disadvantage of Newton's method is the cost of calculating and inverting the Hessian matrix, which makes it computationally expensive for high-dimensional problems.
- **Memory Usage:** Storing the Hessian matrix requires significant memory, making it impractical for large-scale problems where the number of parameters is very high.
- **Complexity in Implementation:** Implementing Newton's method requires computing second-order derivatives, which can be difficult and computationally expensive in practice, especially in complex models such as those used in FWI.

