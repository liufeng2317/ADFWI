## Notes
A note of clarification is in order:

1. Elastic FWI inversion is affected by multiple parameters ($v_p$,$v_s$ and $\rho$), and different inversion strategies need to be combined to obtain better inversion results.
   1. gradient precondition
   2. multi-scale strategy

2. The **Adam** class optimization algorithm is used in this test, which has been shown to be effective and efficient in previous Acoustic studies, but is still in the exploratory stage in Elastic FWI.

3. The gradients of $v_p$ and $v_s$ are compared: the gradient of $v_p$ is clearer and the gradient of vs is relatively messy, so gradient smoothing and regularization strategies are used in this test.