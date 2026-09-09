# <span style="font-size: 20px;">Loss Functions</span>

<span style="font-size: 14px;">Loss functions (also called cost functions or objective functions) quantify how far a model's predictions are from the true values. Training a neural network means minimizing a loss function via gradient descent. The choice of loss directly determines what the model optimizes for.</span>

---

## <span style="font-size: 16px;">Mean Squared Error (MSE)</span>

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

* <span style="font-size: 14px;">Standard loss for regression tasks</span>
* <span style="font-size: 14px;">Gradient with respect to prediction:</span> $\frac{\partial}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$
* <span style="font-size: 14px;">Equivalent to maximum likelihood estimation under Gaussian noise assumption</span>
* <span style="font-size: 14px;">Penalizes large errors quadratically, making it sensitive to outliers</span>
* <span style="font-size: 14px;">**MAE** (Mean Absolute Error) is more robust to outliers but has a non-smooth gradient at zero</span>

---

## <span style="font-size: 16px;">Binary Cross-Entropy (BCE)</span>

$$
\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]
$$

* <span style="font-size: 14px;">Standard loss for binary classification where</span> $\hat{y}_i \in (0, 1)$ <span style="font-size: 14px;">is a probability</span>
* <span style="font-size: 14px;">Equivalent to the negative log-likelihood of a Bernoulli distribution</span>
* <span style="font-size: 14px;">Gradient:</span> $\frac{\partial}{\partial \hat{y}_i} = \frac{1}{n}\left(\frac{\hat{y}_i - y_i}{\hat{y}_i(1 - \hat{y}_i)}\right)$
* <span style="font-size: 14px;">When paired with a sigmoid output, the gradient simplifies to</span> $\frac{1}{n}(\hat{y}_i - y_i)$
* <span style="font-size: 14px;">**Numerical stability**: if</span> $\hat{y} = 0$ <span style="font-size: 14px;">or</span> $\hat{y} = 1$<span style="font-size: 14px;">, then</span> $\log(0) = -\infty$<span style="font-size: 14px;">. Clip predictions to</span> $[\epsilon, 1-\epsilon]$ <span style="font-size: 14px;">where</span> $\epsilon = 10^{-15}$

---

## <span style="font-size: 16px;">Categorical Cross-Entropy (CCE)</span>

<span style="font-size: 14px;">For multi-class classification with</span> $K$ <span style="font-size: 14px;">classes, given true class index</span> $c_i$ <span style="font-size: 14px;">and logits</span> $z_i \in \mathbb{R}^K$<span style="font-size: 14px;">:</span>

$$
\text{CCE} = -\frac{1}{n}\sum_{i=1}^{n}\left[z_{i,c_i} - \log\sum_{j=1}^{K} e^{z_{i,j}}\right]
$$

<span style="font-size: 14px;">This is equivalent to applying softmax then taking the negative log probability of the correct class:</span>

$$
\text{CCE} = -\frac{1}{n}\sum_{i=1}^{n}\log\left(\frac{e^{z_{i,c_i}}}{\sum_j e^{z_{i,j}}}\right)
$$

---

## <span style="font-size: 16px;">The Log-Sum-Exp Trick</span>

<span style="font-size: 14px;">Computing</span> $\log\sum_j e^{z_j}$ <span style="font-size: 14px;">directly is numerically dangerous. If any</span> $z_j$ <span style="font-size: 14px;">is large (e.g., 200), then</span> $e^{200}$ <span style="font-size: 14px;">overflows float64. The fix:</span>

$$
\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}, \quad m = \max_j z_j
$$

<span style="font-size: 14px;">Subtracting</span> $m$ <span style="font-size: 14px;">ensures the largest exponent is</span> $e^0 = 1$<span style="font-size: 14px;">, preventing overflow. The result is mathematically identical. This trick is used everywhere softmax appears: classification losses, attention mechanisms, mixture models.</span>

---

## <span style="font-size: 16px;">Hinge Loss</span>

$$
\text{Hinge} = \frac{1}{n}\sum_{i=1}^{n}\max(0,\; 1 - y_i \cdot \hat{y}_i)
$$

* <span style="font-size: 14px;">The loss function behind Support Vector Machines (SVMs)</span>
* <span style="font-size: 14px;">Labels are</span> $y_i \in \{-1, +1\}$ <span style="font-size: 14px;">and predictions are raw scores (not probabilities)</span>
* <span style="font-size: 14px;">Loss is zero when the prediction has the correct sign AND margin</span> $\geq 1$
* <span style="font-size: 14px;">Encourages a margin of separation, not just correct classification</span>
* <span style="font-size: 14px;">Not differentiable at</span> $y \cdot \hat{y} = 1$<span style="font-size: 14px;">, but subgradients work fine with SGD</span>

---

## <span style="font-size: 16px;">Choosing the Right Loss</span>

| Task | Loss | Output Activation |
|---|---|---|
| Regression | MSE (or MAE) | None (linear) |
| Binary classification | BCE | Sigmoid |
| Multi-class classification | CCE | Softmax (or fused with loss) |
| Maximum-margin classifier | Hinge | None (linear) |

<span style="font-size: 14px;">In practice, frameworks fuse softmax and cross-entropy into a single numerically stable operation. PyTorch's</span> `CrossEntropyLoss` <span style="font-size: 14px;">takes raw logits, not probabilities.</span>

---

## <span style="font-size: 16px;">Gradients and Optimization</span>

* <span style="font-size: 14px;">MSE gradient is proportional to the residual</span> $(\hat{y} - y)$<span style="font-size: 14px;">. Larger errors produce larger updates</span>
* <span style="font-size: 14px;">BCE + sigmoid gradient simplifies to</span> $(\hat{y} - y)$<span style="font-size: 14px;">, same form as MSE + linear</span>
* <span style="font-size: 14px;">CCE + softmax gradient is</span> $(p - \mathbf{1}_{\text{true class}})$<span style="font-size: 14px;">: the softmax probability minus the one-hot target</span>
* <span style="font-size: 14px;">These clean gradient forms are not coincidental: they arise because cross-entropy is the natural loss for exponential family distributions</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">Common follow-up questions in deep learning interviews:</span>


* <span style="font-size: 14px;">**Why use cross-entropy instead of MSE for classification?** MSE applied to sigmoid outputs creates a non-convex loss landscape with slow gradients when predictions are confidently wrong (sigmoid saturation). Cross-entropy's gradient does not suffer from this saturation: a confidently wrong prediction produces a large gradient, enabling fast correction</span>
* <span style="font-size: 14px;">**What is the log-sum-exp trick and why is it necessary?** When computing</span> $\log\sum e^{z_j}$<span style="font-size: 14px;">, large logits cause overflow in the exponential. Subtracting</span> $\max(z)$ <span style="font-size: 14px;">from all logits before exponentiating keeps values in a safe range. The subtracted constant cancels out mathematically, so the result is exact</span>
* <span style="font-size: 14px;">**What is the relationship between cross-entropy and KL divergence?** Cross-entropy</span> $H(p, q) = -\sum p \log q$ <span style="font-size: 14px;">equals the entropy of</span> $p$ <span style="font-size: 14px;">plus the KL divergence:</span> $H(p, q) = H(p) + D_{KL}(p \| q)$<span style="font-size: 14px;">. Since</span> $H(p)$ <span style="font-size: 14px;">is constant during training, minimizing cross-entropy is equivalent to minimizing KL divergence between the true and predicted distributions</span>
* <span style="font-size: 14px;">**When would you use hinge loss over cross-entropy?** Hinge loss focuses on the decision boundary and ignores correctly classified points with sufficient margin. This is desirable when you care about the boundary (SVMs) rather than calibrated probabilities. Cross-entropy produces calibrated probabilities, which is important when you need confidence estimates</span>
* <span style="font-size: 14px;">**Why does PyTorch's CrossEntropyLoss take logits, not probabilities?** Fusing log-softmax with negative log-likelihood into a single operation is more numerically stable and efficient. Computing softmax first and then log reintroduces the overflow problem that log-sum-exp solves. The fused version computes the loss directly from logits using the log-sum-exp trick</span>

---