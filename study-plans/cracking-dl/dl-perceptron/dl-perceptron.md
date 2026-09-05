# <span style="font-size: 20px;">Perceptron</span>

<span style="font-size: 14px;">The perceptron, introduced by Frank Rosenblatt in 1958, is the simplest neural network: a single neuron that learns a linear decision boundary. Despite its simplicity, it introduces the core concepts that scale to modern deep learning: weighted sums, activation functions, and iterative weight updates driven by prediction errors.</span>

---

## <span style="font-size: 16px;">Mathematical Model</span>

<span style="font-size: 14px;">A perceptron takes an input vector</span> $x \in \mathbb{R}^d$<span style="font-size: 14px;">, computes a weighted sum, and applies a step activation:</span>

$$
z = w \cdot x + b = \sum_{j=1}^{d} w_j x_j + b
$$

$$
\hat{y} = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}
$$

<span style="font-size: 14px;">The weight vector</span> $w \in \mathbb{R}^d$ <span style="font-size: 14px;">and bias</span> $b \in \mathbb{R}$ <span style="font-size: 14px;">are the learnable parameters. The bias shifts the decision boundary away from the origin.</span>

---

## <span style="font-size: 16px;">The Step Function</span>

* <span style="font-size: 14px;">Outputs exactly 0 or 1, making the perceptron a binary classifier</span>
* <span style="font-size: 14px;">Not differentiable at</span> $z = 0$<span style="font-size: 14px;">, and has zero gradient everywhere else</span>
* <span style="font-size: 14px;">This is why the perceptron uses a specialized learning rule instead of gradient descent</span>
* <span style="font-size: 14px;">Modern networks replace the step function with smooth activations (sigmoid, ReLU) that enable gradient-based optimization</span>

---

## <span style="font-size: 16px;">Perceptron Learning Rule</span>

<span style="font-size: 14px;">For each training sample</span> $(x_i, y_i)$<span style="font-size: 14px;">:</span>

$$
w \leftarrow w + \eta \cdot (y_i - \hat{y}_i) \cdot x_i
$$

$$
b \leftarrow b + \eta \cdot (y_i - \hat{y}_i)
$$

<span style="font-size: 14px;">where</span> $\eta$ <span style="font-size: 14px;">is the learning rate. The update has three cases:</span>

* <span style="font-size: 14px;">**Correct prediction** (</span>$y_i = \hat{y}_i$<span style="font-size: 14px;">): error is 0, no update</span>
* <span style="font-size: 14px;">**False negative** (</span>$y_i = 1, \hat{y}_i = 0$<span style="font-size: 14px;">): error is +1, weights move toward</span> $x_i$
* <span style="font-size: 14px;">**False positive** (</span>$y_i = 0, \hat{y}_i = 1$<span style="font-size: 14px;">): error is -1, weights move away from</span> $x_i$

<span style="font-size: 14px;">This is an online (stochastic) update applied one sample at a time, making the perceptron an early form of stochastic gradient descent.</span>

---

## <span style="font-size: 16px;">Convergence Theorem</span>

<span style="font-size: 14px;">If the training data is linearly separable, the perceptron learning rule is guaranteed to converge to a solution in a finite number of updates. The number of updates is bounded by:</span>

$$
\text{updates} \leq \frac{R^2 \|w^*\|^2}{\gamma^2}
$$

<span style="font-size: 14px;">where</span> $R = \max_i \|x_i\|$ <span style="font-size: 14px;">is the maximum input norm,</span> $w^*$ <span style="font-size: 14px;">is the optimal weight vector, and</span> $\gamma$ <span style="font-size: 14px;">is the margin. Key implications:</span>

* <span style="font-size: 14px;">Larger margins lead to faster convergence</span>
* <span style="font-size: 14px;">The algorithm finds some separating hyperplane, but not necessarily the maximum-margin one (unlike SVMs)</span>
* <span style="font-size: 14px;">If data is not linearly separable, the algorithm will never converge</span>

---

## <span style="font-size: 16px;">Geometric Interpretation</span>

* <span style="font-size: 14px;">The weight vector</span> $w$ <span style="font-size: 14px;">defines the normal to the decision hyperplane</span>
* <span style="font-size: 14px;">The bias</span> $b$ <span style="font-size: 14px;">controls the offset of the hyperplane from the origin</span>
* <span style="font-size: 14px;">The decision boundary is the set of points where</span> $w \cdot x + b = 0$
* <span style="font-size: 14px;">Points on one side (</span>$w \cdot x + b > 0$<span style="font-size: 14px;">) are classified as 1, the other side as 0</span>
* <span style="font-size: 14px;">Each weight update rotates and translates the hyperplane to correctly classify the misclassified point</span>

---

## <span style="font-size: 16px;">The XOR Limitation</span>

<span style="font-size: 14px;">Minsky and Papert (1969) proved that a single perceptron cannot learn the XOR function because XOR is not linearly separable. The four XOR points:</span>

* $(0,0) \to 0$<span style="font-size: 14px;">,</span> $(0,1) \to 1$<span style="font-size: 14px;">,</span> $(1,0) \to 1$<span style="font-size: 14px;">,</span> $(1,1) \to 0$

<span style="font-size: 14px;">No single line can separate the 0s from the 1s. This limitation motivated multi-layer networks (MLPs) which learn non-linear boundaries by composing multiple perceptrons.</span>

---

## <span style="font-size: 16px;">From Perceptron to Modern Networks</span>

* <span style="font-size: 14px;">Replace step function with smooth activations (sigmoid, ReLU) to enable gradient-based learning</span>
* <span style="font-size: 14px;">Stack multiple neurons into layers to learn non-linear functions</span>
* <span style="font-size: 14px;">Replace the perceptron learning rule with backpropagation</span>
* <span style="font-size: 14px;">The core pattern remains: linear transformation followed by non-linear activation</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">Common follow-up questions in deep learning interviews:</span>


* <span style="font-size: 14px;">**Why can't a perceptron learn XOR?** XOR requires a non-linear decision boundary. A single perceptron can only represent a linear boundary (a hyperplane). You need at least two layers: one to create intermediate representations and another to combine them. A 2-layer MLP with 2 hidden neurons can solve XOR</span>
* <span style="font-size: 14px;">**What happens if the data is not linearly separable?** The algorithm oscillates indefinitely without converging. The weights keep updating without finding a stable solution. Variants like the pocket algorithm (keeps the best weights seen so far) or the averaged perceptron (averages weights across all updates) address this</span>
* <span style="font-size: 14px;">**Perceptron vs logistic regression?** Both are linear classifiers, but logistic regression uses a smooth sigmoid activation and outputs probabilities, enabling gradient descent optimization and calibrated confidence. The perceptron outputs hard 0/1 predictions and uses a specialized update rule. Logistic regression handles non-separable data more gracefully</span>
* <span style="font-size: 14px;">**Does the learning rate matter for convergence?** For linearly separable data, the perceptron converges regardless of the learning rate value. The learning rate scales the update magnitude but does not change the direction of correction. In practice, larger learning rates cause bigger jumps and can lead to oscillation around the boundary before settling</span>
* <span style="font-size: 14px;">**How does the perceptron relate to SVMs?** Both find linear decision boundaries, but SVMs find the maximum-margin boundary (the one farthest from all training points). The perceptron finds any separating boundary with no margin guarantee. SVMs solve a constrained optimization problem; the perceptron uses iterative error-correction updates</span>

---