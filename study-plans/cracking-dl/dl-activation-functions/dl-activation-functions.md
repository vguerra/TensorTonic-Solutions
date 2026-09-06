# <span style="font-size: 20px;">Activation Functions</span>

<span style="font-size: 14px;">Activation functions introduce non-linearity into neural networks. Without them, stacking linear layers would collapse into a single linear transformation regardless of depth. Every activation has a corresponding derivative used during backpropagation to compute gradients.</span>

---

## <span style="font-size: 16px;">Why Non-Linearity Matters</span>

<span style="font-size: 14px;">A neural network layer computes</span> $z = Wx + b$<span style="font-size: 14px;">. Without an activation, two layers give</span> $W_2(W_1 x + b_1) + b_2 = W'x + b'$<span style="font-size: 14px;"> - still a single linear transformation. The activation function</span> $h = f(z)$ <span style="font-size: 14px;">breaks this linearity, enabling networks to approximate any continuous function (Universal Approximation Theorem).</span>

---

## <span style="font-size: 16px;">ReLU and Variants</span>

<span style="font-size: 14px;">**ReLU** (Rectified Linear Unit) is the default activation in most networks:</span>

$$
f(x) = \max(0, x), \quad f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

* <span style="font-size: 14px;">Computationally cheap (just a comparison)</span>
* <span style="font-size: 14px;">Does not saturate for positive inputs, avoiding vanishing gradients</span>
* <span style="font-size: 14px;">**Dying ReLU problem**: neurons with consistently negative inputs have zero gradient and stop learning entirely</span>

<span style="font-size: 14px;">**LeakyReLU** addresses dying neurons by allowing a small gradient when</span> $x \leq 0$<span style="font-size: 14px;">:</span>

$$
f(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}, \quad f'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}
$$

<span style="font-size: 14px;">where</span> $\alpha = 0.01$ <span style="font-size: 14px;">is standard. Every neuron always has a non-zero gradient, preventing complete death.</span>

---

## <span style="font-size: 16px;">Sigmoid and Tanh</span>

<span style="font-size: 14px;">**Sigmoid** squashes inputs to</span> $(0, 1)$<span style="font-size: 14px;">:</span>

$$
\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

* <span style="font-size: 14px;">Historically important but rarely used in hidden layers today</span>
* <span style="font-size: 14px;">Maximum derivative is 0.25 (at</span> $x = 0$<span style="font-size: 14px;">), causing vanishing gradients in deep networks</span>
* <span style="font-size: 14px;">Outputs are always positive, leading to zig-zagging gradient updates</span>
* <span style="font-size: 14px;">Still used in output layers for binary classification and in gating mechanisms (LSTM, GRU)</span>

<span style="font-size: 14px;">**Tanh** squashes inputs to</span> $(-1, 1)$<span style="font-size: 14px;">:</span>

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \quad \tanh'(x) = 1 - \tanh^2(x)
$$

* <span style="font-size: 14px;">Zero-centered outputs avoid the zig-zagging problem of sigmoid</span>
* <span style="font-size: 14px;">Maximum derivative is 1.0 (at</span> $x = 0$<span style="font-size: 14px;">), better than sigmoid's 0.25</span>
* <span style="font-size: 14px;">Still saturates at extremes, so vanishing gradients remain an issue in very deep networks</span>
* <span style="font-size: 14px;">Note:</span> $\tanh(x) = 2\sigma(2x) - 1$<span style="font-size: 14px;"> - they are scaled versions of each other</span>

---

## <span style="font-size: 16px;">GELU (Gaussian Error Linear Unit)</span>

<span style="font-size: 14px;">GELU is the default activation in modern transformers (GPT, BERT, LLaMA). It smoothly gates the input by its percentile under a Gaussian distribution:</span>

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

<span style="font-size: 14px;">where</span> $\Phi(x)$ <span style="font-size: 14px;">is the standard normal CDF. In practice, the tanh approximation is used:</span>

$$
\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)
$$

<span style="font-size: 14px;">For the derivative, let</span> $u = \sqrt{2/\pi} \cdot (x + 0.044715 x^3)$ <span style="font-size: 14px;">and</span> $t = \tanh(u)$<span style="font-size: 14px;">:</span>

$$
\text{GELU}'(x) = 0.5(1 + t) + 0.5 \cdot x \cdot (1 - t^2) \cdot \sqrt{\frac{2}{\pi}} \cdot (1 + 3 \cdot 0.044715 \cdot x^2)
$$

* <span style="font-size: 14px;">Unlike ReLU, GELU is smooth everywhere (differentiable at</span> $x = 0$<span style="font-size: 14px;">)</span>
* <span style="font-size: 14px;">For large positive</span> $x$<span style="font-size: 14px;">, GELU approaches the identity; for large negative</span> $x$<span style="font-size: 14px;">, it approaches zero</span>
* <span style="font-size: 14px;">The derivative can be slightly negative near</span> $x \approx -1$<span style="font-size: 14px;">, unlike ReLU which is always non-negative</span>

---

## <span style="font-size: 16px;">Swish</span>

<span style="font-size: 14px;">Swish was discovered through automated activation search by Google Brain:</span>

$$
\text{Swish}(x) = x \cdot \sigma(x), \quad \text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))
$$

* <span style="font-size: 14px;">Non-monotonic: has a small dip below zero near</span> $x \approx -1.28$
* <span style="font-size: 14px;">Smooth and self-gated (the sigmoid acts as a learnable gate on the linear input)</span>
* <span style="font-size: 14px;">Bounded below (approximately -0.28) but unbounded above</span>
* <span style="font-size: 14px;">Swish with a learnable parameter</span> $\beta$ <span style="font-size: 14px;">(SiLU) is used in architectures like EfficientNet and LLaMA</span>

---

## <span style="font-size: 16px;">The Vanishing Gradient Problem</span>

<span style="font-size: 14px;">During backpropagation, gradients are multiplied by activation derivatives at each layer. If the derivative is consistently less than 1 (as with sigmoid or tanh in saturated regions), gradients shrink exponentially with depth:</span>

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h_n} \cdot \prod_{i=1}^{n} f'(z_i) \cdot W_i
$$

<span style="font-size: 14px;">With sigmoid, the maximum derivative is 0.25, so after 10 layers the gradient could shrink by a factor of</span> $0.25^{10} \approx 10^{-6}$<span style="font-size: 14px;">. ReLU-family activations have derivative 1 for positive inputs, keeping gradients alive in deep networks.</span>

---

## <span style="font-size: 16px;">Choosing Activations in Practice</span>

* <span style="font-size: 14px;">**Hidden layers (general)**: ReLU is the default starting point. Use LeakyReLU if dying neurons are observed</span>
* <span style="font-size: 14px;">**Transformers**: GELU is the standard (GPT, BERT). SwiGLU (Swish-gated variant) is used in LLaMA</span>
* <span style="font-size: 14px;">**Output layer (binary classification)**: Sigmoid to produce probabilities</span>
* <span style="font-size: 14px;">**Output layer (multi-class)**: Softmax (not covered here as it operates on vectors)</span>
* <span style="font-size: 14px;">**Gating mechanisms**: Sigmoid (LSTM gates, attention gates)</span>
* <span style="font-size: 14px;">**Residual networks**: ReLU or GELU; the skip connections help mitigate vanishing gradients regardless</span>

---

## <span style="font-size: 16px;">Derivative Summary</span>

| Activation | Derivative | Key Property |
|---|---|---|
| ReLU | $1$ or $0$ | Sparse, constant gradient |
| LeakyReLU | $1$ or $\alpha$ | Never zero |
| Sigmoid | $\sigma(1-\sigma)$ | Max 0.25, self-referencing |
| Tanh | $1 - \tanh^2$ | Max 1.0, self-referencing |
| GELU | (see formula above) | Can be slightly negative |
| Swish | $\sigma + x\sigma(1-\sigma)$ | Non-monotonic |

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">Common follow-up questions in deep learning interviews:</span>


* <span style="font-size: 14px;">**Why not just use sigmoid everywhere?** Sigmoid saturates at both extremes, giving near-zero gradients. In deep networks, this causes vanishing gradients where early layers barely learn. Its outputs are also not zero-centered, causing inefficient gradient updates. ReLU avoids saturation for positive inputs and is computationally simpler</span>
* <span style="font-size: 14px;">**What is the dying ReLU problem and how do you fix it?** When a ReLU neuron's pre-activation is always negative (due to a large negative bias or unlucky initialization), its output is always zero and its gradient is always zero, so it never updates. Fixes include LeakyReLU, PReLU (learnable slope), or careful initialization (He initialization)</span>
* <span style="font-size: 14px;">**Why is GELU used in transformers instead of ReLU?** GELU provides smooth, probabilistic gating that empirically improves transformer training. Unlike ReLU's hard cutoff at zero, GELU smoothly transitions, which may help with optimization. It also allows small negative values to pass through slightly, preserving more information</span>
* <span style="font-size: 14px;">**Why do sigmoid and tanh have self-referencing derivatives?** Both are solutions to specific differential equations. For sigmoid:</span> $\sigma' = \sigma(1-\sigma)$<span style="font-size: 14px;">. For tanh:</span> $\tanh' = 1 - \tanh^2$<span style="font-size: 14px;">. This is computationally convenient because you compute the forward pass first and reuse that value for the backward pass, avoiding redundant exponential calculations</span>
* <span style="font-size: 14px;">**What is the relationship between sigmoid and tanh?**</span> $\tanh(x) = 2\sigma(2x) - 1$<span style="font-size: 14px;">. Tanh is a rescaled, zero-centered version of sigmoid. This means tanh derivatives can also be expressed in terms of sigmoid. Historically, tanh was preferred over sigmoid in hidden layers because its zero-centered output leads to more balanced gradient updates</span>

---