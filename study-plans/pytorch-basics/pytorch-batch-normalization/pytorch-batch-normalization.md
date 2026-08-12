## <span style="font-size: 20px;">The Problem: Shifting Input Distributions</span>

Training a deep neural network means training many layers simultaneously. Each layer receives activations from the layer below, transforms them, and passes the result upward. As lower layers update their weights, the distribution of activations flowing into upper layers changes. Upper layers must continuously adapt to inputs whose statistics are drifting, which slows convergence and requires careful learning rate tuning.

Consider a two-layer network:

$$
h = \sigma(W_2 \cdot \sigma(W_1 x + b_1) + b_2)
$$

When $W_1$ changes during a gradient update, the distribution of $\sigma(W_1 x + b_1)$ shifts. Layer 2 now receives inputs with different mean, variance, and possibly shape. The gradient signal computed for the old distribution is slightly stale. Multiply this effect across dozens or hundreds of layers and training becomes unstable.

Ioffe and Szegedy (2015) called this internal covariate shift and proposed batch normalization as the fix: before each layer processes its input, normalize the input so its statistics are consistent, regardless of what happened in preceding layers.

## The Batch Normalization Transform

For a mini-batch of $N$ samples, each with $D$ features, the input is a matrix $X \in \mathbb{R}^{N \times D}$. Batch normalization processes each feature independently.

## Computing Per-Feature Statistics

For each feature $j \in \{1, \ldots, D\}$, compute the mean and variance across the batch:

$$
\mu_j = \frac{1}{N} \sum_{i=1}^{N} X_{ij}, \quad \sigma_j^2 = \frac{1}{N} \sum_{i=1}^{N} (X_{ij} - \mu_j)^2
$$

- This is the population variance (dividing by $N$), not the sample variance (dividing by $N-1$)
- Matches the original paper and PyTorch's implementation
- The distinction matters for small batch sizes

## Normalization

Each feature is centered to zero mean and scaled to unit variance:

$$
\hat{X}_{ij} = \frac{X_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

- The $\epsilon$ term (typically $10^{-5}$) prevents division by zero when a feature has constant value across the batch
- Placed inside the square root, not added after, because this choice produces better-behaved gradients

## The Scale-Shift Parameters

If normalization were the whole story, batch normalization would reduce the network's expressive power. Forcing zero mean and unit variance constrains what the layer can represent. For example, a sigmoid activation applied to zero-mean, unit-variance input would always operate in its near-linear region, never reaching the saturated tails.

Learnable parameters $\gamma_j$ (scale) and $\beta_j$ (shift) restore full expressiveness:

$$
Y_{ij} = \gamma_j \hat{X}_{ij} + \beta_j
$$

- These parameters are learned through backpropagation alongside all other model weights
- If the network decides feature $j$ should have mean $\mu$ and standard deviation $\sigma$, it can learn $\gamma_j = \sigma$ and $\beta_j = \mu$
- In the special case where $\gamma_j = \sqrt{\sigma_j^2 + \epsilon}$ and $\beta_j = \mu_j$, the normalization is exactly undone, recovering original activations
- Batch normalization never reduces model capacity: it can always learn to be a no-op if that is optimal
- $\gamma$ is initialized to 1 and $\beta$ to 0, so initial behavior is pure normalization. The network gradually adjusts during training

## Why Batch Normalization Works

The original paper attributed effectiveness to reducing internal covariate shift. Later research revealed a more nuanced picture.

## Loss Landscape Smoothing

Santurkar et al. (2018) showed the primary benefit is smoothing the loss landscape. Without batch normalization, the loss surface of deep networks is highly non-convex with sharp, irregular contours. Gradient descent on such a surface is erratic: the optimal step size varies wildly, and the gradient at any point poorly predicts the gradient nearby.

Batch normalization makes the loss surface smoother and more predictable. It reduces the Lipschitz constant of the loss function and its gradient:

$$
\|\nabla \mathcal{L}(\theta_1) - \nabla \mathcal{L}(\theta_2)\| \leq L \|\theta_1 - \theta_2\|
$$

- Smaller Lipschitz constant $L$ means the gradient changes less rapidly
- Larger learning rates become safe and each gradient step is more reliable
- This explains the two most visible effects: higher learning rates and smoother training curves

## Gradient Flow Improvement

In deep networks, gradients can explode or vanish as they propagate backward through many layers. Each layer multiplies the gradient by its Jacobian, and if these Jacobians consistently have singular values greater than 1 (explosion) or less than 1 (vanishing), gradients at early layers become unusable.

Batch normalization mitigates this by keeping activations in a well-behaved range:

- For ReLU: fewer dead neurons (activations stuck at zero)
- For sigmoid and tanh: activations avoid the flat tails where gradients vanish

## Reduced Sensitivity to Initialization

Without batch normalization, weight initialization is critical. If initial weights are too large, activations explode; too small, and they vanish. Careful schemes (Xavier for sigmoid/tanh, Kaiming for ReLU) maintain activation statistics through the network.

Batch normalization reduces this sensitivity by explicitly normalizing activations at each layer. Even with suboptimal initialization, normalization corrects the statistics. Initialization still affects early training dynamics and convergence speed, but the network is much more robust to initialization choices.

## Training vs. Inference Behavior

Batch normalization behaves differently depending on whether the model is training or evaluating. This dual behavior is one of its most important and most error-prone aspects.

## During Training

The mean and variance are computed from the current mini-batch. This introduces a dependency between samples: each sample's normalized value depends on all other samples in the batch.

Alongside per-batch statistics, running estimates are maintained using exponential moving averages:

$$
\mu_{\text{running}} = (1 - m) \cdot \mu_{\text{running}} + m \cdot \mu_{\text{batch}}
$$

$$
\sigma^2_{\text{running}} = (1 - m) \cdot \sigma^2_{\text{running}} + m \cdot \sigma^2_{\text{batch}}
$$

where $m$ is the momentum parameter (default 0.1 in PyTorch). These running statistics accumulate over the entire training process and represent the model's estimate of population statistics.

## During Inference

At inference time, there may be no batch (single-sample prediction) or the batch composition may be arbitrary. Using batch statistics would make predictions non-deterministic: the same input could produce different outputs depending on what other samples happen to be in the batch.

Instead, inference uses the running statistics accumulated during training:

$$
\hat{X}_{ij} = \frac{X_{ij} - \mu_{\text{running},j}}{\sqrt{\sigma^2_{\text{running},j} + \epsilon}}
$$

Combined with learned $\gamma$ and $\beta$, this is a fixed affine transform that can be fused with the preceding linear layer for efficiency.

## The Mode Switch

Calling eval mode switches batch normalization from batch statistics to running statistics. Calling train mode switches back. Forgetting this switch is one of the most common bugs in PyTorch code:

- During validation: always use eval mode
- During training: always use train mode
- If eval mode is left on during training, batch norm does not update running statistics, leading to poor evaluation behavior later

## The Backward Pass Through Batch Normalization

Computing gradients through batch normalization requires careful application of the chain rule because normalization couples all samples in the batch.

Let $\hat{x}_i = (x_i - \mu) / \sqrt{\sigma^2 + \epsilon}$ and $y_i = \gamma \hat{x}_i + \beta$. Given upstream gradient $\partial \mathcal{L} / \partial y_i$, the gradients with respect to learnable parameters are:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \gamma}
&= \sum_{i=1}^{N} \frac{\partial \mathcal{L}}{\partial y_i} \cdot \hat{x}_i \\
\frac{\partial \mathcal{L}}{\partial \beta}
&= \sum_{i=1}^{N} \frac{\partial \mathcal{L}}{\partial y_i}
\end{aligned}
$$

The gradient with respect to the input is more complex because $\mu$ and $\sigma^2$ both depend on all $x_i$:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \bigg(
&\frac{\partial \mathcal{L}}{\partial y_i}
- \frac{1}{N} \sum_{k} \frac{\partial \mathcal{L}}{\partial y_k} \\
&- \frac{\hat{x}_i}{N} \sum_{k} \frac{\partial \mathcal{L}}{\partial y_k} \hat{x}_k
\bigg)
\end{aligned}
$$

The three terms have intuitive interpretations:

- The first passes the gradient through directly
- The second removes the mean of the gradients (because the mean was subtracted in the forward pass)
- The third removes the correlation between gradient and normalized input (because variance was divided out in the forward pass)

## Batch Normalization as Regularization

Batch normalization has a regularization effect, even though it was not designed for this purpose. The mechanism is noise from using batch statistics:

- Each sample's normalized value depends on the other samples in the batch. Different batches produce different normalization statistics, so the same input is normalized differently depending on its batch context
- This is a form of data-dependent noise injection, similar in spirit to dropout
- The effect is strongest for small batch sizes (more noise in statistics) and weakest for large batch sizes (statistics approach population values)
- This is one reason reducing batch size can improve generalization, and why very large batch sizes sometimes need additional regularization

An important consequence: using batch normalization with dropout can be problematic. Both inject noise during training, and the combination can introduce too much noise, leading to worse performance than either alone. Many modern architectures use batch normalization without dropout.

## Normalization Variants

Batch normalization normalizes across the batch dimension. Several alternatives normalize across different dimensions, each with distinct tradeoffs.

## Layer Normalization

Computes statistics across all features within a single sample:

$$
\mu_i = \frac{1}{D} \sum_{j=1}^{D} X_{ij}, \quad \sigma_i^2 = \frac{1}{D} \sum_{j=1}^{D} (X_{ij} - \mu_i)^2
$$

- Each sample is normalized independently, eliminating batch dependency
- Well-suited for sequence models (RNNs, Transformers) where batch sizes may be small or variable
- Default choice in Transformer architectures

## Instance Normalization

- Computes statistics per sample per channel, normalizing across spatial dimensions only
- For an image with shape $(C, H, W)$, each channel of each sample gets its own mean and variance
- Useful in style transfer and image generation, where global statistics carry style information that should be removed

## Group Normalization

- Divides channels into groups and normalizes within each group per sample
- Interpolates between layer normalization (one group with all channels) and instance normalization (each channel its own group)
- Works well when batch sizes are very small (e.g., detection and segmentation models where large images limit batch size)

## When to Use Each

- **Batch normalization**: Large batch sizes, convolutional networks for classification. Default choice for CNNs like ResNet
- **Layer normalization**: Transformers, RNNs, any architecture needing batch-independent normalization
- **Group normalization**: Small batch sizes, detection/segmentation models where memory constrains batch size
- **Instance normalization**: Style transfer and generative models

## Batch Normalization in Convolutional Networks

For convolutional layers, input has shape $(N, C, H, W)$: batch size, channels, height, width. BatchNorm2d computes one mean and one variance per channel, averaged over batch and spatial dimensions:

$$
\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{i=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} X_{i,c,h,w}
$$

- Every spatial position in every sample contributes to the channel's statistics, giving $N \cdot H \cdot W$ values per estimate
- Even with small batch sizes, spatial dimensions provide many samples, making statistics relatively stable
- Scale and shift parameters are per-channel: $\gamma_c$ and $\beta_c$. This respects convolution's translation invariance

## The Batch Size Dependency Problem

Batch normalization's reliance on batch statistics creates a fundamental dependency on batch size:

- For large batches ($N \geq 32$): batch mean and variance are good estimates of population statistics, batch normalization works well
- For small batches ($N \leq 4$): estimates are noisy, batch normalization can hurt performance
- Particularly problematic for object detection and semantic segmentation, where large images with feature pyramids leave room for only 1-2 images per GPU
- Solutions: group normalization, or synchronized batch normalization that aggregates statistics across multiple GPUs
- Batch size also affects running statistics quality: if trained with very small batches, the accumulated running statistics may themselves be noisy

## Where to Place Batch Normalization

The original paper placed it between the linear transformation and activation:

$$
\text{Linear} \to \text{BatchNorm} \to \text{Activation}
$$

- The reasoning: normalization should stabilize the input to the activation function
- Some practitioners place it after the activation; both orderings work in practice
- Important detail: when using batch normalization, the bias term in the preceding layer is redundant. Batch normalization subtracts the mean, eliminating any constant offset, and the $\beta$ parameter serves the same role as bias. Common practice: disable bias in layers immediately preceding batch normalization

## Batch Normalization and Weight Initialization

Before batch normalization, weight initialization was critical for deep network training. Xavier initialization (2010) maintained activation variance for tanh/sigmoid, and Kaiming initialization (2015) adapted this for ReLU. Both require analysis of activation functions and layer dimensions.

Batch normalization reduces this sensitivity because it normalizes activations explicitly:

- Even with poor initialization, output of each layer has consistent statistics
- The network still converges, just potentially slower
- Initialization still matters for the first few steps (before running statistics stabilize) and the overall optimization trajectory
- Good initialization with batch normalization leads to faster convergence than bad initialization with batch normalization, but the difference is less dramatic than without normalization

## Practical Considerations

## Epsilon Placement

The epsilon is placed inside the square root: $\sqrt{\sigma^2 + \epsilon}$, not added after: $\sqrt{\sigma^2} + \epsilon$. The difference matters for the backward pass:

- With epsilon inside: gradient is $1 / \sqrt{\sigma^2 + \epsilon}$, smooth everywhere
- With epsilon outside: gradient of the square root is $1 / (2\sqrt{\sigma^2})$, infinite when $\sigma^2 = 0$. The epsilon addition afterward cannot fix the singularity

## Population vs. Sample Variance

Batch normalization uses population variance (dividing by $N$) during training, not sample variance (dividing by $N-1$). This corresponds to the biased variance computation. Using the unbiased version gives wrong results.

## Frozen Batch Normalization

In transfer learning, it is common to freeze batch normalization layers when fine-tuning:

- Use pre-trained running statistics without updating them
- Do not train the $\gamma$ and $\beta$ parameters
- Important when the fine-tuning dataset is small or has a very different distribution: without freezing, batch statistics from the small dataset would overwrite well-estimated running statistics from large-scale pre-training

## Batch Normalization and Distributed Training

In multi-GPU training, each GPU processes a subset of the batch. Standard batch normalization computes statistics independently per GPU, so the effective batch size for normalization is the per-GPU size, not the total size. SyncBatchNorm synchronizes statistics across GPUs, giving each GPU access to the full batch's statistics. This is important when per-GPU batch size is small.

## The Debate Around Internal Covariate Shift

The original motivation was reducing internal covariate shift. However, Santurkar et al. (2018) showed that batch normalization does not actually reduce the change in layer input distributions between training steps. Networks with batch normalization can exhibit just as much distributional shift.

Instead, the benefit comes from making the optimization landscape smoother: the loss function, its gradients, and the Hessian become more Lipschitz, enabling larger learning rates and more predictable optimization. The technique works, but the reason differs from the original proposal. Understanding this helps practitioners decide when to use batch normalization vs. alternatives.
