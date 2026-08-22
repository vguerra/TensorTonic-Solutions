## <span style="font-size: 20px;">The Overfitting Problem</span>

Neural networks are universal function approximators. Given enough parameters and training time, a network can map every training input to its exact target with zero loss. This sounds like success but is actually failure: the model has memorized the training set instead of learning the underlying patterns that generalize to new data. This is overfitting.

Consider a model with parameters $\theta$ trained to minimize the empirical risk:

$$
\mathcal{L}_{\text{train}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(f_\theta(x_i), y_i)
$$

where $f_\theta$ is the model, $(x_i, y_i)$ are training examples, and $\ell$ is a per-sample loss. The true goal, however, is to minimize the population risk: the expected loss over the entire data distribution, including unseen examples. We approximate this with a held-out validation set:

$$
\mathcal{L}_{\text{val}}(\theta) = \frac{1}{M} \sum_{j=1}^{M} \ell(f_\theta(x_j'), y_j')
$$

During healthy training, both losses decrease together. The model is learning genuine patterns. At some point, they diverge:

$$
\begin{aligned}
&\mathcal{L}_{\text{train}}(\theta^{(e+1)}) < \mathcal{L}_{\text{train}}(\theta^{(e)}) \\
&\text{but} \quad \mathcal{L}_{\text{val}}(\theta^{(e+1)}) > \mathcal{L}_{\text{val}}(\theta^{(e)})
\end{aligned}
$$

Training loss keeps falling while validation loss starts rising. The model is using its capacity to fit noise, outliers, and training-set-specific quirks that do not transfer. The optimal model lies near the point where validation loss reaches its minimum.

## The Bias-Variance Tradeoff

Overfitting connects to a fundamental decomposition of prediction error. For any estimator, expected prediction error on a new point decomposes as:

$$
\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}
$$

- **Bias** measures how far the average prediction is from the truth. A model that is too simple has high bias: it consistently misses the target because it cannot represent the true function. This is underfitting
- **Variance** measures how much predictions change across different training sets. A model that is too complex has high variance: it fits each training set closely but gives wildly different predictions depending on which data it saw. This is overfitting
- Early stopping controls this tradeoff by halting at the sweet spot where combined error is lowest
- Early in training: underfitting (high bias, low variance). As training continues, bias decreases but variance increases
- Validation loss captures the combined effect, and early stopping halts at the inflection point

## Training Curves as Diagnostics

- Both curves decreasing together: learning genuine patterns, continue training
- Validation plateaus while training decreases: starting to overfit, generalization gap widening
- Validation increases while training decreases: clear overfitting, past the optimal point
- Both plateau at a high value: underfitting, need more capacity, not early stopping
- Validation noisy but trending down: learning rate may be too high, or validation set too small

## Early Stopping as Implicit Regularization

Regularization is any technique that reduces the generalization gap without reducing training set performance proportionally. Explicit regularization like L2 weight decay adds a penalty:

$$
\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \lambda \|\theta\|^2
$$

This penalty discourages large parameter values, effectively shrinking the model's hypothesis space. Early stopping achieves a similar effect through a completely different mechanism.

## The Connection to L2 Regularization

For linear models with gradient descent, there is a precise mathematical equivalence. Consider a linear model $f(x) = Wx$ initialized at $W_0 = 0$ and trained with learning rate $\eta$ for $T$ steps on MSE loss. The gradient descent update is:

$$
W_{t+1} = W_t - \eta \nabla \mathcal{L}(W_t) = W_t - \eta(W_t X^T X - X^T Y)
$$

After $T$ steps, the solution can be written using the eigendecomposition $X^T X = U \Lambda U^T$:

$$
W_T = \sum_i \left(1 - (1 - \eta \lambda_i)^T\right) \frac{u_i u_i^T}{\lambda_i} X^T Y
$$

Compare with the L2-regularized (closed-form) solution:

$$
\begin{aligned}
W_\lambda &= (X^T X + \lambda I)^{-1} X^T Y \\
&= \sum_i \frac{\lambda_i}{\lambda_i + \lambda} \frac{u_i u_i^T}{\lambda_i} X^T Y
\end{aligned}
$$

Both expressions have identical structure: they filter each eigenvector's contribution by a factor between 0 and 1.

- The early stopping factor $1 - (1 - \eta \lambda_i)^T$ approaches 1 for large eigenvalues (well-represented directions in the data) and stays near 0 for small eigenvalues (noise directions). Large eigenvalues correspond to strong signals that are learned quickly; small eigenvalues correspond to noise patterns that require many iterations to fit
- The L2 factor $\lambda_i / (\lambda_i + \lambda)$ performs the same filtering: large eigenvalues pass through nearly unchanged, while small eigenvalues are suppressed
- Fewer training steps corresponds to stronger regularization (larger effective $\lambda$)
- For nonlinear neural networks, this exact equivalence breaks down, but the intuition carries over: limiting training time limits how far parameters can move from initialization, constraining the effective complexity of the learned function

## Why Initialization Matters

Since early stopping constrains how far parameters travel from their initial values, the initialization itself becomes part of the regularization. Random initialization typically places the model in a region of parameter space that corresponds to simple functions (near-zero weights produce near-linear behavior). Gradient descent then progressively fits more complex patterns. Stopping early means the model stays closer to this simple starting point.

This is why initialization schemes like Xavier and Kaiming are important beyond just enabling stable gradient flow: they also set the implicit regularization baseline for early stopping.

## The Patience Mechanism

Validation loss is not a smooth monotonic curve. It fluctuates due to several sources of noise:

- Mini-batch sampling effects cause the training updates to be stochastic. Two consecutive epochs can push the model in slightly different directions, causing temporary validation loss increases that reverse in subsequent epochs
- The validation set itself is a finite sample. Especially for small validation sets, the estimated loss has high variance around the true population loss
- Learning rate dynamics can cause oscillation. A learning rate that is slightly too high causes the model to overshoot, recover, and overshoot again, leading to zigzag patterns in both losses

Stopping at the first sign of validation loss increase would be premature in all these cases. The patience parameter $p$ addresses this by requiring $p$ consecutive epochs of non-improvement before halting. The algorithm tracks:

$$
\mathcal{L}_v^* = \min_{i \leq e} \mathcal{L}_v^{(i)}
$$

At each epoch $e$:

- If $\mathcal{L}_v^{(e)} < \mathcal{L}_v^*$: the best loss is updated and the patience counter resets to zero
- Otherwise: the counter increments
- When the counter reaches $p$: training terminates

## Choosing the Right Patience

The patience value encodes a tradeoff between two risks:

- Too small ($p = 1$): the algorithm is too aggressive. Any single-epoch fluctuation triggers stopping. The model may stop well before the true optimum, leaving significant performance on the table
- Too large ($p = 50$): the algorithm is too lenient. The model trains for many unnecessary epochs after it has started overfitting, wasting compute and potentially settling into a worse region
- Common values range from 3 to 10. As a rule of thumb, noisier training (smaller batches, higher learning rates, smaller validation sets) requires more patience. If the learning curve is smooth and well-behaved, less patience suffices
- An important subtlety: the patience counter tracks consecutive failures. A single improved epoch resets the counter entirely. This means the total number of "wasted" epochs after the true optimum is at most $p$, but the total training time can extend well beyond the optimal epoch if there are occasional improvements interspersed with degradations

## Improvement Criteria

The simplest criterion is absolute improvement: does the new validation loss beat the current best? This is what most basic implementations use. But there are several alternatives worth understanding.

## Minimum Delta

Some implementations require a minimum improvement threshold $\delta$:

$$
\mathcal{L}_v^{(e)} < \mathcal{L}_v^* - \delta
$$

This ignores negligibly small improvements that may just be noise. Without a delta, the counter can reset on a vanishingly small improvement that is statistically meaningless, extending training unnecessarily. Choosing $\delta$ requires some knowledge of the loss scale: a delta of 0.001 is meaningful when the loss is around 1.0 but negligible when the loss is around 100.

## Relative Improvement

A relative threshold adapts to the scale of the loss:

$$
\mathcal{L}_v^{(e)} < \mathcal{L}_v^* \cdot (1 - \delta_{\text{rel}})
$$

This is useful when the loss magnitude varies significantly across problems or during training. A relative delta of 0.01 means the loss must improve by at least 1% to count.

## Smoothed Loss

Some practitioners use an exponential moving average of the validation loss rather than the raw value:

$$
\bar{\mathcal{L}}_v^{(e)} = \alpha \cdot \mathcal{L}_v^{(e)} + (1 - \alpha) \cdot \bar{\mathcal{L}}_v^{(e-1)}
$$

with smoothing factor $\alpha \in (0, 1)$. This reduces sensitivity to individual outlier epochs but introduces a lag: the smoothed loss responds more slowly to genuine trend changes. A small $\alpha$ (e.g., 0.1) gives heavy smoothing but large lag; a large $\alpha$ (e.g., 0.9) preserves responsiveness but provides less noise reduction.

## Training vs. Validation Loops

Each epoch has two phases:

- **Training phase**: Model in train mode (activates dropout, batch norm uses batch stats). For each batch: zero gradients, forward pass, backward pass, optimizer step. Accumulate extracted scalar loss values to avoid retaining the graph
- **Validation phase**: Model in eval mode (dropout off, batch norm uses running stats). No gradients computed, no parameter updates. Loss computed purely for monitoring

## Why Mode Switching Matters

- Dropout adds randomness during training but passes all activations during evaluation. Leaving train mode on during validation makes the loss noisier
- Batch normalization uses per-batch statistics during training but running (population) statistics during evaluation. Batch statistics for validation would make loss depend on batch composition
- Forgetting to switch is the most common training loop bug; symptoms are subtle

## Checkpoint and Restore Patterns

Basic early stopping halts training and returns the model at the stopped epoch. But the stopped epoch is not the best epoch: it is $p$ epochs after the best epoch. A better pattern saves and restores the best model:

- When the validation loss improves, save a copy of the model's state dictionary (a dictionary mapping parameter names to their tensor values)
- When the patience counter triggers stopping, load the saved state dictionary back into the model before returning
- This ensures the returned model has the parameters from the actual best epoch, not the final (worse) epoch

## Saving to Disk vs. Memory

- For small models, copying the state dictionary in memory is fast and sufficient
- For large models, saving checkpoints to disk is more robust: if training crashes, the best checkpoint survives. The tradeoff is I/O overhead
- A common compromise is to keep the best state dictionary in memory but periodically write disk checkpoints as a safety net

## Relationship with Other Regularization Methods

Early stopping does not exist in isolation. It interacts with other regularization techniques in important ways.

## Weight Decay

L2 weight decay adds $\lambda \|\theta\|^2$ to the loss, penalizing large weights. Since early stopping also constrains parameter magnitude (by limiting how far they move from initialization), combining both can lead to over-regularization: the model underfits because it is too constrained. In practice, you can use both together, but each requires less strength when the other is present. If you use strong weight decay, you may need more patience (or no early stopping at all). If you rely heavily on early stopping, you may need less weight decay.

## Dropout

Dropout randomly zeros neurons during training, forcing the network to learn redundant representations. This adds noise to the training process, which makes the validation loss noisier and may require higher patience values for early stopping. Dropout and early stopping are complementary: dropout regularizes within each epoch, while early stopping regularizes across epochs.

## Data Augmentation

Augmenting the training data (random crops, flips, color jitter) effectively increases the dataset size, which delays overfitting. When using strong data augmentation, the optimal stopping point shifts to later epochs, so early stopping still applies but with more patience. Without augmentation, overfitting starts sooner, and early stopping plays a more prominent role.

## Learning Rate Scheduling

Learning rate schedulers reduce the learning rate during training, which can cause the validation loss to improve again after a plateau. This interacts with early stopping: if the patience is too low, training might stop just before a scheduled learning rate drop that would have improved the model. Many practitioners use early stopping in combination with reduce-on-plateau scheduling, where the learning rate decreases when validation loss stagnates, and training only stops after the reduced learning rate also fails to improve the loss.

## Practical Implementation Details

- **Loss averaging**: Use the average over batches, not the sum. If batch counts differ between training and validation, sums are not comparable. The last batch may be smaller; weighting by batch size is more precise but simple averaging usually suffices
- **Scalar extraction**: The loss tensor carries its computational graph. Extracting the float value discards the graph, preventing memory leaks and converting to a plain number for logging
- **Epoch indexing**: Be consistent about 1-indexed vs. 0-indexed. "Stopped at epoch 5" should clearly mean 5 complete epochs occurred
- **When stopping does not trigger**: If all max epochs complete, check whether validation loss is still decreasing (model needed all epochs) or patience was too high (overfitting occurred undetected)

## Common Mistakes

- **Forgetting mode switching**: The most common bug. Symptoms are subtle: validation loss is noisier than expected (dropout still active) or evaluation performance is inconsistent (batch norm using per-batch statistics)
- **Forgetting to disable gradients during validation**: Does not produce incorrect results but wastes memory by building a computational graph that is never used. For large models, this can cause out-of-memory errors during the validation pass
- **Using the stopped-epoch model**: The stopped epoch is $p$ epochs after the best epoch, so the model at the stopped epoch has degraded from the peak. Always restore the best checkpoint
- **Resetting patience incorrectly**: Using $\leq$ instead of $<$ in the comparison means the counter resets on ties, which is rarely desired behavior
- **Monitoring training loss**: Training loss always decreases (or should, with a properly tuned learning rate), so the patience counter would never trigger. Early stopping must monitor validation loss to detect overfitting
