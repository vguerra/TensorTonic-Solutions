<span style="font-size: 14px;">Gradient accumulation lets you simulate a larger effective batch size by running several forward-backward passes over small micro-batches, summing gradients, and performing a single weight update. This is standard when GPU memory cannot fit the desired batch size.</span>

## <span style="font-size: 14px;">Why batch size matters</span>

* <span style="font-size: 14px;">Larger batches give more stable gradient estimates and often smoother convergence</span>
* <span style="font-size: 14px;">Memory for activations and intermediate values grows with batch size during backprop</span>
* <span style="font-size: 14px;">When the target batch size</span> $B$ <span style="font-size: 14px;">does not fit, we process</span> $K$ <span style="font-size: 14px;">micro-batches of size</span> $B/K$ <span style="font-size: 14px;">and accumulate their gradients</span>

## <span style="font-size: 14px;">How accumulation works</span>

* <span style="font-size: 14px;">For each micro-batch: forward pass, compute loss, call</span> `.backward()`<span style="font-size: 14px;">. PyTorch adds the new gradients into the existing</span> `.grad` <span style="font-size: 14px;">tensors instead of overwriting them</span>
* <span style="font-size: 14px;">Do not call</span> `optimizer.zero_grad()` <span style="font-size: 14px;">between micro-batches within the accumulation window</span>
* <span style="font-size: 14px;">After</span> $K$ <span style="font-size: 14px;">micro-batches: divide the accumulated gradient by</span> $K$ <span style="font-size: 14px;">to average, then take one optimizer step and zero gradients before the next accumulation cycle</span>
* <span style="font-size: 14px;">Wrap the weight update in</span> `torch.no_grad()` <span style="font-size: 14px;">so the in-place update is not recorded in the computational graph</span>

## <span style="font-size: 14px;">Mathematical justification</span>

<span style="font-size: 14px;">For a loss that is a sum over samples, the gradient of the sum is the sum of the gradients. So averaging gradients over</span> $K$ <span style="font-size: 14px;">micro-batches is equivalent to one backward pass over the concatenated batch, up to the same scaling:</span>

$$
\frac{1}{K} \sum_{k=1}^{K} \nabla_\theta L_k = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell_i
$$

<span style="font-size: 14px;">when</span> $B = K \times \text{micro-batch size}$ <span style="font-size: 14px;">and</span> $L_k$ <span style="font-size: 14px;">is the loss for micro-batch</span> $k$<span style="font-size: 14px;">. So one step after</span> $K$ <span style="font-size: 14px;">accumulations matches one step with a batch of size</span> $B$<span style="font-size: 14px;">, while using only</span> $B/K$ <span style="font-size: 14px;">samples in memory at a time.</span>

## <span style="font-size: 14px;">Implementation details</span>

* <span style="font-size: 14px;">After</span> `.backward()`<span style="font-size: 14px;">,</span> `param.grad` <span style="font-size: 14px;">holds the sum of gradients from all backward calls since the last zero. Divide by</span> $K$ <span style="font-size: 14px;">before applying the optimizer if your optimizer expects the mean gradient over the effective batch</span>
* <span style="font-size: 14px;">If you use a loss that already averages over the micro-batch, the accumulated</span> `.grad` <span style="font-size: 14px;">is the sum of</span> $K$ <span style="font-size: 14px;">averaged gradients; dividing by</span> $K$ <span style="font-size: 14px;">then gives the mean over the full effective batch</span>
* <span style="font-size: 14px;">Batch normalization and similar layers see only the micro-batch; for very small micro-batches, consider</span> `nn.GroupNorm` <span style="font-size: 14px;">or running stats over the full effective batch if needed</span>
