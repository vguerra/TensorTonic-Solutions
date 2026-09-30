# <span style="font-size: 20px;">Cross Entropy Loss (Mean Reduction)</span>

<span style="font-size: 14px;">Cross entropy is the first kernel in the curriculum that needs **cross-program** combination. Each program computes a per-row scalar (the negative log likelihood of that row's target), and every row's scalar has to be summed into a single global loss value before the host divides by the batch size. That is the canonical use case for $\texttt{tl.atomic\_add}$ into a length-$1$ buffer, combined with the numerical-stability trick that every log-sum-exp implementation eventually adopts: subtract the row maximum before exponentiating.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given logits of shape $(B, C)$ and integer targets of shape $(B,)$, the mean-reduction cross entropy is</span>

$$
\text{loss} = \frac{1}{B} \sum_{i=0}^{B-1} \left( \log \sum_{c=0}^{C-1} e^{\texttt{logits}[i, c]} - \texttt{logits}[i, t_i] \right)
$$

<span style="font-size: 14px;">The per-row term is the gap between the row's log-sum-exp and the logit at the target index. Each row independently produces a scalar; the global loss is the mean of those scalars. The stable form factors a per-row max $m_i = \max_c \texttt{logits}[i, c]$ out of the exponentials:</span>

$$
\text{loss}_i = m_i + \log \sum_{c=0}^{C-1} e^{\texttt{logits}[i, c] - m_i} - \texttt{logits}[i, t_i]
$$

<span style="font-size: 14px;">After the substitution, the largest exponent inside the sum is exactly zero, so the sum is bounded in $[1, C]$ and the log of the sum is bounded in $[0, \log C]$. Without the subtraction, $e^{\texttt{logits}}$ can overflow fp32 for any row containing a logit above $\approx 88$.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">Rows are independent up to the final mean. The launch grid is $(B,)$, one **program** per row. Each program reads its row index from $\texttt{tl.program\_id(0)}$, computes its row offset as $\texttt{row} \cdot \texttt{stride\_logits\_row}$, and operates on the whole row in registers. The cross-program combine is a single $\texttt{tl.atomic\_add}$ of the per-row loss into a length-$1$ buffer; the divide by $B$ is done on the host after the kernel returns.</span>

<span style="font-size: 14px;">This is the simplest possible cross-program reduction: one scalar per program, atomic-added into one global accumulator. Compared to a tree reduction across rows (which would need a second kernel pass over the per-row losses), the atomic form is one kernel launch instead of two and has no temporary buffer. The price is that the atomic-add order is non-deterministic, so the final loss has accumulator-order-dependent rounding noise.</span>

<span style="font-size: 14px;">The host-side divide by $B$ is a single $\texttt{loss\_out.div\_(B)}$ call. Done inside the kernel it would require either every program to know $B$ (fine) and write the divided value (creates an additional race condition, since the atomic-add is into a sum, not a sum-then-divide), or a second kernel pass. The cleanest split is to atomically add the row losses and divide once on the host.</span>

<span style="font-size: 14px;">An alternative decomposition is to write each per-row loss into a length-$B$ buffer ($\texttt{out}[\texttt{row}] = \text{row\_loss}$) and follow with a separate reduction kernel. That removes the atomic contention but adds a kernel launch and a temporary buffer. For training batches in the hundreds-to-thousands of rows, the atomic form wins on aggregate latency because the atomic contention on a length-$1$ buffer is tolerable at that scale; for very large $B$ (tens of thousands), the two-stage form becomes competitive.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">Tile shape is $(\texttt{BLOCK\_SIZE},)$ with $\texttt{BLOCK\_SIZE} = \texttt{triton.next\_power\_of\_2}(C)$, declared $\texttt{tl.constexpr}$. For language-model vocabularies ($C$ in the tens of thousands), the block size may approach the per-program register budget; for small classifiers ($C = 10, 100, 1000$), it is comfortably inside. This problem assumes $C$ fits in one tile, which covers the test harness's sizes.</span>

<span style="font-size: 14px;">The tail mask $\texttt{cols} < C$ disables overshoot. The load of the logits row uses $\texttt{other} = -\infty$, not $0.0$: a zero would silently participate in $\texttt{tl.max}$ (becoming the maximum for any row of negative logits) and would also contribute $e^0 = 1$ to the exp-sum. $-\infty$ is the identity for $\max$ and produces $e^{-\infty} = 0$ for the sum, so both reductions are exact regardless of the tail-lane count.</span>

<span style="font-size: 14px;">There is no mask on the target load: the target is a single integer per program, addressed by $\texttt{target\_ptr} + \texttt{row}$, with no tile structure. The target-logit load addresses $\texttt{logits\_ptr} + \texttt{row} \cdot \texttt{stride\_logits\_row} + \texttt{target\_id}$ - one scalar fetched from the row at the target column. This is the only place in the kernel where the program reads a position that is not part of the full-row tile, and it is by design: the target logit is one specific value, not a reduction. In principle the target logit could be picked out of the row tile rather than reloaded, but the scalar reload is simpler and the bandwidth cost is negligible.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">The logits row is loaded once into a $(\texttt{BLOCK\_SIZE},)$ tile and held in registers for three operations: the row max, the shift-and-exp into the sum, and (potentially) the gather of the target logit. The single load amortizes over three arithmetic touches. The target integer is one HBM load, and the target-logit gather is another. The atomic store into the length-$1$ buffer writes $4$ bytes total per program.</span>

<span style="font-size: 14px;">Per row, HBM traffic is $4C$ bytes of logits, $4$ bytes of target, $4$ bytes of target logit (duplicated from inside the row, possibly served by the L2), and $4$ bytes of atomic write. Total $\approx 4C + 12$ bytes per program. The $12$ overhead is negligible for any realistic $C$. The atomic write contends with every other program writing into the same address, but each program contributes only $4$ bytes per launch, so contention bandwidth is at most $4B$ bytes for the whole launch.</span>

<span style="font-size: 14px;">There is no shared memory in the kernel proper. Both intra-row reductions ($\texttt{tl.max}$ and $\texttt{tl.sum}$) collapse the tile to a scalar via the compiler's tree reduction, which uses a small SRAM staging area between warps. The author does not name the staging area or insert any barrier.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per logit element the kernel does about $5$ FLOPs (one max contribution, one shift subtract, one exp, one sum contribution, and a small amortized share of the log and the target subtract). Bytes moved are $4$ per element on the load side, with the per-row output and the atomic add amortized to near zero. Arithmetic intensity:</span>

$$
\frac{\approx 5 \text{ FLOPs}}{\approx 4 \text{ bytes}} \approx 1.25 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">That is higher than LayerNorm and RMSNorm but still firmly **memory-bound** on modern accelerators (whose roofline crossover is in the $10$ FLOPs/byte range). The exp dominates the arithmetic count; the rest is a max and a sum. The kernel's runtime is set by HBM bandwidth on the logits read, not by the exp pipeline.</span>

<span style="font-size: 14px;">A useful comparison: fused softmax has nearly the same intensity (same exp, same per-row max-subtract, but writes a full normalized row instead of one scalar). Cross entropy is essentially softmax that throws away the per-element normalized values and keeps only the log of the normalizer minus the target logit. The shared bandwidth ceiling explains why fused softmax and cross-entropy benchmarks track each other within a few percent on the same input shape; the cross-entropy kernel saves the output write (one scalar instead of a $C$-wide row) but pays a small atomic-add cost in exchange.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author chooses:** the grid $(B,)$, the constexpr block size as $\texttt{next\_power\_of\_2}(C)$, the tail mask, the $\texttt{other} = -\infty$ on the logits load, the max-subtract before the exp, the gather of the target logit via a scalar pointer load, the atomic-add into the length-$1$ buffer, and the host-side divide by $B$.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering $\texttt{tl.max}$ and $\texttt{tl.sum}$ to tree reductions, picking the vector width for the row load, allocating registers for the row tile, scheduling the exp and the sum, and lowering $\texttt{tl.atomic\_add}$ to a hardware atomic instruction. The author writes one symbol for the atomic; the compiler ensures the atomicity holds across all programs writing to the same address.</span>

<span style="font-size: 14px;">The author also decides what happens at the boundary between the per-row reduction (deterministic inside one program) and the cross-row combine (nondeterministic via atomics). The harness tolerance of $\texttt{atol} = 10^{-2}, \texttt{rtol} = 10^{-2}$ exists precisely because the atomic-add order is not fixed, and the final loss can differ in the last few decimal digits between runs. Tightening the tolerance below $10^{-3}$ would fail every other launch.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A non-fused PyTorch eager implementation is at least three kernels: one to compute the row max, one to subtract and exp and sum, one to take the log and subtract the target logit. (In practice $\texttt{F.cross\_entropy}$ is itself fused, but the naive decomposition is what the author would reach for without a fused primitive.) Each unfused step writes intermediate values to HBM and reads them back; the logits row would be touched three to four times.</span>

<span style="font-size: 14px;">The fused Triton form reads the logits row once and writes only the per-row scalar (atomically combined into a single global value). HBM traffic on the logits drops by $\approx 3\times$, and because the kernel is memory-bound, the runtime drops by a similar factor.</span>

<span style="font-size: 14px;">A further fusion is the gradient computation. The cross-entropy backward produces $\partial \text{loss} / \partial \texttt{logits}[i, c] = (1/B) \cdot (\text{softmax}(\texttt{logits}[i])_c - [c = t_i])$. A production kernel fuses the forward loss, the softmax computation, and the gradient store into one kernel, so the gradient is materialized in one HBM round-trip with no separate softmax tensor. The standalone forward kernel here is the teaching baseline; the deployed form is the fused forward + backward.</span>

<span style="font-size: 14px;">A vocabulary-projection-aware optimization, used in the largest language models, fuses the cross entropy with the unembedding matrix multiply that produced the logits. The unembedding output ($\text{hidden}$ times the embedding matrix) is the logits, and a fused kernel can compute the per-token loss directly from the hidden state without ever materializing the $C$-wide logits tensor. For vocabularies in the hundreds of thousands, this is the difference between materializing a $B \cdot V$ logits tensor (gigabytes) and computing a length-$B$ loss vector directly. The standalone kernel here assumes the logits are already in HBM.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $B = 2$, $C = 3$, $\texttt{BLOCK\_SIZE} = 4$, $\texttt{logits} = [[1, 2, 3], [0, 0, 5]]$, $\texttt{target} = [2, 0]$. The launch grid is $(2,)$.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{row} = 0$): load the row into the tile as $[1, 2, 3, -\infty]$ (lane $3$ is past $C$ and uses $\texttt{other} = -\infty$). $\texttt{tl.max} = 3$. Shifted exponentials: $e^{1-3} + e^{2-3} + e^{3-3} + 0 = 0.135 + 0.368 + 1.000 = 1.503$. $\log(1.503) + 3 = 0.408 + 3 = 3.408$. Target index is $2$, target logit is $\texttt{logits}[0, 2] = 3$. Row loss: $3.408 - 3 = 0.408$. Atomic-add $0.408$ into $\texttt{loss\_out}[0]$.</span>

<span style="font-size: 14px;">**Program 1** ($\texttt{row} = 1$): tile is $[0, 0, 5, -\infty]$. $\texttt{tl.max} = 5$. Shifted exponentials: $e^{0-5} + e^{0-5} + e^{5-5} + 0 = 0.0067 + 0.0067 + 1.000 = 1.013$. $\log(1.013) + 5 = 0.013 + 5 = 5.013$. Target index is $0$, target logit is $0$. Row loss: $5.013 - 0 = 5.013$. Atomic-add $5.013$.</span>

<span style="font-size: 14px;">After both programs complete, $\texttt{loss\_out}[0] = 0.408 + 5.013 = 5.421$. The host divides by $B = 2$ to get the final mean loss $2.710$. The shift-by-max made the second row stable: without it, $e^5 \approx 148$ is fine in fp32 but a row containing $\texttt{logits} = 100$ would have produced $e^{100} \approx 2.7 \times 10^{43}$, well past fp32's max of $\approx 3.4 \times 10^{38}$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Skipping the max-subtract.** A naive $\log \sum e^{\texttt{logits}}$ overflows fp32 on any row whose logits exceed $\approx 88$. Real language-model logits routinely range over tens of units, and even modest scaling can push them past the overflow threshold. The max-subtract is required for correctness on any row, not an optimization.</span>
* <span style="font-size: 14px;">**Loading masked lanes as $0.0$.** Zero is not the identity for either $\texttt{tl.max}$ or the exp-sum. A zero in the max pool wins against any negative logit; a zero in the exp-sum contributes $e^0 = 1$. The correct $\texttt{other}$ value is $-\infty$, which is the additive identity for $\max$ and gives $e^{-\infty} = 0$ for the sum.</span>
* <span style="font-size: 14px;">**Forgetting to pre-zero the loss buffer.** $\texttt{tl.atomic\_add}$ accumulates onto whatever value the buffer already holds. Without $\texttt{loss\_out.zero\_()}$ before the launch, the kernel adds the per-row losses to the previous launch's residual, silently producing nonsensical losses on every run after the first.</span>
* <span style="font-size: 14px;">**Tightening the tolerance below the atomic-order noise.** Atomic-add ordering is nondeterministic, so the final summed loss varies in its last few decimal digits between runs. Combined $\texttt{atol} = 10^{-2}, \texttt{rtol} = 10^{-2}$ is the realistic tolerance for the atomic form; anything tighter will flake even for a correct implementation.</span>

---