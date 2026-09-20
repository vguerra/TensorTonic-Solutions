# <span style="font-size: 20px;">Row-wise LogSumExp</span>

<span style="font-size: 14px;">LogSumExp emits one scalar per row instead of one normalized row of outputs, but it shares the entire stability story and the entire program shape with fused softmax. The shift-and-exp pattern is the same; the divide is replaced by a log, applied to the row sum once after the reduction, plus an additive correction that puts the subtracted max back. The kernel is the most compact possible demonstration of how a numerically-stable identity becomes a one-pass register-resident pipeline once the row fits in a single tile.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For each row $i$ of an $(M, N)$ float matrix, the kernel emits</span>

$$
\texttt{out}[i] = \log \sum_{j=0}^{N-1} \exp(x[i, j])
$$

<span style="font-size: 14px;">The direct evaluation overflows fp32 for any logit above about $88$. The stable form pulls the row max out:</span>

$$
\texttt{out}[i] = m_i + \log \sum_{j=0}^{N-1} \exp(x[i, j] - m_i), \qquad m_i = \max_j x[i, j]
$$

<span style="font-size: 14px;">The identity is exact in real arithmetic ($\log(e^{m} \cdot S) = m + \log S$) and the shift makes every exponent argument non-positive, capping $\exp$ outputs in $(0, 1]$ and keeping the sum representable.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional with $M$ **programs**, one per input row. Each program reads its row index from $\texttt{tl.program\_id(0)}$, loads its full row into a single $\texttt{BLOCK\_SIZE}$-lane register tile, runs two reductions ($\texttt{tl.max}$ then $\texttt{tl.sum}$ of shifted exps), and writes one fp32 scalar to $\texttt{out}[\texttt{row\_idx}]$. Rows are independent under the operation, so the grid is embarrassingly parallel at the program level even though the per-row work contains two reductions.</span>

<span style="font-size: 14px;">The decomposition is the same **row-parallel reduction** shape as fused softmax, with one difference at the output: softmax writes $N$ values per row, logsumexp writes $1$. That difference makes the output bandwidth requirement asymmetric. Softmax is $N$-in, $N$-out per row; logsumexp is $N$-in, $1$-out per row. The kernel still reads the row exactly once, so the in-bandwidth is identical, but the out-bandwidth drops from $4 N M$ to $4 M$ bytes total. For long rows that fold up into a single scalar, the output stream is negligible compared with the input stream.</span>

<span style="font-size: 14px;">As with softmax, the cap on the row-parallel form is that the entire row must fit in a single Triton block of registers. For row widths up to tens of thousands, that is fine. Beyond that, the kernel must switch to a chunked online formulation that streams the row through fixed-size chunks while maintaining a running max and a running normalizer (the same recurrence FlashAttention uses). The current problem stays in the single-tile regime.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">$\texttt{BLOCK\_SIZE} = \texttt{triton.next\_power\_of\_2}(N)$, computed on the host and passed as a $\texttt{tl.constexpr}$. The power-of-two extent lets the compiler unroll the two reduction trees ($\texttt{tl.max}$ and $\texttt{tl.sum}$) at compile time and pick a vector load width for the row. As with softmax, the runtime row width $N$ is decoupled from the compile-time block extent, and the bridge between them is the tail mask.</span>

<span style="font-size: 14px;">The mask is $\texttt{mask} = \texttt{cols} < N$ on the $\texttt{tl.load}$, paired with $\texttt{other} = -\texttt{float('inf')}$. The sentinel choice is non-negotiable: masked lanes must lose the row-max comparison ($-\infty < $ anything finite) and must contribute zero to the shifted exp sum ($\exp(-\infty - m_i) = 0$). The $-\infty$ sentinel does both, and it is the only value that does both. A naive $\texttt{other=0.0}$ would corrupt the row max for all-negative rows and would contribute $\exp(0 - m_i)$ to the sum for ordinary rows, inflating the final $\log$.</span>

<span style="font-size: 14px;">The store side is a single $\texttt{tl.store(out\_ptr + row\_idx, lse)}$ with no mask, because each program owns exactly one output address and never overshoots. This is the bandwidth payoff for emitting a scalar per row: no store-side masking is needed at all.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Each program reads its row from HBM into a register tile in one logical $\texttt{tl.load}$, runs the in-register pipeline (max, subtract, exp, sum, log, add), and writes one fp32 to HBM. The row tile is loaded once and reused by every subsequent operation: the same tile feeds the max reduction, the subtract-and-exp transform, and the sum reduction. No intermediate value ever spills to HBM, and the compiler does not stage anything into shared memory at the author's instruction (the compiler does stage cross-warp partials for both reductions, but transparently).</span>

<span style="font-size: 14px;">Total HBM traffic per launch is $4 N M$ bytes of input and $4 M$ bytes of output, against the naive three-kernel implementation that would move $4 \cdot 4 N M$ bytes (read $x$ for the max, write a per-row max buffer, read $x$ and the maxes for the exp-and-sum, write a per-row sum buffer, then a final tiny kernel takes the log and adds the max). The fusion saves roughly $4 \times$ the HBM traffic on what is otherwise an identical computation, and on a memory-bound kernel that translates almost directly into a $4 \times$ runtime improvement.</span>

<span style="font-size: 14px;">The output side is cheap. Each program issues one $\texttt{tl.store}$ to a distinct address, no two programs write to the same cell, and the writes are fully parallel. The grid of $M$ programs writes $M$ fp32 values to a length-$M$ output, contiguous, and the compiler can emit those as coalesced stores when neighboring programs happen to land on neighboring SMs.</span>

<span style="font-size: 14px;">A useful number: at $M = 1024$, $N = 4096$, the input is $16$ MB and the output is $4$ KB. The kernel completes in $\approx 16$ microseconds at $1$ TB/s of HBM bandwidth, plus launch overhead. The output write is statistically free, dominated entirely by the input stream. At $M = 16$ and $N = 4096$, the input drops to $256$ KB and the launch overhead starts to dominate; that is the regime where small-batch inference lives, and where batching multiple rows into a single program (instead of one program per row) might pay back. For the canonical training-batch shape with $M$ in the hundreds or thousands, one-program-per-row is exactly right.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per input element, the kernel reads $4$ bytes, performs roughly two FLOPs (one subtract, one exp; the contributions of the row max, log, and add amortize over the row), and writes essentially nothing (one fp32 per row, not per element). **Arithmetic intensity** is</span>

$$
\frac{2 \text{ FLOPs}}{4 \text{ bytes}} = 0.5 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">well under the roofline crossover of $\approx 10$ FLOPs/byte. The kernel is firmly memory-bound, and the runtime is set by HBM bandwidth for the input stream plus launch overhead. The exp is the most expensive arithmetic step but it is hidden behind load latency on a bandwidth-limited kernel; the log is a single op per row and is essentially free.</span>

<span style="font-size: 14px;">The stability argument generalizes beyond logsumexp. The identity $\log \sum \exp(x) = m + \log \sum \exp(x - m)$ holds for any choice of $m$; choosing $m = \max_j x_j$ is the choice that minimizes the dynamic range of the post-shift values. Every $\exp$ argument ends up in $(-\infty, 0]$, every $\exp$ output in $(0, 1]$, and the sum is at least $1$ (the lane with the original max contributes $\exp(0) = 1$). The final $\log$ therefore operates on an input that is at least $1$, which means the log is well-defined and produces a finite non-negative number. Adding back $m_i$ gives the unshifted LSE. There is no choice of $m$ that breaks the identity; there are many choices that break the stability.</span>

<span style="font-size: 14px;">The test harness includes a $\texttt{large\_positive}$ pattern (logits in $[0, 20]$) and a $\texttt{large\_range}$ pattern ($[-50, 50]$), specifically to catch kernels that compute $\log \sum \exp$ without the max-subtract. Both patterns produce $\texttt{+inf}$ inside the kernel's exp call without the shift, propagate to $\log(+\infty) = +\infty$, and surface as NaN-failure in the harness.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author chooses:** the grid of $M$ programs, the constexpr $\texttt{BLOCK\_SIZE} = \texttt{next\_power\_of\_2}(N)$, the row-stride arithmetic, the load mask with the $-\infty$ sentinel, the explicit shift-by-max followed by exp-and-sum, the final $m_i + \log(\cdot)$ reassembly. The shift-and-restore identity is the central numerical-stability decision, mathematical rather than compiler-discoverable.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering the row $\texttt{tl.load}$ to coalesced HBM transactions, lowering $\texttt{tl.max}$ and $\texttt{tl.sum}$ to warp-shuffle reduction trees with shared-memory staging, lowering $\texttt{tl.exp}$ and $\texttt{tl.log}$ to fast fp32 approximations, scheduling the load and the in-register pipeline so that arithmetic overlaps load latency, and emitting the final scalar $\texttt{tl.store}$ as a single fp32 write. The author never names a warp, never declares shared memory, and never inserts a barrier.</span>

<span style="font-size: 14px;">One subtle point about the compiler-side work: inside one program, the same register tile must feed both reductions, and the compiler must preserve the tile across the first reduction so the second can reuse it. Triton tracks tile liveness through the SSA representation, so the author's expression $\texttt{tl.sum(tl.exp(row - row\_max), axis=0)}$ is materialized as a tree where the input tile, the row max, and the shifted-and-exp'd tile all live in registers simultaneously, with the compiler handling spills if the tile is too large. For typical row widths the spills do not occur and everything stays register-resident.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive logsumexp is $\texttt{tl.log(tl.sum(tl.exp(row), axis=0))}$, three lines that compile cleanly and produce NaN on any row with logits above about $88$. The optimization (which is a correctness fix) is the shift-and-restore: take the row max, subtract it from every lane, exp the shifted values, sum, log, and add the max back. The extra cost is one $\texttt{tl.max}$ reduction and one lane-wise subtract, both of which run in registers and barely move the runtime.</span>

<span style="font-size: 14px;">A more interesting axis is the relationship to fused softmax. Both kernels have identical loads, identical masks, identical row-max reductions, and identical shifted-exp sums. They differ only at the end: softmax divides each lane by the sum and writes $N$ lanes back; logsumexp computes $\log(\texttt{sum})$, adds the max, and writes one scalar. In a model that needs both (for instance, classification training that wants softmax probabilities and cross-entropy loss derived from logsumexp), a single fused kernel can emit both outputs in one pass, halving the input bandwidth relative to two separate kernels. The current problem just emits the LSE, but the shared structure is worth knowing about.</span>

<span style="font-size: 14px;">For very long rows beyond a single tile, both kernels degenerate into the same chunked online recurrence, where a running max and a running normalizer are updated chunk by chunk. That is the FlashAttention pattern; for the current problem size it is unnecessary. The recurrence is worth stating once for grounding: given the running pair $(m, \ell)$ and a new chunk with local max $m'$ and local sum-of-exps $s'$, the merged state is $m_{\text{new}} = \max(m, m')$ and $\ell_{\text{new}} = \ell \cdot \exp(m - m_{\text{new}}) + s' \cdot \exp(m' - m_{\text{new}})$, with the final LSE being $m_{\text{new}} + \log \ell_{\text{new}}$. The two-pass form here is a special case where the row fits in one chunk and the recurrence collapses to a single step.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take one row $x = [0, 1, 2]$ with $N = 3$. The host computes $\texttt{BLOCK\_SIZE} = \texttt{next\_power\_of\_2}(3) = 4$ and launches one program for that row.</span>

<span style="font-size: 14px;">The program builds $\texttt{cols} = [0, 1, 2, 3]$ and $\texttt{mask} = [\texttt{T}, \texttt{T}, \texttt{T}, \texttt{F}]$. The $\texttt{tl.load}$ with $\texttt{other} = -\infty$ produces $\texttt{row} = [0, 1, 2, -\infty]$. The first reduction gives $\texttt{row\_max} = \texttt{tl.max(row)} = 2$. The lane-wise subtract yields $[-2, -1, 0, -\infty]$. The lane-wise exp produces $[e^{-2}, e^{-1}, 1, 0]$, with the masked lane exact-zero because $\exp(-\infty) = 0$. The second reduction gives $s = e^{-2} + e^{-1} + 1 \approx 1.503$. The log of that is $\log(1.503) \approx 0.407$. The final LSE is $2 + 0.407 \approx 2.407$, written to $\texttt{out}[\texttt{row\_idx}]$.</span>

<span style="font-size: 14px;">Compare with the bare math: $\log(e^0 + e^1 + e^2) = \log(1 + e + e^2) \approx \log(11.107) \approx 2.407$. The two values agree, demonstrating that the shift is exact and the masked lane contributes nothing. Without the additive correction at the end (the $+ \texttt{row\_max}$), the kernel would have emitted $0.407$, off by exactly the row max; this is the most common bug to look for in a logsumexp implementation, because the shifted intermediate is itself a well-defined log of a probability-like sum and looks plausible to a casual eye.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to add $\texttt{row\_max}$ back.** Emitting $\log(\sum \exp(\texttt{row} - \texttt{row\_max}))$ alone is off by exactly $\texttt{row\_max}$. The full identity is $m + \log(\sum \exp(x - m))$, and dropping the additive correction is the canonical logsumexp bug because the intermediate value still looks like a finite probability-log.</span>
* <span style="font-size: 14px;">**Loading masked lanes with $0.0$.** $\exp(0) = 1$, so each masked lane contributes $1$ to the sum after the subtract-and-exp, inflating the LSE. The sentinel must be $-\infty$ so that masked lanes lose the max comparison and exponentiate to $0$.</span>
* <span style="font-size: 14px;">**Calling $\texttt{tl.log}$ per lane.** The log must be taken on the single scalar row sum, not lane-wise on the row tile. Per-lane log produces $\log \exp(x_j - m) = x_j - m$, which sums to something entirely different from $\log \sum \exp(\cdot)$.</span>
* <span style="font-size: 14px;">**Hardcoding the row stride.** Writing $\texttt{row\_idx} \cdot N$ assumes $x$ is row-contiguous with no padding. Passing $\texttt{x.stride(0)}$ from the host keeps the kernel correct for transposed or non-contiguous inputs.</span>

---