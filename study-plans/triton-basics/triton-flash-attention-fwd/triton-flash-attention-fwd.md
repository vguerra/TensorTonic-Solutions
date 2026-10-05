# <span style="font-size: 20px;">FlashAttention Forward</span>

<span style="font-size: 14px;">FlashAttention is the hot path of modern transformer inference, fused into a single Triton kernel. The kernel computes $\mathrm{softmax}(QK^\top / \sqrt{D}) V$ without ever writing the $N \times N$ score matrix to HBM. The parallel pattern is the most architecturally ambitious in the curriculum: a **tile loop with an online recurrence**, where each program owns one row-block of $Q$, streams across $K, V$ in column-blocks, and maintains a per-row running max and denominator in registers so that the softmax is normalized in one pass rather than two. Dao et al. introduced this fusion in 2022; everything downstream in LLM inference is built on it.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given $Q, K, V \in \mathbb{R}^{N \times D}$, the kernel writes</span>

$$
\texttt{out}[i, :] = \sum_{j=0}^{N-1} \frac{\exp(s_{ij} - \max_k s_{ik})}{\sum_{k} \exp(s_{ik} - \max_k s_{ik})} \cdot V[j, :], \quad s_{ij} = \frac{Q[i, :] \cdot K[j, :]}{\sqrt{D}}
$$

<span style="font-size: 14px;">This is the same closed form as the naive three-stage computation; the value of FlashAttention is purely in how the kernel reorganizes the work. Single-head, no batch, no causal mask: the cleanest possible setting for the fusion.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional: $\lceil N / \texttt{BLOCK\_M} \rceil$ **programs**, each owning one contiguous row-block of $Q$ and the corresponding row-block of $\texttt{out}$. Program $p$ reads its index from $\texttt{tl.program\_id(0)}$ and computes row offsets $\texttt{offs\_m} = p \cdot \texttt{BLOCK\_M} + \texttt{tl.arange}(0, \texttt{BLOCK\_M})$. The whole row-block of $Q$ is loaded once at the top of the kernel and pinned in registers for the entire lifetime of the program. The program then walks $K$ and $V$ in $\texttt{BLOCK\_N}$ column-blocks inside an inner Python loop, accumulating into a `(BLOCK_M, BLOCK_DMODEL)` output tile.</span>

<span style="font-size: 14px;">No two programs touch the same output row. There is no cross-program reduction, no atomic, no scratch buffer. The independence is at the $Q$ row-block boundary, which is exactly why the kernel scales linearly in $N$ for parallelism. Inside one program, the work is serial across the column dimension because the running max and denominator are inherently sequential statistics that require the prior chunk's state before consuming the next one.</span>

<span style="font-size: 14px;">The decomposition is asymmetric in a useful way: the parallel axis is $Q$, the sequential axis is $K, V$. This matches the bandwidth profile. $Q$ is read once per program and stays in registers, so spreading it across programs maximizes parallelism without re-reading. $K, V$ are each read once per program too, but the inner loop walks them sequentially so the running state can fold them into the softmax recurrence. Choosing the parallelism the other way (parallel over $K, V$ blocks, sequential over $Q$ rows) would require a cross-program combine of the partial sums, which would either need atomics or a second kernel; the chosen layout avoids both.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">Three $\texttt{tl.constexpr}$ block sizes drive the kernel: $\texttt{BLOCK\_M} = 16$ for the $Q$ row-block, $\texttt{BLOCK\_N} = 16$ for the $K, V$ column-block, and $\texttt{BLOCK\_DMODEL}$ rounded up to the next power of two above $D$. The first two are tunable in principle; for the canonical implementation they are chosen small because the program already holds three tiles in registers ($Q$, the score row, and the running output accumulator), and pushing them larger blows the register budget for typical $D$.</span>

<span style="font-size: 14px;">Three masks are in play. The $Q$ row mask $\texttt{offs\_m} < N$ gates the initial $Q$ load and the final store. The $K, V$ row mask $\texttt{cur\_n} < N$ gates the column-block loads. The head-dim mask $\texttt{offs\_d} < D$ gates the trailing fp32 lanes when $D$ is not a power of two. The two row masks intersect with the head-dim mask to form a 2D boolean for $\texttt{tl.load}$. The non-trivial mask is on the score tile: padded $K$ rows must be forced to $-\infty$ in the score, not zero, with $\texttt{tl.where}(\texttt{cur\_n} < N, s, -\infty)$. Zero scores become weight 1 after $\exp$ and would silently bias the output toward the padded entries.</span>

---

## <span style="font-size: 16px;">The Online Recurrence</span>

<span style="font-size: 14px;">For one $Q$ row, the running state is a triple $(m_i, l_i, o_i)$: the running max of the scores seen so far, the running softmax denominator, and the running output accumulator. Initialized to $(-\infty, 0, 0)$. For each new $K, V$ column-block, the kernel computes the local score $s$, takes its block max $m_\text{local}$, and updates with the identity</span>

$$
m_\text{new} = \max(m_i, m_\text{local}), \quad \alpha = \exp(m_i - m_\text{new})
$$

$$
l_i \leftarrow \alpha \cdot l_i + \sum_{j \in \text{block}} \exp(s_j - m_\text{new})
$$

$$
o_i \leftarrow \alpha \cdot o_i + \sum_{j \in \text{block}} \exp(s_j - m_\text{new}) \cdot V_j
$$

<span style="font-size: 14px;">The rescale $\alpha$ is the algebraic identity that makes this correct: it converts the old denominator and the old partial output, both normalized against $m_i$, to be normalized against the new $m_\text{new}$. After the last column-block, dividing $o_i / l_i$ yields the exact softmax-weighted output. The recurrence is mathematically equivalent to the two-pass formulation; what it buys is that the score matrix is never written to HBM.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">The $Q$ row-block of shape `(BLOCK_M, D)` is loaded from HBM exactly once per program. The $K, V$ tiles are loaded once per column-block; for a single program walking the whole column axis they are touched $\lceil N / \texttt{BLOCK\_N} \rceil$ times. The score tile, the softmax probabilities, the running max, the running denominator, and the running output accumulator all live in registers across the inner loop. Nothing intermediate ever reaches HBM. This is the single most consequential property of the kernel.</span>

<span style="font-size: 14px;">Quantify the bandwidth saving. For $N = 4096$ the naive three-stage attention writes a $4096 \times 4096$ fp32 score matrix to HBM (64 MB), then reads it back for the softmax (another 64 MB read and 64 MB write), then reads it again for the $\mathrm{attn} \cdot V$ step. That is roughly $4 \cdot 4 N^2$ bytes of score traffic on top of the $Q, K, V, \texttt{out}$ traffic. FlashAttention reads $Q, K, V$ once each ($3 \cdot 4 N D$ bytes) and writes $\texttt{out}$ once ($4 N D$ bytes), with zero score traffic. For $D = 64, N = 4096$ that is $\approx 4$ MB of input/output traffic against $\approx 256$ MB of score traffic that disappears. The ratio improves quadratically with $N$.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Naive attention is firmly **memory-bound**: the $N^2$ writes and reads dominate everything else. FlashAttention's whole point is to move the kernel onto the compute-bound side of the roofline by eliminating that bandwidth. Per program, the kernel does $O(N \cdot D)$ FLOPs for the $QK^\top$ scores, $O(N)$ FLOPs for the softmax update, and $O(N \cdot D)$ FLOPs for the accumulator, all on the $\texttt{BLOCK\_M}$ rows. Total FLOPs per program scale as $\Theta(\texttt{BLOCK\_M} \cdot N \cdot D)$ against HBM bytes that scale as $\Theta(D \cdot (\texttt{BLOCK\_M} + N))$. The arithmetic intensity is roughly $\Theta(N)$ FLOPs per byte for the dominant inner-loop traffic, well above the roofline crossover.</span>

<span style="font-size: 14px;">In practice the canonical reference uses elementwise multiply-and-sum to implement the tile-by-tile matmul rather than $\texttt{tl.dot}$, because the tile sizes ($16 \times 16$ scores, $16 \times \texttt{BLOCK\_DMODEL}$ output) and the on-the-fly rescaling do not map cleanly to tensor-core MMA shapes in this minimal form. Production FlashAttention kernels reach for $\texttt{tl.dot}$ on larger tiles to pull tensor cores into the arithmetic, which is the next-step optimization once the algorithmic fusion is correct.</span>

<span style="font-size: 14px;">The roofline placement also flips with sequence length. For very short sequences (a few hundred tokens), the $Q$ and $K, V$ loads dominate and the kernel sits near the memory-bound side. For long sequences (tens of thousands), the inner loop runs many times per program with a fixed $Q$ tile resident in registers and the per-load arithmetic ratio climbs toward the compute ceiling. The structural advantage over naive attention is the same in both regimes (no $N^2$ HBM scratch), but the practical wall-clock win grows with $N$ because the eliminated traffic grows quadratically while the retained traffic grows linearly.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author handles:** the row-block decomposition, the constexpr block sizes, the inner column-block loop, the order of the rescale-then-accumulate updates, the masking of padded $K$ rows to $-\infty$, the initialization of the running max to a large negative number, and the final divide by the denominator. The whole online-softmax dance is a kernel-design choice the compiler cannot infer.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering the tile loads and stores to wide vector PTX, allocating registers for $Q$, the score tile, and the accumulator, scheduling the broadcast-multiply-sum that implements the inner matmul, and emitting fp32 FMA instructions for the rescale-and-accumulate updates. The compiler also picks how the `(BLOCK_M, BLOCK_DMODEL)` accumulator is sharded internally for ILP. The author never names a warp, never declares scratchpad memory, never inserts a barrier; the compiler infers what is needed from the tile algebra.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive baseline is three separate kernels: a tiled $QK^\top$ matmul that writes the $N \times N$ score, a row-wise softmax over the score, and a second tiled matmul against $V$. Three kernel launches, three HBM round-trips through the $N^2$ buffer. FlashAttention collapses the entire pipeline into one launch, one round-trip per operand, and no $N^2$ tensor. For long-context inference ($N$ in the tens of thousands) the naive kernel runs out of memory before it runs out of time.</span>

<span style="font-size: 14px;">Further optimizations on top of the canonical kernel: replace the elementwise multiply-and-sum with $\texttt{tl.dot}$ to engage tensor cores; autotune $\texttt{BLOCK\_M}, \texttt{BLOCK\_N}$ and $\texttt{num\_warps}$ over the input shape; pipeline the next $K, V$ load with the current accumulate via $\texttt{num\_stages} > 1$; add causal masking inline so a separate pass is unnecessary. Each of these is incremental on top of the algorithmic fusion, which is itself the largest single speedup in the modern transformer stack.</span>

<span style="font-size: 14px;">The FlashAttention-2 refinement reorders the inner-loop arithmetic so that the rescale of $o_i$ is deferred to the final divide rather than applied at every column-block. Mathematically the two are equivalent up to floating-point order; the deferred form lets the inner loop be a tighter accumulate without a multiply-by-$\alpha$ at every step. FlashAttention-3 layers further parallelism across $Q$ row-blocks and uses an asynchronous warp-specialized schedule on Hopper-class hardware. The point for this problem is that the canonical online recurrence here is the floor of that staircase: once the recurrence is correct, everything else is a schedule transformation that preserves the same algebraic identity.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take one query row and $N = 4$ keys, with $\texttt{BLOCK\_N} = 2$. The scaled scores are $s = [1.0, 2.0, 0.5, 1.5]$ across the row, processed in two column-blocks.</span>

<span style="font-size: 14px;">**Block 0** holds $[1.0, 2.0]$. Initial $(m, l, o) = (-\infty, 0, 0)$. Block max is $2.0$, so $m_\text{new} = 2.0$ and $\alpha = \exp(-\infty - 2.0) = 0$. The denominator becomes $0 \cdot 0 + \exp(-1.0) + \exp(0.0) \approx 0.368 + 1.0 = 1.368$. The output accumulator becomes $0 \cdot 0 + \exp(-1.0) \cdot V_0 + \exp(0.0) \cdot V_1$. State: $(2.0, 1.368, o)$.</span>

<span style="font-size: 14px;">**Block 1** holds $[0.5, 1.5]$. Block max is $1.5$, still below $2.0$, so $m_\text{new} = 2.0$ and $\alpha = \exp(2.0 - 2.0) = 1$. The denominator becomes $1.368 + \exp(-1.5) + \exp(-0.5) \approx 1.368 + 0.223 + 0.607 = 2.198$. The output accumulator receives $\exp(-1.5) V_2 + \exp(-0.5) V_3$ added at scale $1$. Final divide by $l = 2.198$ produces exactly the softmax-weighted output, identical to a two-pass implementation modulo float32 drift.</span>

<span style="font-size: 14px;">If Block 1 had introduced a new max, say $m_\text{local} = 3.0$, then $\alpha = \exp(2.0 - 3.0) \approx 0.368$ would multiply both the running denominator and the running output before adding the new contributions, re-anchoring them to the new max. That single rescale is the whole online trick.</span>

<span style="font-size: 14px;">Compare the work against a two-pass version on the same row. The two-pass version would walk the row once to compute $m = 2.0$, walk it again to compute $l = \exp(-1) + \exp(0) + \exp(-1.5) + \exp(-0.5) \approx 2.198$, then walk a third time to write the normalized output. The online version walks the row once for $(m, l)$ and once for the final divide. Inside FlashAttention, the row is a row of the $N \times N$ score matrix, and the cost of "walking the row" is the cost of reading the corresponding $K, V$ column-blocks: cutting it from three passes to two doubles down on the bandwidth saving the algorithmic fusion already provides.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Padded $K$ columns left at score zero.** Zero scores become weight $1$ after $\exp$ and bias the output toward whatever $V$ rows are in the padded slots. Use $\texttt{tl.where}(\texttt{cur\_n} < N, s, -10^{30})$ on the score tile before the max and the exp.</span>
* <span style="font-size: 14px;">**Updating $o$ and $l$ before the rescale by $\alpha$.** The rescale converts the old state from the old max to the new one. Adding the new block's contribution first leaves it un-anchored to the wrong reference, and the bug is silent until the running max actually changes.</span>
* <span style="font-size: 14px;">**Applying $1/\sqrt{D}$ after the softmax.** Softmax is not linear, so post-scaling is mathematically wrong. The $\sqrt{D}$ divide goes inside the score tile before the max-shifted exp.</span>
* <span style="font-size: 14px;">**Storing the score tile to HBM.** Any code path that writes the `(BLOCK_M, BLOCK_N)` score back to global memory defeats the kernel and turns it into the naive three-stage formulation. The score lives in registers from creation to consumption.</span>

---