# <span style="font-size: 20px;">Online Softmax</span>

<span style="font-size: 14px;">The standard row-wise softmax walks each row twice: once to find the max, once to compute the normalized exponentials. The online formulation does it in a single forward pass over chunks, maintaining a running max $m$ and a running denominator $l$ that are updated together as each chunk is consumed. This kernel isolates the trick that powers FlashAttention's inner softmax block into a standalone problem, so the algebra is studied without the surrounding attention machinery. The parallel pattern is **one program per row with an internal sequential recurrence**, the cleanest setting in which to internalize the running-max identity before applying it inside attention.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an $(M, N)$ input matrix $x$, the kernel writes</span>

$$
\texttt{out}[i, j] = \frac{\exp(x[i, j] - m_i)}{l_i}, \quad m_i = \max_k x[i, k], \quad l_i = \sum_k \exp(x[i, k] - m_i)
$$

<span style="font-size: 14px;">The kernel computes $m_i$ and $l_i$ in a single forward pass over the row in $\texttt{BLOCK\_SIZE}$ chunks, then makes a second pass to write the normalized output. The total HBM traffic per row is one read for the streaming statistics, one read for the normalization, and one write for the output: two reads instead of the three a literal two-pass implementation would need.</span>

---

## <span style="font-size: 16px;">The Online Update</span>

<span style="font-size: 14px;">For a row that has been partly consumed with running state $(m, l)$, the kernel processes the next chunk $c$ as</span>

$$
m_\text{local} = \max_{j \in c} x_j, \quad m_\text{new} = \max(m, m_\text{local})
$$

$$
l_\text{new} = l \cdot \exp(m - m_\text{new}) + \sum_{j \in c} \exp(x_j - m_\text{new})
$$

<span style="font-size: 14px;">The rescale by $\exp(m - m_\text{new})$ is the algebraic identity that makes the recurrence correct. Before the update, $l$ is the sum of $\exp(x_j - m)$ over the chunks already consumed, anchored at the old max $m$. After the update, the same sum needs to be anchored at $m_\text{new}$, and the factor $\exp(m - m_\text{new})$ converts the basis: for any $j$ already in the sum, $\exp(x_j - m) \cdot \exp(m - m_\text{new}) = \exp(x_j - m_\text{new})$. The new chunk's exponentials are then computed directly at the new basis and added in. When $m_\text{new} = m$ (no new max), the rescale is $\exp(0) = 1$ and the update reduces to an ordinary running sum.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional with $M$ **programs**, one per row. Each program reads $\texttt{row} = \texttt{tl.program\_id(0)}$, computes the row's base pointer offset $\texttt{row} \cdot N$, and runs two inner Python loops over the chunks. The first inner loop is the streaming forward pass: load chunk, update $(m, l)$. The second inner loop is the normalization: reload chunk, compute $\exp(x - m) / l$, store. Both loops walk the same $N$ axis in $\texttt{BLOCK\_SIZE}$ strides.</span>

<span style="font-size: 14px;">The running scalars $m$ and $l$ are program-local fp32 values held in registers. No two programs share them; there is no inter-program reduction. Rows are entirely independent, which is what lets the grid scale linearly in $M$ for parallelism. Inside one program, the chunk loop is sequential because each update reads the prior $(m, l)$, but the per-chunk arithmetic (the chunk max, the chunk sum) is a tile-level reduction that the compiler lowers efficiently inside one program.</span>

<span style="font-size: 14px;">A useful contrast: the fused-softmax kernel from earlier in the curriculum holds the entire row as one tile inside a single program, taking a single $\texttt{tl.max}$ and a single $\texttt{tl.sum}$ over the whole tile. That kernel has no chunk loop because everything fits in one tile. The online kernel here trades that simplicity for the ability to handle rows wider than any single tile can hold, which is the structural property that makes the recurrence necessary for FlashAttention.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The single constexpr meta-parameter is $\texttt{BLOCK\_SIZE} = 1024$. The choice matters more here than in vector add because it controls the trade-off between the chunk's register cost and the loop's iteration count. A larger block size means each chunk does more arithmetic per loop iteration and amortizes the loop overhead; a smaller block size keeps the per-program register footprint low and matters when many programs run concurrently on one streaming multiprocessor.</span>

<span style="font-size: 14px;">The mask per chunk is $\texttt{cols} < N$ where $\texttt{cols}$ is the chunk's offset tile. Two non-trivial details. First, the streaming load uses $\texttt{other} = -10^{30}$ rather than the default zero; padded lanes must not pull the chunk max down. Second, the chunk sum applies $\texttt{tl.where}(\texttt{mask}, \exp(x - m_\text{new}), 0)$ explicitly, because $\exp(-10^{30} - m_\text{new})$ is a perfect-zero in fp32 but the safer pattern is to gate the sum directly. The second-pass load uses $\texttt{other} = 0$ because masked lanes are gated out of the store anyway.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Each row of $x$ is read twice from HBM, once per pass over the chunks. The output row is written once. Total HBM traffic per row: $2 \cdot N \cdot 4 + N \cdot 4 = 12 N$ bytes in fp32. The literal two-pass softmax does the same: one read for the max, one read for the sum (each pass is a read), one write for the output. The online algorithm collapses the first two reads into one because the max and the denominator are maintained together, dropping the count from three to two passes for the algorithm view, but with two HBM reads in the kernel implementation because the second pass still needs the row data to compute the normalized exponentials.</span>

<span style="font-size: 14px;">The chunk tiles live in registers; the running scalars $(m, l)$ live in registers across the entire inner loop. There is no SRAM staging because the program holds at most one $\texttt{BLOCK\_SIZE}$-wide chunk at any time, and the compiler keeps it in registers for the lifetime of the loop iteration. The L2 cache may help on the second pass: if the row fits in L2, the reload during normalization hits cache rather than HBM. For typical inference dimensions ($N$ in the thousands, row size in the tens of KB), the row fits comfortably in L2 and the second read is effectively free in bandwidth terms.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output element, the kernel does roughly $4$ FLOPs (one exp on the first pass, one max, one update of $l$, plus the second-pass exp and divide) against $12$ bytes of HBM traffic. Arithmetic intensity is $\approx 0.33$ FLOPs per byte. **Memory-bound**, like every other reduction in the curriculum. The exp instructions are non-trivial in their own right (each costs more than a multiply) but they are still vastly outnumbered by the byte traffic at this intensity.</span>

<span style="font-size: 14px;">What the online formulation buys over the literal two-pass version is not bandwidth (both have the same HBM cost) but **register footprint for arbitrary row widths**. The fused-softmax kernel that holds the whole row in one tile needs $\texttt{BLOCK\_SIZE} \ge N$, which fails for rows larger than the maximum tile size the compiler supports. The online kernel handles arbitrary $N$ with a fixed chunk size, which is what makes it the right building block for FlashAttention, where the relevant "row" is a row of the $N \times N$ score matrix and $N$ can be in the tens of thousands.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author handles:** the row-parallel decomposition, the constexpr chunk size, the two-loop structure (forward streaming + second-pass normalization), the order of the rescale-then-add update on $l$, the initial values $m = -10^{30}$ and $l = 0$, the padded-lane $-\infty$ on the streaming load, and the explicit $\texttt{tl.where}$ on the chunk sum. The whole algebraic identity behind the running-max update is a kernel-design choice that the compiler does not see.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering the chunk load to wide PTX vector instructions, allocating registers for the chunk tile and the running scalars, scheduling the in-tile reductions (the $\texttt{tl.max}$ and $\texttt{tl.sum}$ collapses), and unrolling the chunk loop where it can. The compiler also picks $\texttt{num\_warps}$ internally to shard the chunk across enough warps for latency hiding on the loads. The author never names a warp count and never declares scratchpad memory; the in-tile reductions are lowered to shuffle instructions by the compiler.</span>

<span style="font-size: 14px;">The fp32 promotion of the running scalars is a quiet but consequential choice. Even when the input row is fp16 or bf16, $m$ and $l$ should be fp32 because the running sum may aggregate thousands of exponentials and the precision loss in lower-precision accumulation compounds quickly. The kernel here uses fp32 inputs so the question is moot, but the rule transfers directly to mixed-precision variants of the kernel and to the same running scalars inside FlashAttention.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The literal two-pass alternative does the same HBM work but with a different inner structure: pass 1 computes only $m$ (no $l$), pass 2 computes $l$ from the now-known $m$, pass 3 writes the normalized output. Three passes, three HBM reads of the row, slightly more register traffic. The online formulation collapses passes 1 and 2 into a single forward pass by maintaining $(m, l)$ jointly, saving one read.</span>

<span style="font-size: 14px;">The fused-softmax alternative (the textbook "load the whole row in one tile") is faster than the online version when the row fits in a single $\texttt{tl.constexpr}$ tile: it reads each row once, holds it whole, takes the max, subtracts, exponentiates, sums, divides, and writes back. One HBM read instead of two. The online formulation costs an extra row read in exchange for handling rows of unbounded width. For $N$ small enough to fit in one tile (a few thousand), the fused kernel is the right choice; for the rows that appear inside FlashAttention (where $N$ is the sequence length and can be tens of thousands), the online formulation is the only option.</span>

<span style="font-size: 14px;">Further optimizations on top of the online kernel: autotune $\texttt{BLOCK\_SIZE}$ over the active $N$ to balance register footprint against loop overhead, store $m$ and $l$ in a small per-row scratch so the second pass can skip the recomputation of $m$ (already done in the online formulation; the second pass uses the final $m$ directly), and software-pipeline the second pass via $\texttt{num\_stages}$ so the next chunk's load overlaps with the current chunk's arithmetic.</span>

<span style="font-size: 14px;">The reason this kernel earns a section in the curriculum despite being slower than fused softmax on rows that fit is that it is the standalone version of the identity that FlashAttention applies inside attention. Reading the recurrence here in isolation makes the FlashAttention inner loop legible: when FlashAttention's inner $K, V$ column-block iteration updates $(m_i, l_i, o_i)$, it is running exactly the update derived here, with the output accumulator $o_i$ rescaled by the same $\exp(m - m_\text{new})$ factor that rescales $l_i$ here. Once the standalone form is understood, attention's inner loop is the same identity layered on top of a matrix-vector product.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take a row $x = [1.0, 2.0, 0.5, 1.5]$ with $\texttt{BLOCK\_SIZE} = 2$, so the streaming pass has two chunks $[1.0, 2.0]$ and $[0.5, 1.5]$. Initial state $(m, l) = (-10^{30}, 0)$.</span>

<span style="font-size: 14px;">**Chunk 0:** local max is $2.0$, so $m_\text{new} = 2.0$ and the rescale factor is $\exp(-10^{30} - 2.0) \approx 0$. The new $l$ is $0 \cdot 0 + \exp(-1.0) + \exp(0.0) \approx 0.368 + 1.000 = 1.368$. State after: $(2.0, 1.368)$.</span>

<span style="font-size: 14px;">**Chunk 1:** local max is $1.5$, less than the running $2.0$, so $m_\text{new} = 2.0$ and the rescale factor is $\exp(2.0 - 2.0) = 1$. The new $l$ is $1.368 \cdot 1 + \exp(-1.5) + \exp(-0.5) \approx 1.368 + 0.223 + 0.607 = 2.198$. State after: $(2.0, 2.198)$.</span>

<span style="font-size: 14px;">**Second pass:** with $m = 2.0$ and $l = 2.198$, the kernel writes $\texttt{out}[j] = \exp(x_j - 2.0) / 2.198$ for each $j$. The result is identical (modulo float32 drift) to the two-pass softmax that would have computed $m = 2.0$ first, then $l = \exp(-1) + \exp(0) + \exp(-1.5) + \exp(-0.5) = 2.198$, then the same division. The intermediate states of the online formulation matched the running prefix sums of the two-pass formulation at every chunk boundary.</span>

<span style="font-size: 14px;">If chunk $1$ had instead been $[3.0, 0.0]$, the running max would have jumped to $3.0$ and the rescale factor would be $\exp(2.0 - 3.0) \approx 0.368$. The old $l = 1.368$ becomes $1.368 \cdot 0.368 \approx 0.503$ before the new chunk's exponentials $\exp(0) + \exp(-3) \approx 1.050$ are added in, yielding $l \approx 1.553$. That single rescale of the prior partial sum is the entire correctness argument for the recurrence.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Padded lanes loaded as zero on the streaming pass.** Without $\texttt{other} = -10^{30}$ on the streaming-load mask, padded lanes read as zero and the chunk max for a row with all-negative values clamps to zero, which then propagates as a spurious max into the running state.</span>
* <span style="font-size: 14px;">**Update order: rescale before add.** Multiply $l$ by $\exp(m - m_\text{new})$ first, then add the new chunk's exponentials. Reversing the order leaves the new chunk's contribution at the wrong reference and the result drifts by a factor of $\exp(m - m_\text{new})$.</span>
* <span style="font-size: 14px;">**Initial max of zero.** Start $m$ at a large negative value such as $-10^{30}$, not $0$. A row of large negative values otherwise normalizes against an artificial zero max and produces a tiny numerator over a near-zero denominator.</span>
* <span style="font-size: 14px;">**Skipping the second pass.** The first pass produces only the running statistics $(m, l)$, not the output. Writing the unnormalized exponentials from the first pass yields a function that sums to $l$ per row instead of to $1$, and the subsequent code that consumes the softmax (cross-entropy, attention) silently misbehaves.</span>

---