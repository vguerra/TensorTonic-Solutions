# <span style="font-size: 20px;">GEMV: Matrix Vector Product</span>

<span style="font-size: 14px;">GEMV is matmul stripped of one dimension's reuse. Each output row's dot product touches every element of the vector $x$ once and every element of its own row of $A$ once, with no two output rows sharing any element of $A$. The kernel below uses a **rectangular tile** of shape $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N})$ that one program walks along the $N$ axis in chunks, accumulating into a $\texttt{BLOCK\_M}$-wide fp32 vector. The arithmetic intensity is an order of magnitude lower than tiled matmul because the vector half of the operation provides no reuse; GEMV stays memory-bound for any reasonable block size.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given $A \in \mathbb{R}^{M \times N}$ and $x \in \mathbb{R}^{N}$, the kernel writes $\texttt{out} \in \mathbb{R}^{M}$:</span>

$$
\texttt{out}[i] = \sum_{j=0}^{N-1} A[i, j] \cdot x[j], \quad 0 \le i < M
$$

<span style="font-size: 14px;">Total work is $2 M N$ FLOPs against $4 (M N + N + M)$ bytes of HBM traffic; for large $M$ and $N$ the dominant term is $4 M N$ bytes of $A$. Each row of $A$ participates in exactly one output element, so $A$ has no reuse at all along $M$; the vector $x$ is read once per output row in a naive formulation but can be reused $\texttt{BLOCK\_M}$ times when a row block is processed together.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional with $\lceil M / \texttt{BLOCK\_M} \rceil$ programs. Each **program** owns a slab of $\texttt{BLOCK\_M}$ consecutive output rows, holds a $\texttt{BLOCK\_M}$-wide register accumulator, and walks the $N$ axis of $A$ together with the corresponding chunk of $x$ in $\texttt{BLOCK\_N}$-sized steps. The output is partitioned: no two programs write the same element of $\texttt{out}$, and there is no cross-program reduction along $N$ in this version.</span>

<span style="font-size: 14px;">The parallel pattern is **per-row reduction with row-block grouping**: row reductions are independent, so different output rows could run on different programs, but grouping $\texttt{BLOCK\_M}$ consecutive rows into one program lets the kernel reuse the vector $x$ across all $\texttt{BLOCK\_M}$ rows of the block on every iteration. The choice of $\texttt{BLOCK\_M}$ trades reuse on $x$ against the number of programs available to fill the SMs.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The tile is rectangular: $\texttt{BLOCK\_M}$ rows by $\texttt{BLOCK\_N}$ columns, both declared $\texttt{tl.constexpr}$. A standard choice is $\texttt{BLOCK\_M} = 32$ and $\texttt{BLOCK\_N} = 64$: $\texttt{BLOCK\_M}$ is set to a small multiple of the warp width so the row dimension distributes cleanly across warps, $\texttt{BLOCK\_N}$ is set wider so each iteration brings in enough of $x$ to amortize the load over the $\texttt{BLOCK\_M}$ rows. The accumulator is a single $\texttt{BLOCK\_M}$-wide vector in fp32, materialized as $\texttt{tl.zeros((BLOCK\_M,), dtype=tl.float32)}$ before the inner loop.</span>

<span style="font-size: 14px;">Three masks cover the boundary cases. The row-bound mask $\texttt{m\_mask} = \texttt{offs\_m} < M$ is computed once outside the loop. The column-bound mask $\texttt{n\_mask} = (n\_\texttt{start} + \texttt{offs\_n}) < N$ is recomputed each iteration. The $A$ load combines them as $\texttt{m\_mask}[:, \texttt{None}] \,\&\, \texttt{n\_mask}[\texttt{None}, :]$; the $x$ load uses $\texttt{n\_mask}$ alone; the final store uses $\texttt{m\_mask}$. The $\texttt{other} = 0.0$ literal on both loads turns out-of-range lanes into additive identities so the $\texttt{tl.sum}$ along the column axis does not pick up garbage.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">$A$, $x$, and $\texttt{out}$ live in HBM. Each program loads a $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N})$ tile of $A$ and a $(\texttt{BLOCK\_N},)$ chunk of $x$ per inner iteration; both end up in **registers** for the lifetime of the iteration. The accumulator vector lives in registers across the entire $N$-loop and is materialized to HBM exactly once, at the end, via the masked store of $\texttt{out\_ptr} + \texttt{offs\_m}$. There is no SRAM staging because there is no $\texttt{tl.dot}$ for the compiler to back; the $\texttt{tl.sum}$ reduction over the column axis is a register-level operation.</span>

<span style="font-size: 14px;">Reuse is asymmetric. Inside one iteration the loaded $(\texttt{BLOCK\_N},)$ chunk of $x$ is multiplied against all $\texttt{BLOCK\_M}$ rows of the $A$ tile, so each fp32 of $x$ is used $\texttt{BLOCK\_M}$ times. The $A$ tile, by contrast, has no reuse: every element appears in exactly one $(A[i, j], x[j])$ product. Total HBM traffic per program is $\texttt{BLOCK\_M} \cdot N$ fp32s of $A$ (the row block in full) and $N / \texttt{BLOCK\_N}$ chunks $\cdot \texttt{BLOCK\_N}$ fp32s of $x$, which is $N$ fp32s of $x$ per program. With $\lceil M / \texttt{BLOCK\_M} \rceil$ programs, $x$ is read $\lceil M / \texttt{BLOCK\_M} \rceil$ times in total; the larger $\texttt{BLOCK\_M}$ is, the fewer the redundant reads of $x$.</span>

<span style="font-size: 14px;">L2 reuse on $x$ is the unstated optimization. Programs run concurrently, and the chunks of $x$ that one program reads tend to live in L2 long enough for the next program to find them cached. On a workload where $x$ fits in L2 (a few hundred KB), the effective HBM traffic on $x$ is close to one full read regardless of how many programs are launched; on larger $N$ the L2 hit rate decays and $x$ becomes a real HBM cost.</span>

<span style="font-size: 14px;">A useful framing of the same point: with $\lceil M / \texttt{BLOCK\_M} \rceil$ programs each reading $N$ fp32s of $x$, the **total** worst-case HBM traffic on $x$ alone is $\lceil M / \texttt{BLOCK\_M} \rceil \cdot N \cdot 4$ bytes. At $\texttt{BLOCK\_M} = 32$ on $M = 4096$ this is $128 N$ fp32s of redundant vector reads, an order of magnitude above the optimal single read. The L2 absorbs most of that redundancy in practice; the kernel author has the option, but rarely the need, to enforce a single read by partitioning $x$ across programs and combining with a separate post-processing pass.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per inner iteration, the program reads $\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N} + \texttt{BLOCK\_N}$ fp32s and performs $2 \cdot \texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N}$ FLOPs. Arithmetic intensity is:</span>

$$
I = \frac{2 \cdot \texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N}}{4 (\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N} + \texttt{BLOCK\_N})} = \frac{\texttt{BLOCK\_M}}{2 (\texttt{BLOCK\_M} + 1)} \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">As $\texttt{BLOCK\_M}$ grows, this asymptotes to $0.5$ FLOPs per byte: a sharp cap. Even at $\texttt{BLOCK\_M} = 32$, the intensity is $\frac{32}{66} \approx 0.485$, still well under the typical roofline crossover near $10$ FLOPs/byte. GEMV is firmly **memory-bound** for any practical block shape, dominated by the cost of streaming $A$ from HBM. The kernel runs at HBM bandwidth, not tensor-core throughput.</span>

<span style="font-size: 14px;">Compare with tiled matmul, whose intensity is $\frac{\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N}}{2(\texttt{BLOCK\_M} + \texttt{BLOCK\_N})}$, evaluating to $16$ at $(64, 64)$. The difference is the missing dimension of reuse: matmul reuses both operands across the perpendicular tile axis, GEMV reuses only the vector. The ratio of intensities is approximately $\texttt{BLOCK\_N}$ in favor of matmul. This is the structural reason transformer inference, which is dominated by GEMV-shaped operations at decoding time, is bandwidth-bound while training, which is dominated by matmul-shaped operations, is compute-bound on the same hardware.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">The compiler picks the vector widths for the $A$ tile and the $x$ chunk loads, allocates the registers for the $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N})$ tile and the $\texttt{BLOCK\_M}$-wide accumulator, and lowers $\texttt{tl.sum(..., axis=1)}$ to a warp-level reduction tree of $\log_2(\texttt{BLOCK\_N})$ steps. None of these decisions appear in the kernel source. The $\texttt{tl.sum}$ in particular is one of the cleaner places where the tile model abstracts away an entire CUDA pattern: a hand-written GEMV would issue per-thread partial sums and then a warp-shuffle reduction across lanes; the Triton form is one line.</span>

<span style="font-size: 14px;">The author chooses the grid (1D over row blocks), the tile shape, the accumulator dtype, the broadcast direction on the multiply ($x\_\texttt{chunk}[\texttt{None}, :]$ to broadcast the vector chunk across the $\texttt{BLOCK\_M}$ rows of the $A$ tile), and the mask placement. The author also chooses whether to keep the entire $N$ reduction inside one program (this kernel) or split it across programs and combine with $\texttt{tl.atomic\_add}$ (a SPLIT-K-style variant useful for very long $N$ relative to $M$). The simple form is appropriate when $M$ is large enough that the row-block grid already fills the SMs; the SPLIT variant is appropriate when $M$ is small relative to the device width.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A scalar GEMV that processes one row per program with no inner block makes $N$ scalar loads of $A$ and $N$ scalar loads of $x$ per output element, for $2 M N$ HBM transactions in the worst case. The tiled form with $\texttt{BLOCK\_N} = 64$ collapses each row's $N$ loads into $N / 64$ vector loads, and the $\texttt{BLOCK\_M}$-fold row grouping reduces the number of distinct programs that re-read $x$. The combination drops HBM transactions by roughly $64 \texttt{BLOCK\_M}$, which on a $\texttt{BLOCK\_M} = 32$ kernel is about $2048\times$ fewer transactions, recovering effectively all the bandwidth.</span>

<span style="font-size: 14px;">Beyond the simple tiled form, two further moves are possible. **SPLIT-N** parallelizes across the $N$ reduction by giving each program a sub-chunk of $N$ to reduce and combines the partials via $\texttt{tl.atomic\_add}$ into the length-$M$ output; this trades atomic contention for additional program-level parallelism, useful when $M$ alone does not fill the device. **Persistent kernels** keep one program per SM and have it walk a stream of row blocks in a loop, amortizing launch overhead when $M$ is moderate. Both are situational; the kernel here is the straightforward form that wins on bandwidth alone for typical shapes.</span>

<span style="font-size: 14px;">There is no equivalent of $\texttt{tl.dot}$ for GEMV in mainstream Triton: the multiply-and-reduce inner step is expressed as $\texttt{tl.sum(a\_tile} \cdot x\_\texttt{chunk}[\texttt{None}, :], \texttt{axis=1})$ rather than as a single primitive. The compiler still emits efficient code for this pattern (the multiply lowers to fp32 fused multiply-adds, the sum to a warp-shuffle reduction), but the tensor cores are not exercised because the operation is vector-by-matrix rather than matrix-by-matrix. This is consistent with the memory-bound diagnosis: even if the kernel could route the work through tensor cores, the HBM ceiling on $A$ traffic would bind first.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $M = 5$, $N = 6$, $\texttt{BLOCK\_M} = 4$, $\texttt{BLOCK\_N} = 4$. The launch grid is $\lceil 5/4 \rceil = 2$ programs.</span>

<span style="font-size: 14px;">**Program $0$** owns rows $[0, 1, 2, 3]$. $\texttt{m\_mask} = [\texttt{T}, \texttt{T}, \texttt{T}, \texttt{T}]$. The $N$-loop runs $\lceil 6/4 \rceil = 2$ iterations. Iteration $1$: $n\_\texttt{start} = 0$, $\texttt{n\_mask} = [\texttt{T}, \texttt{T}, \texttt{T}, \texttt{T}]$ for columns $[0, 1, 2, 3]$. Load the $(4, 4)$ tile $A[0{:}4, 0{:}4]$, load the chunk $x[0{:}4]$, multiply with broadcasting, $\texttt{tl.sum}$ along axis $1$ producing four partial sums (one per row), add to $\texttt{acc}$. Iteration $2$: $n\_\texttt{start} = 4$, $\texttt{n\_mask} = [\texttt{T}, \texttt{T}, \texttt{F}, \texttt{F}]$ for columns $[4, 5, 6, 7]$. Load $A[0{:}4, 4{:}6]$ with the rightmost two columns zeroed, load $x[4{:}6]$ with the same mask, multiply, sum, accumulate. After two iterations $\texttt{acc}$ holds $\sum_{j=0}^{5} A[i, j] x[j]$ for $i \in [0, 3]$.</span>

<span style="font-size: 14px;">**Program $1$** owns rows $[4, 5, 6, 7]$. $\texttt{m\_mask} = [\texttt{T}, \texttt{F}, \texttt{F}, \texttt{F}]$, so only the first row of the block contributes a real output. The same two $N$-iterations run with the same column masks, with three of the four $A$ rows zeroed by the row mask. The final store writes $\texttt{out}[4]$ and masks off $\texttt{out}[5..7]$, which do not exist. Counting HBM loads, program $0$ reads $4 \cdot 6 = 24$ valid fp32s of $A$ plus $6$ of $x$; program $1$ reads $1 \cdot 6 = 6$ valid fp32s of $A$ plus $6$ of $x$ (the rest are masked off and the compiler may or may not still issue the transactions for them, depending on the alignment of the masked-off lanes).</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Accumulator dtype too narrow.** A $\texttt{tl.float16}$ accumulator silently loses precision for $N \ge 128$ because the running dot product's exponent outpaces fp16's $11$-bit mantissa. The accumulator must be $\texttt{tl.float32}$ even when $A$ and $x$ are fp16, with the cast to the storage dtype on the final store.</span>

* <span style="font-size: 14px;">**Forgetting $\texttt{other} = 0.0$ on the masked loads.** Without an explicit additive identity for out-of-range lanes, the masked load yields whatever the compiler picks (typically zero, but not guaranteed across versions), and the $\texttt{tl.sum}$ along the column axis can pick up garbage on the $N$-tail iteration. Always pass $\texttt{other} = 0.0$ explicitly on both the $A$ tile load and the $x$ chunk load.</span>

* <span style="font-size: 14px;">**Wrong broadcast direction.** Broadcasting $x\_\texttt{chunk}[:, \texttt{None}]$ instead of $x\_\texttt{chunk}[\texttt{None}, :]$ stretches the vector across the row axis instead of the column axis, producing a completely wrong product. The shape contract is that $x\_\texttt{chunk}$ aligns with the $\texttt{BLOCK\_N}$ columns of the $A$ tile.</span>

* <span style="font-size: 14px;">**Forgetting the row mask on the store.** When $M$ is not a multiple of $\texttt{BLOCK\_M}$, the last program's accumulator covers rows past the end of $\texttt{out}$. Without $\texttt{m\_mask}$ on the final $\texttt{tl.store}$, the kernel writes past the output buffer.</span>

---