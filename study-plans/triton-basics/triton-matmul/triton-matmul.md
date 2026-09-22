# <span style="font-size: 20px;">Tiled Matrix Multiplication</span>

<span style="font-size: 14px;">Every loaded element of $A$ in a naive matmul is read $N$ times across the output rows and every element of $B$ is read $M$ times across the output columns. The kernel below is the canonical Triton answer to that arithmetic: a **2D tile** owned by one program, a $K$-loop that accumulates into a register tile, and a single $\texttt{tl.dot}$ call per inner iteration that lowers to a tensor-core matmul. The reuse factor is what turns matrix multiplication from a memory-bound operation into a compute-bound one, and the tile is the unit that materializes that reuse.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, the kernel writes $C \in \mathbb{R}^{M \times N}$:</span>

$$
C[i, j] = \sum_{k=0}^{K-1} A[i, k] \cdot B[k, j]
$$

<span style="font-size: 14px;">All three tensors are row-major fp32, the launcher allocates $C$, and the kernel writes into it in place. The total work is $2 M N K$ FLOPs against $4 (M K + K N + M N)$ bytes of HBM traffic at the best case (each element of $A$ and $B$ loaded once and each element of $C$ stored once).</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is two-dimensional, with $\lceil M / \texttt{BLOCK\_M} \rceil$ programs on axis $0$ and $\lceil N / \texttt{BLOCK\_N} \rceil$ on axis $1$. Each **program** is identified by the pair $(\texttt{pid\_m}, \texttt{pid\_n})$ read from $\texttt{tl.program\_id(0)}$ and $\texttt{tl.program\_id(1)}$, and owns exactly one $\texttt{BLOCK\_M} \times \texttt{BLOCK\_N}$ output tile of $C$. The set of owned output tiles partitions the matrix: no two programs write the same element of $C$, and there is no cross-program reduction along $K$ for this version. Output independence makes the kernel embarrassingly parallel at the tile level, even though each tile contains a $K$-long inner reduction the single program must perform sequentially.</span>

<span style="font-size: 14px;">The 2D grid mirrors the 2D structure of the output. The parallel pattern is **per-tile reduction**: each program performs a private dot product over $K$ for its tile and stores the result; the only synchronization in the entire kernel is the implicit one between programs in the launcher, which simply waits for all of them to finish before returning control to the host. The compiler will dispatch the programs onto SMs in whatever schedule the runtime sees fit, so the kernel must be correct for any ordering of $(\texttt{pid\_m}, \texttt{pid\_n})$ pairs.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The tile dimensions $\texttt{BLOCK\_M}$, $\texttt{BLOCK\_N}$, and $\texttt{BLOCK\_K}$ are declared $\texttt{tl.constexpr}$, fixed at compile time so the compiler can size registers, unroll the $K$-loop body, and select tensor-core MMA instructions of the right shape. Common starting values are $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$ and $\texttt{BLOCK\_K} = 32$: powers of two so the compiler can vectorize cleanly, large enough that each $\texttt{tl.dot}$ keeps the tensor cores busy, small enough that the per-program register and shared-memory footprint stays inside what the SM exposes.</span>

<span style="font-size: 14px;">Three runtime dimensions can overshoot their tile-aligned bounds, so the kernel masks all three. Inside the $K$-loop, $\texttt{k\_mask} = (k + \texttt{offs\_k}) < K$ disables out-of-range columns of the loaded $A$ slab and out-of-range rows of the loaded $B$ slab, with $\texttt{other} = 0.0$ so masked lanes contribute additive identities to the accumulator. The final store guards against the $M$ and $N$ tails with $(\texttt{offs\_m}[:, \texttt{None}] < M) \,\&\, (\texttt{offs\_n}[\texttt{None}, :] < N)$. Without the $K$ mask the kernel reads past the end of $A$ or $B$ on non-aligned $K$; without the $M$ or $N$ store mask it writes past the end of $C$. Masks here are a correctness obligation, not a performance lever.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Operand tiles begin in HBM. The kernel issues one $\texttt{tl.load}$ for the $(\texttt{BLOCK\_M}, \texttt{BLOCK\_K})$ slab of $A$ and one for the $(\texttt{BLOCK\_K}, \texttt{BLOCK\_N})$ slab of $B$ each $K$-loop iteration; the compiler stages those slabs into on-chip **SRAM** (the on-chip scratchpad the compiler manages for $\texttt{tl.dot}$) and feeds them into the tensor-core MMA instruction. The accumulator $\texttt{acc}$ lives in **registers** for the entire $K$-loop and is materialized to HBM only by the final masked store.</span>

<span style="font-size: 14px;">The reuse factor is the whole point. Inside one program's $K$-loop iteration, the loaded $(\texttt{BLOCK\_M}, \texttt{BLOCK\_K})$ slab of $A$ contributes to every one of the $\texttt{BLOCK\_N}$ output columns: each fp32 of $A$ is used $\texttt{BLOCK\_N}$ times. Symmetrically, the loaded $(\texttt{BLOCK\_K}, \texttt{BLOCK\_N})$ slab of $B$ contributes to every one of the $\texttt{BLOCK\_M}$ output rows: each fp32 of $B$ is used $\texttt{BLOCK\_M}$ times. A naive triple-loop matmul that touches HBM for every multiply would read $A$ and $B$ once per output element, giving zero reuse; tiling transforms the same operation into a memory pattern where every loaded byte performs many FLOPs before being discarded.</span>

<span style="font-size: 14px;">L2 reuse exists across programs. Adjacent programs along $\texttt{pid\_n}$ load the same slab of $A$ in their respective $K$-loops; if their lifetimes overlap on the same SM cluster, the L2 will serve the second program's $A$ slab without going back to HBM. The naive row-major program-ID schedule exploits this poorly; the standard fix is a **grouped program-ID remap** that interleaves $\texttt{pid\_m}$ and $\texttt{pid\_n}$ to keep adjacent programs working on overlapping operands. This is a separate optimization and not what this kernel does.</span>

<span style="font-size: 14px;">A second compiler-driven memory mechanism that this kernel benefits from implicitly is **software pipelining** of the $K$-loop, controlled by $\texttt{num\_stages}$. With $\texttt{num\_stages} = 2$, the compiler issues the load for the next iteration's $A$ and $B$ slabs while the current iteration's $\texttt{tl.dot}$ is in flight, double-buffering the SRAM staging area so the tensor cores never stall waiting on HBM. The cost is a doubling of SRAM footprint; the benefit is that the latency of operand loads is hidden behind the latency of MMA execution whenever the operand bandwidth and MMA throughput are reasonably balanced.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">For one $K$-loop iteration, the program loads $\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_K} + \texttt{BLOCK\_K} \cdot \texttt{BLOCK\_N}$ fp32 values and performs $2 \cdot \texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N} \cdot \texttt{BLOCK\_K}$ FLOPs. The per-iteration arithmetic intensity is therefore</span>

$$
I = \frac{2 \cdot \texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N} \cdot \texttt{BLOCK\_K}}{4 (\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_K} + \texttt{BLOCK\_K} \cdot \texttt{BLOCK\_N})} = \frac{\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N}}{2 (\texttt{BLOCK\_M} + \texttt{BLOCK\_N})} \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">For $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$ this evaluates to $16$ FLOPs/byte, well past the typical roofline crossover (around $10$ FLOPs/byte on modern accelerators with $\sim 1$ TB/s HBM and tens of TFLOPs of fp32). The kernel is **compute-bound** with sensible block sizes. The intensity does not depend on $\texttt{BLOCK\_K}$ in this form because the formula assumes one $K$ iteration; across the full $K$-loop, larger $\texttt{BLOCK\_K}$ amortizes more arithmetic over each operand load, raising the effective reuse and pushing further past the roofline.</span>

<span style="font-size: 14px;">Compare with GEMV, which has no $N$-axis reuse on $x$ and only $\texttt{BLOCK\_M}$-fold reuse on each row chunk: intensity is below one FLOP/byte and the kernel sits firmly on the memory side of the roofline. Compare with vector add at $\approx 0.08$ FLOPs/byte, two orders of magnitude lower. Matmul is the operation where the tile model finally produces enough arithmetic per loaded byte to exit the memory-bound regime.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">$\texttt{tl.dot}$ is the most consequential single instruction in Triton, and almost everything about its execution is handled by the compiler. The author writes $\texttt{acc} \mathrel{+}= \texttt{tl.dot}(a, b)$ on a $(\texttt{BLOCK\_M}, \texttt{BLOCK\_K})$ and $(\texttt{BLOCK\_K}, \texttt{BLOCK\_N})$ pair; the compiler lowers it to a sequence of **tensor-core MMA** instructions of the right shape for the target architecture, allocates the SRAM staging buffers for the two operands, inserts the synchronization needed to coordinate the MMA-issuing warps, and swizzles the operand layout so the bank-conflict pattern that would otherwise serialize SRAM access is avoided. None of that is named in the kernel source.</span>

<span style="font-size: 14px;">The author chooses the decomposition: grid shape, tile dimensions, the accumulator dtype ($\texttt{tl.float32}$ even when inputs are fp16 or bf16), the mask placement, and the $K$-loop step ($\texttt{BLOCK\_K}$). The author also chooses whether to parallelize across the $K$ dimension (a split-K variant that introduces $\texttt{tl.atomic\_add}$ for the cross-program combine) or keep $K$ private to each program. The compiler picks $\texttt{num\_warps}$ and $\texttt{num\_stages}$ at sensible defaults when they are not specified; the author overrides them through autotune when the defaults are wrong for a specific shape.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The kernel above is already the standard tiled form, and against a per-element implementation it improves HBM traffic by a factor that scales with the smaller of $\texttt{BLOCK\_M}$ and $\texttt{BLOCK\_N}$. A per-element matmul would issue $M N K$ pairs of fp32 loads from HBM ($2 M N K$ loads total) and $M N$ stores; the tiled version issues $\lceil M / \texttt{BLOCK\_M} \rceil \cdot \lceil N / \texttt{BLOCK\_N} \rceil \cdot \lceil K / \texttt{BLOCK\_K} \rceil$ tile-pair loads, each carrying $\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_K} + \texttt{BLOCK\_K} \cdot \texttt{BLOCK\_N}$ elements, with the same $M N$ stores at the end. The reduction in HBM traffic is roughly a factor of $\min(\texttt{BLOCK\_M}, \texttt{BLOCK\_N})$, in line with the reuse factors quoted earlier.</span>

<span style="font-size: 14px;">Layered on top, three further optimizations are common in production matmul kernels: **autotune** over the tile shape and pipeline depth (the next problem in this section), **grouped program-ID remap** to exploit L2 reuse between adjacent programs (often called L2-friendly schedule or super-grouping), and $\texttt{tl.make\_block\_ptr}$ with $\texttt{tl.advance}$ to express the operand stride math symbolically once and let the compiler emit cleaner pointer arithmetic for the $K$-loop. Each adds a few percent to a couple of times speedup depending on shape; none change the asymptotic roofline placement, which is already compute-bound.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $M = N = 4$, $K = 8$, $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = \texttt{BLOCK\_K} = 2$. The launch grid is $2 \times 2 = 4$ programs, each owning one of the four $2 \times 2$ output tiles. Consider the program at $(\texttt{pid\_m}, \texttt{pid\_n}) = (0, 0)$, which owns $C[0{:}2, 0{:}2]$.</span>

<span style="font-size: 14px;">The $K$-loop runs $K / \texttt{BLOCK\_K} = 4$ iterations, with $k = 0, 2, 4, 6$. Each iteration loads a $(2, 2)$ slab of $A$ at rows $[0, 1]$ and columns $[k, k+1]$, a $(2, 2)$ slab of $B$ at rows $[k, k+1]$ and columns $[0, 1]$, calls $\texttt{tl.dot}$ producing a $(2, 2)$ partial product, and accumulates into $\texttt{acc}$. After four iterations, $\texttt{acc}$ holds $\sum_{k=0}^{7} A[i, k] \cdot B[k, j]$ for $i, j \in \{0, 1\}$, which is exactly the desired output tile.</span>

<span style="font-size: 14px;">Counting HBM loads: each iteration reads $4 + 4 = 8$ fp32 values, so the program reads $32$ fp32s of operands across the four iterations and writes $4$ fp32s of output. A per-element implementation of the same tile would read $A[i, k]$ and $B[k, j]$ for each $(i, j, k)$ triple in the tile, $2 \cdot 4 \cdot 8 = 64$ fp32 loads, double the cost. The reuse factor of $\min(\texttt{BLOCK\_M}, \texttt{BLOCK\_N}) = 2$ matches the $64 / 32 = 2$ ratio observed here. Scaling the same tile up to $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$ widens the gap: the per-element variant of the same output tile would read $2 \cdot 64 \cdot 64 \cdot \texttt{BLOCK\_K}$ fp32s while the tiled version reads only $(64 + 64) \cdot \texttt{BLOCK\_K}$, a $32\times$ traffic reduction.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Accumulator dtype too narrow.** Declaring $\texttt{acc}$ as $\texttt{tl.float16}$ silently loses precision once $K$ grows past a few dozen, because the running sum's exponent outpaces the fp16 mantissa's $11$-bit resolution. The accumulator must be $\texttt{tl.float32}$ even when the inputs and outputs are lower precision, and the cast to the storage dtype happens only on the final store.</span>

* <span style="font-size: 14px;">**Missing the $K$ mask.** When $K$ is not a multiple of $\texttt{BLOCK\_K}$, the final $K$-loop iteration overshoots the operand buffers. Without $\texttt{k\_mask} = (k + \texttt{offs\_k}) < K$ on both the $A$ load and the $B$ load, the kernel pulls garbage from past the end of either operand and adds it to the accumulator, breaking every non-aligned shape.</span>

* <span style="font-size: 14px;">**Wrong stride order on pointer construction.** Building $A$ pointers with $\texttt{offs\_m}$ on $\texttt{stride\_ak}$ and $\texttt{offs\_k}$ on $\texttt{stride\_am}$ silently produces $A^{\top} B$ instead of $A B$. The kernel compiles, runs, and may even pass square tests with random inputs that happen to be symmetric. The rule is to pull $\texttt{A.stride(0), A.stride(1)}$ from PyTorch and pair them with the row and column index expressions in that order.</span>

* <span style="font-size: 14px;">**Block sizes not declared $\texttt{tl.constexpr}$.** A runtime block size forces the compiler to emit a generic loop rather than the unrolled tensor-core MMA sequence. The kernel still produces correct results but loses essentially all the benefit of the tile model, often running an order of magnitude slower.</span>

---