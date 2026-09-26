# <span style="font-size: 20px;">Fused Matmul + Bias + ReLU</span>

<span style="font-size: 14px;">An unfused linear layer does the matmul, writes the full $(M, N)$ output to HBM, reads it back to add the bias, writes again, reads a third time to apply ReLU, and writes a final time. That is one matmul plus four extra HBM round-trips on the activation tensor. The kernel below collapses all of it into one launch: the matmul accumulator never leaves registers between the last $\texttt{tl.dot}$ and the masked store, the bias is loaded once and broadcast across the row axis in-place, and $\texttt{tl.maximum}$ applies the ReLU before the single store. **Epilogue fusion** is the canonical optimization that makes Triton kernels worth writing for transformer inference.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, and $\text{bias} \in \mathbb{R}^{N}$, the kernel computes:</span>

$$
\texttt{out}[i, j] = \max\!\left(0,\; \sum_{k=0}^{K-1} A[i, k] \cdot B[k, j] + \text{bias}[j]\right)
$$

<span style="font-size: 14px;">The bias is per-column, broadcast across rows. The ReLU is applied to the post-bias value, not the pre-bias value; the order matters for negative bias entries. The launcher allocates $\texttt{out}$ and the kernel writes into it once.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The grid is identical to plain tiled matmul: 2D over the output tile structure, $\lceil M / \texttt{BLOCK\_M} \rceil \times \lceil N / \texttt{BLOCK\_N} \rceil$ programs. Each **program** owns a $\texttt{BLOCK\_M} \times \texttt{BLOCK\_N}$ output tile, runs its private $K$-loop with $\texttt{tl.dot}$ inner steps accumulating into a fp32 register tile, applies the epilogue (bias add, ReLU) on the completed accumulator in registers, and stores once with a masked $\texttt{tl.store}$. The decomposition is unchanged from the matmul case because the fused epilogue runs at the same granularity as the matmul: per-tile, per-program, in registers.</span>

<span style="font-size: 14px;">The parallel pattern is **per-tile reduction with fused epilogue**, which is the production-engineering form of a tiled matmul. Once a kernel emits a complete accumulator tile in registers, any element-wise or row-broadcast or column-broadcast operation on that tile is essentially free relative to the matmul cost. Fusion stacks naturally: matmul plus bias plus activation, matmul plus residual add, matmul plus LayerNorm fragment. The kernel below is the simplest concrete example; the same shape supports much more elaborate epilogues.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">Block sizes match the plain matmul: $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$, $\texttt{BLOCK\_K} = 32$, all $\texttt{tl.constexpr}$. The accumulator $\texttt{acc}$ is a $(64, 64)$ fp32 register tile. The bias tile is a $(\texttt{BLOCK\_N},)$ vector loaded once after the $K$-loop completes, with mask $\texttt{offs\_n} < N$ and $\texttt{other} = 0.0$ to handle the rightmost $N$-tile.</span>

<span style="font-size: 14px;">Mask discipline extends to four sites. The $K$-loop $A$ and $B$ loads are masked on the $M$ and $K$ tails (for $A$) and the $K$ and $N$ tails (for $B$), the bias load is masked on the $N$ tail, and the final store is masked on the $M$ and $N$ tails. The bias mask matters because the rightmost program along $\texttt{pid\_n}$ overshoots when $N$ is not a multiple of $\texttt{BLOCK\_N}$; without it the kernel reads past the end of the bias vector and contaminates the accumulator before the ReLU. Masking the load with $\texttt{other} = 0.0$ contributes zero to the out-of-range lanes of $\texttt{acc}$, which the store mask then discards anyway.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">$A$, $B$, $\text{bias}$, and $\texttt{out}$ live in HBM. The $K$-loop streams $A$ and $B$ tiles through compiler-staged SRAM into the tensor-core MMA instruction; the accumulator lives in **registers** for the entire lifetime of the program. The bias tile is loaded directly into registers; for a $(\texttt{BLOCK\_N},) = (64,)$ fp32 vector this is one $256$-byte burst, far below any other transaction in the kernel. The single store at the end materializes the post-ReLU accumulator to HBM, $\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N} \cdot 4$ bytes per program.</span>

<span style="font-size: 14px;">The reuse story for the matmul portion is unchanged: each $A$ slab is reused $\texttt{BLOCK\_N}$ times across the output column axis, each $B$ slab is reused $\texttt{BLOCK\_M}$ times across the row axis. The bias contributes a degenerate reuse: each fp32 of $\text{bias}$ is used $\texttt{BLOCK\_M}$ times (once for every row in the tile) when added to the accumulator, and is loaded once per program. Across programs, each fp32 of $\text{bias}$ is loaded $\lceil M / \texttt{BLOCK\_M} \rceil$ times (once per row block), with L2 absorbing most of the redundancy because the bias is small.</span>

<span style="font-size: 14px;">The saved HBM traffic is the whole point. An unfused implementation would write the full $(M, N)$ matmul output to HBM ($4 M N$ bytes), read it back to add bias ($4 M N$ bytes), write again ($4 M N$ bytes), read again to apply ReLU ($4 M N$ bytes), and write the final result ($4 M N$ bytes). That is $20 M N$ bytes of HBM traffic on the activation tensor. The fused kernel does $4 M N$ bytes (one final store), a $5\times$ reduction on the activation traffic. For a linear layer in a transformer, where the activation is typically the largest tensor in the operation, this is the single largest available optimization.</span>

<span style="font-size: 14px;">Kernel-launch overhead amplifies the difference. Each unfused stage costs a launch latency on the order of $5$ to $10$ microseconds plus the HBM-bound execution time. Three launches versus one means three times the launch overhead, and on small $M N$ the overhead can exceed the actual data-movement time. Fusion collapses three launches into one and recovers both the bandwidth and the launch latency in a single move.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">The matmul portion sets the floor: intensity $\frac{\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N}}{2(\texttt{BLOCK\_M} + \texttt{BLOCK\_N})}$ FLOPs per byte, evaluating to $16$ at $(64, 64)$, which is compute-bound on modern accelerators. The bias add contributes $\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N}$ extra FLOPs and $4 \texttt{BLOCK\_N}$ extra bytes per program; the ReLU contributes $\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N}$ more FLOPs and zero extra bytes. Both raise intensity slightly (more FLOPs over the same byte budget), pushing the kernel further into the compute-bound regime.</span>

<span style="font-size: 14px;">The relevant comparison is not the fused kernel's intensity but the unfused pipeline's effective intensity. Each unfused stage (bias add, ReLU) on its own is an element-wise kernel with intensity around $0.1$ FLOPs per byte, hard memory-bound, running at HBM speed. The fused form removes those stages entirely; their FLOPs are absorbed into the matmul kernel where they are essentially free, and their HBM bandwidth costs are removed. The end-to-end intensity of the linear-plus-bias-plus-ReLU operation goes up because the same FLOPs are now done over fewer bytes of traffic, not because any individual operation became more arithmetically dense.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">The author writes the matmul body (grid, tile shape, $K$-loop, $\texttt{tl.dot}$), the bias load with broadcast, the in-register add, the $\texttt{tl.maximum}$, and the final masked store. The fusion is structural, not a compiler pass: the author chose to put the epilogue inside the same kernel as the matmul rather than launch two more kernels, and the compiler emits the resulting code straightforwardly.</span>

<span style="font-size: 14px;">The compiler handles the broadcast efficiently. Triton's broadcasting follows NumPy semantics: a $(\texttt{BLOCK\_N},)$ vector indexed with $[\texttt{None}, :]$ has shape $(1, \texttt{BLOCK\_N})$ and broadcasts against the $(BLOCK\_M, \texttt{BLOCK\_N})$ accumulator. The codegen does not materialize a replicated $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N})$ bias tile in registers; it emits an MMA-shaped add that reads the bias once per output column and adds it to all rows, identical in cost to the matmul add-into-accumulator that already runs every $K$-loop iteration. The $\texttt{tl.maximum}$ lowers to a per-lane compare-and-select on the register tile.</span>

<span style="font-size: 14px;">One author decision worth highlighting: the bias load happens after the $K$-loop, not before. Loading it before makes no difference to correctness but extends the bias's lifetime in registers across the entire $K$-loop, raising register pressure and pushing the compiler toward worse spill decisions. Loading the bias right at the moment the accumulator is complete keeps the register footprint minimal for the duration of the inner loop and matches the standard production-kernel pattern.</span>

<span style="font-size: 14px;">The fused store handles the boundary lanes implicitly. The accumulator's out-of-range lanes may hold arbitrary values after the bias add and the ReLU, because the load masks zeroed only the operand inputs, not the accumulator positions themselves; the bias add can therefore propagate non-zero values into those lanes. The store mask discards them, so the final HBM write is correct regardless. Reusing the operand load masks to also pre-clear the accumulator is unnecessary overhead.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The unfused PyTorch path is $\texttt{torch.relu(A @ B + bias)}$, which dispatches three kernels: cuBLAS for the matmul, an element-wise kernel for the bias add, and an element-wise kernel for the ReLU. Each element-wise kernel is HBM-bound, runs at peak HBM bandwidth, and contributes about $2 M N / B_{\text{HBM}}$ seconds to the total wall time on top of the matmul. For a $(4096, 4096) \times (4096, 4096)$ linear layer with $1$ TB/s HBM, the two element-wise stages contribute on the order of $130$ microseconds each, while the matmul itself at a few percent of peak tensor-core throughput takes a few hundred microseconds. Fusion saves the two element-wise stages outright.</span>

<span style="font-size: 14px;">The fused kernel cannot fuse arbitrarily far. Once the epilogue requires cross-tile or cross-row information (a row-wise softmax, a column-wise reduction, a LayerNorm that needs the variance of the full row), the operation no longer fits inside one tile and cannot be done in registers. The natural composition is **matmul plus tile-local epilogue**: bias add, scaling, dropout, ReLU, GELU, residual add against a tensor of the same shape. Anything that needs cross-tile information becomes a separate kernel or a more elaborate fused-attention-style construction.</span>

<span style="font-size: 14px;">Two further fused-matmul variants are worth naming. **Matmul plus residual add** loads a same-shape residual tile after the $K$-loop and adds it to the accumulator before any activation; the residual reuses the store-mask predicate exactly. **Matmul plus quantized output** applies a scale and clamp to the accumulator and casts to int8 or fp8 before the store; this is the production path for quantized transformer inference and reuses the same fused-epilogue structure with a different element-wise expression. The kernel here is the canonical pattern; the variations differ only in what runs in the epilogue window between the $K$-loop and the store.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $M = N = 2$, $K = 4$, $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 2$, $\texttt{BLOCK\_K} = 2$. The launch grid is $1 \times 1 = 1$ program covering the entire $(2, 2)$ output tile.</span>

<span style="font-size: 14px;">The $K$-loop runs $K / \texttt{BLOCK\_K} = 2$ iterations. Iteration $1$: load $A[0{:}2, 0{:}2]$ and $B[0{:}2, 0{:}2]$, $\texttt{tl.dot}$ produces a $(2, 2)$ partial, accumulate. Iteration $2$: load $A[0{:}2, 2{:}4]$ and $B[2{:}4, 0{:}2]$, $\texttt{tl.dot}$ produces another partial, accumulate. After the loop $\texttt{acc}$ holds the full $A B$ tile. The bias load pulls $\text{bias}[0{:}2]$ into a $(2,)$ vector; the broadcast add $\texttt{acc} \mathrel{+}= \text{bias}[\texttt{None}, :]$ adds $\text{bias}[0]$ to both entries in $\texttt{acc}[:, 0]$ and $\text{bias}[1]$ to both entries in $\texttt{acc}[:, 1]$. The $\texttt{tl.maximum(acc, 0.0)}$ clamps any negative entries to zero. The masked store writes the four resulting fp32s to $\texttt{out}[0{:}2, 0{:}2]$.</span>

<span style="font-size: 14px;">Counting HBM operations across the whole kernel: $4 + 4$ fp32s for the two $A$ slabs, $4 + 4$ fp32s for the two $B$ slabs, $2$ fp32s for the bias, $4$ fp32s for the final store. Total HBM traffic is $22$ fp32s. The unfused pipeline for the same problem would do $16 + 16 + 16 + 16 + 16 = 80$ fp32s on the activation tensor alone ($M N$ reads and writes per stage, four extra stages), in addition to whatever the matmul costs on $A$ and $B$. The fusion saves about three-quarters of the HBM traffic for this tiny example, and the saving grows with $M \cdot N$ relative to $K$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong operation order.** Applying ReLU before the bias add gives $\text{ReLU}(A B) + \text{bias}$, a different function: any negative pre-bias accumulator entry is zeroed before the bias can rescue it, and any positive bias value is added to a clamped zero instead of a negative number. The order is bias first, then ReLU.</span>

* <span style="font-size: 14px;">**Wrong broadcast direction on the bias.** Using $\text{bias}[:, \texttt{None}]$ broadcasts the bias across the column axis instead of the row axis, producing a kernel that adds $\text{bias}[i]$ instead of $\text{bias}[j]$ to $\texttt{out}[i, j]$. The kernel runs and may pass square symmetric tests; it fails everywhere else. The shape contract is $\text{bias}[\texttt{None}, :]$ to broadcast a column vector across rows.</span>

* <span style="font-size: 14px;">**Missing the bias load mask.** The rightmost $N$-tile reads past the end of the bias vector when $N$ is not a multiple of $\texttt{BLOCK\_N}$. Without $\texttt{offs\_n} < N$ on the bias load with $\texttt{other} = 0.0$, the kernel contaminates the accumulator before the ReLU and produces wrong outputs along the boundary.</span>

* <span style="font-size: 14px;">**Bias load before the $K$-loop.** Loading the bias at the top of the kernel extends its lifetime across the entire $K$-loop, raising register pressure and potentially forcing accumulator spills. The standard pattern is to load the bias only after the $K$-loop completes, applying it in the tight epilogue window just before the store.</span>

---