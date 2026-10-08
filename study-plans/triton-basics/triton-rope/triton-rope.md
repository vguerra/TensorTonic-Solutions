# <span style="font-size: 20px;">Rotary Position Embedding</span>

<span style="font-size: 14px;">RoPE looks heavy mathematically and turns out to be one of the simplest kernels in the curriculum. Each token row rotates pairs of its embedding channels by a position-dependent angle drawn from precomputed $\cos$ and $\sin$ tables. The pattern is a **pure per-row map with no reduction**, despite the framing of "encoding absolute position via 2D rotations on $D/2$ subspaces". One program per row, four loads, two FMA pairs, two stores. The whole kernel is bandwidth-bound and runs at HBM speed; the systems story is in the strided pair layout and the tiny tables that stay hot in cache. Su et al. introduced the formulation in 2021; every modern transformer that does not use ALiBi or absolute embeddings reaches for this kernel.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given $x \in \mathbb{R}^{N \times D}$ with $D$ even, and precomputed tables $\cos, \sin \in \mathbb{R}^{N \times D/2}$, the output is</span>

$$
\texttt{out}[i, 2j] = x[i, 2j] \cdot \cos[i, j] - x[i, 2j+1] \cdot \sin[i, j]
$$

$$
\texttt{out}[i, 2j+1] = x[i, 2j] \cdot \sin[i, j] + x[i, 2j+1] \cdot \cos[i, j]
$$

<span style="font-size: 14px;">Per pair, this is the 2D rotation matrix applied to the column vector $(x_{2j}, x_{2j+1})^\top$. The interleaved layout (even and odd channels packed adjacent in memory) is one convention; the other common convention rotates the first half against the second half. This problem uses the interleaved convention, which matters because the two are not interchangeable and a kernel that follows the wrong layout produces silently wrong output.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional with $N$ **programs**, one per token row. Each program reads $\texttt{row} = \texttt{tl.program\_id(0)}$ and owns the entire $D$-wide vector for that token: it loads the row of $x$, the row of $\cos$, the row of $\sin$, and writes the row of $\texttt{out}$. There is no cross-program communication, no reduction, no shared state. Even within a row, the $D/2$ pairs are fully independent because each pair depends only on its own two channels.</span>

<span style="font-size: 14px;">The reason to put one program per row rather than per pair is overhead amortization. The angle tables are read at row granularity; the strides for $x$ and $\texttt{out}$ are computed once per row; and the launch grid stays at a sensible $N$ rather than blowing up to $N \cdot D / 2$. For typical sequence lengths in the thousands, this gives plenty of parallelism to fill the device while keeping each program's footprint small.</span>

<span style="font-size: 14px;">An alternative grid would be 2D over (row, pair-tile), which splits long rows across multiple programs. For very large $D$ (say $D = 8192$ after a head merge) that helps because one $\texttt{BLOCK\_SIZE}$ may not cover the whole row. For typical per-head $D$ in the range $64$ to $128$, the 1D grid is the right choice because a single $\texttt{BLOCK\_SIZE} \ge D/2$ covers the row and the per-program work stays substantial.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The single constexpr meta-parameter is $\texttt{BLOCK\_SIZE}$, rounded up to the next power of two above $D/2$. The pair index $j$ runs over $\texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$ and is masked by $\texttt{mask} = j < D/2$. Because $D$ is always even (a precondition), the half count $D/2$ is the natural granularity for the tile; the same mask gates the even loads, the odd loads, the $\cos$ load, the $\sin$ load, and both stores. One mask, four loads, two stores, all sharing the predicate.</span>

<span style="font-size: 14px;">Strided access is the subtle part. The even channels live at offsets $2j$ within the row, the odd at $2j+1$. The pointer expressions $x_\texttt{ptr} + \texttt{row} \cdot D + 2j$ and $x_\texttt{ptr} + \texttt{row} \cdot D + 2j + 1$ make this explicit. The $\cos$ and $\sin$ tables are dense along $D/2$ with strides $\texttt{row} \cdot (D/2) + j$. Two different row strides, $D$ for $x$ and $D/2$ for the tables, are a common source of off-by-half bugs.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">No reuse across programs. Each program reads its row of $x$, its row of $\cos$, its row of $\sin$, and writes its row of $\texttt{out}$, all in HBM. Within a program, the four loaded tiles live in registers and feed directly into the two FMA pairs. There is no SRAM staging because there is nothing to stage; the kernel passes data straight from registers to the arithmetic unit and back to HBM.</span>

<span style="font-size: 14px;">The interesting cache story is the angle tables. The $\cos$ and $\sin$ tables are precomputed once for a given sequence length and shared across all token rows of a batch in production. Total size is $2 \cdot N \cdot D/2 \cdot 4 = 4 N D$ bytes in fp32, which for $N = 4096, D = 64$ is $1$ MB. That fits comfortably in the L2 cache on every modern accelerator, so a second RoPE call within the same batch hits cache for the tables and pays HBM cost only on $x$ and $\texttt{out}$. The kernel as written does not exploit this explicitly; the hardware does it for free.</span>

<span style="font-size: 14px;">The strided even-odd loads are worth one note. A naive reading might worry that loading $2j$ and $2j+1$ separately costs two HBM transactions per pair. In practice the compiler observes that the two loads cover adjacent offsets and coalesces them into the same transactions as a contiguous $\texttt{BLOCK\_SIZE} \cdot 2$ tile. There is no bandwidth penalty for the interleaved layout, only the slight register cost of holding $x_\text{even}$ and $x_\text{odd}$ as separate tiles.</span>

<span style="font-size: 14px;">The decision to load even and odd as separate tiles rather than as one contiguous tile and deinterleave in registers is a Triton ergonomics choice. The compiler could just as well lower a single contiguous load and a register-side shuffle into the same machine code, but expressing the access pattern as two strided loads makes the dependency on the angle tables explicit to the reader and to the autotuner. Either form lowers to the same memory transactions; the strided form is what reads idiomatically.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output pair, the kernel reads $2$ values from $x$, $2$ from the angle tables, writes $2$ to $\texttt{out}$, and performs $4$ multiplies and $2$ adds. That is $6$ FLOPs per $6 \cdot 4 = 24$ bytes of HBM traffic, an arithmetic intensity of $0.25$ FLOPs per byte. Firmly **memory-bound**, comparable to vector addition or any pointwise activation. The roofline crossover on modern accelerators sits around $10$ FLOPs per byte for fp32; RoPE is two orders of magnitude under it.</span>

<span style="font-size: 14px;">The implication is that arithmetic optimizations buy nothing. The kernel runs at HBM bandwidth; the only meaningful tuning knob is whether the four loads coalesce cleanly. Fusing RoPE with the surrounding operations (the QKV projection that produces $x$, or the attention kernel that consumes $\texttt{out}$) removes a round-trip through HBM and is the real performance lever. Standalone RoPE is the educational form; production stacks fuse it.</span>

<span style="font-size: 14px;">One subtlety distinguishes RoPE from other pointwise kernels at the same intensity: it has two inputs that share a footprint smaller than the main data. The angle tables are $D / 2$ wide per row while $x$ is $D$ wide, so the tables contribute $1/3$ of the input bandwidth. As $N$ grows the absolute bandwidth grows linearly, but the table footprint stays at $4 N D$ bytes regardless of how many times the kernel is called across a batch, which is why the L2 reuse story across batched calls is real. A single forward pass on a single sample is HBM-bound; a forward pass across many samples that share the same $(N, D)$ shape can ride a hot table cache.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author handles:** the one-program-per-row decomposition, the constexpr block size, the strided pointer arithmetic for even and odd channels, the single tail mask shared across the four loads and two stores, the rotation algebra ($x_e c - x_o s$ for even, $x_e s + x_o c$ for odd), and the row-stride distinction between $x$ ($D$) and the angle tables ($D/2$). Most of these are layout decisions the compiler cannot infer from the shape of the tensors alone.</span>

<span style="font-size: 14px;">**Compiler handles:** coalescing the four loads into wide PTX vector instructions, allocating registers for the four input tiles, scheduling the four FMA-family operations to overlap with the next load, and emitting the two writes as similarly wide stores. The compiler also picks the warp count internally; for a kernel this small, the default $\texttt{num\_warps} = 4$ is typically optimal because there is no register pressure to push higher.</span>

<span style="font-size: 14px;">The compiler also gets the two register-tile multiplications ($x_e \cdot c$ and $x_o \cdot s$, then $x_e \cdot s$ and $x_o \cdot c$) for free as standard FMA instructions, with both subtraction and addition forms available. The four multiplies fuse into two FMA-add instructions per output pair on hardware with hardware FMA support, halving the instruction count without any author-side change.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The canonical kernel above is already the optimized form for the standalone case. The kernel performs exactly the minimum HBM traffic (one read of $x$, one read of the tables, one write of $\texttt{out}$) and runs at near-peak bandwidth. The remaining performance wins are at the integration boundary, not inside the kernel.</span>

<span style="font-size: 14px;">**Fusion with the QKV projection:** the kernel that produces $x$ from the input hidden states (a matmul or grouped matmul) writes $x$ to HBM only to have RoPE read it back. Fusing the RoPE rotation into the epilogue of the projection eliminates the round-trip. This is the production pattern in most inference engines.</span>

<span style="font-size: 14px;">**Fusion with the attention kernel:** if RoPE is applied to $Q$ and $K$ immediately before the attention computation, the rotated tensors can be produced inside the attention kernel's load path rather than written and re-read. This is what fused FlashAttention-with-RoPE kernels do, and it is the larger optimization in practice because attention reads $Q$ and $K$ many times in its inner loop.</span>

<span style="font-size: 14px;">**Block-size choice:** for very small $D$ (say $D = 16$), $\texttt{BLOCK\_SIZE} = 16$ wastes most of the launch overhead on tiny programs. For larger $D$ (say $D = 128$), $\texttt{BLOCK\_SIZE} = 64$ fits the half-row in one tile and lets the compiler emit wide vector loads. The reference implementation rounds up to the next power of two, which handles both extremes safely.</span>

<span style="font-size: 14px;">**Cache layout for the angle tables:** the reference passes $\cos$ and $\sin$ as separate $(N, D/2)$ tensors. An alternative is to interleave them into one $(N, D)$ tensor that mirrors the layout of $x$, which lets a single load fetch both at once. The reference's separate-tensor layout is more common because the tables are typically computed once at model load and shared across many forward passes, and keeping them separate avoids a duplicate write at construction.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 1, D = 4$, so the row has channels $(x_0, x_1, x_2, x_3)$ and the angle tables have entries $(\cos_0, \cos_1)$ and $(\sin_0, \sin_1)$. The grid is a single program. $\texttt{BLOCK\_SIZE} = 2$ (the next power of two above $D/2 = 2$), $j \in \{0, 1\}$, mask $\{T, T\}$.</span>

<span style="font-size: 14px;">**Loads:** $x_\text{even} = (x_0, x_2)$ from offsets $\{0, 2\}$, $x_\text{odd} = (x_1, x_3)$ from offsets $\{1, 3\}$, $c = (\cos_0, \cos_1)$ from the dense $\cos$ row, $s = (\sin_0, \sin_1)$ from the dense $\sin$ row.</span>

<span style="font-size: 14px;">**Rotation:** $\texttt{out\_even} = (x_0 \cos_0 - x_1 \sin_0, x_2 \cos_1 - x_3 \sin_1)$ and $\texttt{out\_odd} = (x_0 \sin_0 + x_1 \cos_0, x_2 \sin_1 + x_3 \cos_1)$.</span>

<span style="font-size: 14px;">**Stores:** $\texttt{out\_even}$ writes to offsets $\{0, 2\}$ and $\texttt{out\_odd}$ writes to $\{1, 3\}$, reinterleaving the output back to the original $(D,)$ layout.</span>

<span style="font-size: 14px;">For $D = 6$ and $\texttt{BLOCK\_SIZE} = 4$, the mask is $\{T, T, T, F\}$. Lane $3$ of $j$ corresponds to pair index $3$, which would read from offsets $\{6, 7\}$ in $x$; those are past the end of the $D = 6$ row, and the mask zeros the loads so the masked store does nothing. The half-row fits in lanes $0, 1, 2$ and the kernel writes exactly the right $6$ channels.</span>

<span style="font-size: 14px;">A sanity check on the algebra: for $\cos = 1, \sin = 0$ (zero angle), the rotation reduces to $\texttt{out\_even} = x_\text{even}, \texttt{out\_odd} = x_\text{odd}$, so the kernel is the identity. For $\cos = 0, \sin = 1$ (quarter turn), $\texttt{out\_even} = -x_\text{odd}, \texttt{out\_odd} = x_\text{even}$, the standard counter-clockwise 90-degree rotation per pair. These corner cases are useful to verify the sign convention in a debugger.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong pair convention.** Some references rotate $x[:, 0:D/2]$ against $x[:, D/2:]$ instead of even-odd pairs. This problem uses the interleaved convention $2j, 2j+1$; mixing the two produces a numerically valid but semantically wrong embedding that breaks any model trained against the interleaved scheme.</span>
* <span style="font-size: 14px;">**Sign error in $\texttt{out\_odd}$.** The canonical rotation matrix has $+x_\text{even} \sin$ in the odd output. Flipping the sign to $-x_\text{even} \sin$ gives a clockwise rotation, which composes with itself the wrong way over multiple positions and silently degrades long-context behavior.</span>
* <span style="font-size: 14px;">**Wrong angle-table stride.** The $\cos$ and $\sin$ rows have width $D/2$, not $D$. Using $\texttt{row} \cdot D + j$ instead of $\texttt{row} \cdot (D/2) + j$ reads from the next row's table and shifts the position by one, which is invisible to row 0 (the wrap-around does not happen) and catastrophic to every row after it.</span>
* <span style="font-size: 14px;">**Forgetting the tail mask on the strided stores.** When $D/2$ is not a power of two, the last $\texttt{BLOCK\_SIZE} - D/2$ lanes are out of range. The same $j < D/2$ mask must gate the two stores; otherwise the kernel writes past the row into the next token's embedding.</span>

---