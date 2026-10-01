# <span style="font-size: 20px;">LayerNorm Backward</span>

<span style="font-size: 14px;">The backward pass of LayerNorm is the kernel where Norm and Training stops being easy. Three gradient tensors come out at once: $dx$ (same shape as the input), and the parameter gradients $d\gamma$ and $d\beta$ (length $N$). The $dx$ gradient is per-row and looks structurally like the forward pass with two extra reductions; the parameter gradients are sums **across** rows, which means one program cannot own them alone. The natural design is one program per row, computing $dx$ with two intra-row reductions and atomically accumulating the row's contribution into $d\gamma$ and $d\beta$. This mirrors the official Triton tutorial for fused LayerNorm and is the canonical example of mixing intra-program reductions with cross-program atomics in a single kernel.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given the forward input $x \in \mathbb{R}^{M \times N}$, the scale $\gamma \in \mathbb{R}^{N}$, and the upstream gradient $dy \in \mathbb{R}^{M \times N}$, the backward computes $\hat{x}[i, j] = (x[i, j] - \mu_i) / \sqrt{\sigma_i^2 + \epsilon}$ and $dy_{\text{norm}}[i, j] = dy[i, j] \cdot \gamma[j]$, then</span>

$$
dx[i, j] = r_{\text{std}, i} \cdot \left( dy_{\text{norm}}[i, j] - \frac{1}{N} \sum_k dy_{\text{norm}}[i, k] - \hat{x}[i, j] \cdot \frac{1}{N} \sum_k dy_{\text{norm}}[i, k] \cdot \hat{x}[i, k] \right)
$$

$$
d\gamma[j] = \sum_i dy[i, j] \cdot \hat{x}[i, j], \qquad d\beta[j] = \sum_i dy[i, j]
$$

<span style="font-size: 14px;">The $dx$ formula has two row-mean terms $c_1 = \text{mean}(dy_{\text{norm}})$ and $c_2 = \text{mean}(dy_{\text{norm}} \cdot \hat{x})$ that depend on the whole row, so $dx[i, j]$ cannot be written until both row-mean scalars are known. The parameter gradients are column reductions: $d\gamma[j]$ collects one contribution from every row at column $j$.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is $(M,)$, one **program** per row, exactly as in the forward pass. Each program:</span>

<span style="font-size: 14px;">1. **Loads** the row of $x$, the row of $dy$, and the shared $\gamma$ into registers.</span>

<span style="font-size: 14px;">2. **Recomputes** $\mu_i$, $\sigma_i^2$, $r_{\text{std}, i}$, and $\hat{x}_i$ from $x_i$ alone, the same way the forward pass did. No saved statistics are read from a buffer; recomputation is cheaper than the HBM bandwidth a saved-mean/saved-rstd tensor would cost.</span>

<span style="font-size: 14px;">3. **Reduces** $c_1 = \text{mean}(dy \cdot \gamma)$ and $c_2 = \text{mean}(dy \cdot \gamma \cdot \hat{x})$ with two $\texttt{tl.sum}$ calls.</span>

<span style="font-size: 14px;">4. **Computes** $dx[i, j] = r_{\text{std}} \cdot (dy_{\text{norm}} - c_1 - \hat{x} \cdot c_2)$ in registers and stores it to HBM with one $\texttt{tl.store}$.</span>

<span style="font-size: 14px;">5. **Atomically adds** $dy \cdot \hat{x}$ to $d\gamma$ (length $N$, indexed by $\texttt{cols}$) and $dy$ to $d\beta$ (also length $N$) via $\texttt{tl.atomic\_add}$ with the same $\texttt{cols} < N$ mask used on the loads and store.</span>

<span style="font-size: 14px;">The atomic adds are the cross-row combine. Without them, the parameter gradients would be unwritable from a per-row decomposition - each row only knows its own contribution, and the sum across rows is what every other row also wants to write into. The atomic primitive is the only way to combine partial results across programs without a second kernel.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">Tile shape is $(\texttt{BLOCK\_SIZE},)$ with $\texttt{BLOCK\_SIZE} = \texttt{triton.next\_power\_of\_2}(N)$, declared $\texttt{tl.constexpr}$. Three tile-shaped buffers participate: the row of $x$, the row of $dy$, and the shared $\gamma$. All three use the same tail mask $\texttt{cols} < N$ on every $\texttt{tl.load}$ with $\texttt{other} = 0.0$.</span>

<span style="font-size: 14px;">The mask discipline matters more here than in the forward pass because the kernel does four reductions on the row tiles (the two for $\mu$/$\sigma^2$ recomputation, plus $c_1$ and $c_2$). Garbage in any one of them propagates into a wrong $r_{\text{std}}$, a wrong $\hat{x}$, a wrong $c_1$, a wrong $c_2$, and ultimately a wrong $dx$ everywhere in the row. The $\texttt{other} = 0.0$ load value is what keeps masked lanes neutral in all four reductions.</span>

<span style="font-size: 14px;">The atomic adds also use the same mask. Without it, the atomic-add at lane $j \geq N$ would write into addresses past the $d\gamma$ and $d\beta$ buffers. The mask gates the atomic so masked lanes contribute zero (a no-op rather than an out-of-bounds write), which is the correct behavior at the tail of any partial row.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Per row, the kernel reads $4N$ bytes of $x$, $4N$ bytes of $dy$, and $4N$ bytes of $\gamma$ (warm in L2 after the first few rows). It writes $4N$ bytes of $dx$ once, and atomic-adds $4N$ bytes worth of contributions into each of $d\gamma$ and $d\beta$. Total HBM traffic per row is approximately $20N$ bytes - the highest of any kernel in the norm family.</span>

<span style="font-size: 14px;">The intra-row reuse is intense. The $x$ row is touched in two reductions (for $\sum x$ and $\sum x^2$) plus the $\hat{x}$ subtract; the $dy$ row is touched in the $c_1$ reduction, the $c_2$ reduction, and the $dx$ assembly plus the $d\gamma$ atomic; $\gamma$ is touched in the $dy_{\text{norm}}$ multiply. Counting register-level reuses, the kernel does roughly $15$ arithmetic touches per element across one HBM load. That ratio is what makes the recomputation strategy work: HBM bandwidth on $x$ is paid once and amortized over the forward statistics, the $\hat{x}$ multiply, and the $dy \cdot \hat{x}$ accumulation into $d\gamma$.</span>

<span style="font-size: 14px;">The atomic contention on $d\gamma$ and $d\beta$ is the unique cost of this kernel. Every program writes to the same length-$N$ buffer; with $M$ programs, each address sees $M$ atomic updates over the launch. Modern accelerators serialize atomic updates to the same address but parallelize updates to different addresses, so the contention on any single $d\gamma[j]$ slot is bounded by $M$ serial adds rather than the full $M \cdot N$ updates. For $M$ in the hundreds, this is fast in absolute terms; for $M$ in the millions, the atomic step starts to dominate and a two-stage parallel reduction (each program writes its row to a temporary $(M, N)$ buffer; a second kernel reduces over $M$) becomes faster.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output element of $dx$, the kernel does roughly $15$ FLOPs (the forward statistics, $c_1$, $c_2$, the $dx$ assembly, and the two atomic-add operands). Bytes moved are $\approx 20$ per element (three reads, one $dx$ write, two atomic writes). Intensity:</span>

$$
\frac{\approx 15 \text{ FLOPs}}{\approx 20 \text{ bytes}} \approx 0.75 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">Higher than the forward LayerNorm ($\approx 0.4$) but still firmly **memory-bound** on modern accelerators. The kernel runs at HBM bandwidth, and the optimizations that move runtime are the ones that reduce bytes moved or reduce the atomic contention - not the ones that reduce arithmetic.</span>

<span style="font-size: 14px;">The runtime profile of this kernel is roughly: HBM read time on the $x$, $dy$, $\gamma$ rows; a small amount of register arithmetic for the reductions and the $dx$ assembly; HBM write time on $dx$; and a final atomic-add latency on the two parameter-gradient buffers. The first dominates for small $M$ (bandwidth-bound) and the last dominates for very large $M$ (atomic-contention-bound). The crossover is in the low millions of rows, far above the test harness's sizes.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author chooses:** the grid, the constexpr block size, the recompute-vs-load decision for the forward statistics ($\mu$, $\sigma^2$, $r_{\text{std}}$, $\hat{x}$ are not saved between forward and backward; the backward recomputes them from $x$), the order of the two reductions ($c_1$ before $c_2$ so $c_2$ can use the same $dy_{\text{norm}}$ tile already in registers), the host-side zeroing of $d\gamma$ and $d\beta$ before the launch, and the atomic-add strategy versus a two-stage parallel reduction.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering the four reductions to tree reductions on the tile, scheduling the three HBM loads to overlap, allocating registers for the three tiles, and emitting the atomic-add as a hardware atomic instruction. The compiler also keeps $\hat{x}$ in registers between its construction and its uses in $c_2$, the $dx$ assembly, and the $d\gamma$ atomic, so the kernel never materializes $\hat{x}$ to HBM.</span>

<span style="font-size: 14px;">The most consequential author-side decision is the recomputation choice. Saving $\mu_i$ and $r_{\text{std}, i}$ from the forward pass into two length-$M$ buffers and loading them in the backward would cost $8M$ bytes of HBM traffic in exchange for skipping the $\sum x$ and $\sum x^2$ reductions in the backward. For $M$ rows of width $N$, the forward statistics cost $2 N \cdot M$ FLOPs; the HBM round-trip costs $8M$ bytes. The recompute is cheaper whenever $N > 4$ (always), and avoids holding a forward-pass tensor across the backward boundary. This is the same recomputation pattern that gradient checkpointing generalizes.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A naive backward decomposition is at least four kernels: one to recompute the forward statistics, one to compute $c_1$ and $c_2$, one to assemble $dx$, and one (or two) to sum across rows for $d\gamma$ and $d\beta$. Each unfused step writes intermediates to HBM and reads them back; the $x$ row would be touched in two separate kernels, the $\hat{x}$ tensor might be materialized in HBM as an intermediate, and the parameter-gradient sums would be cross-row reductions that need either a two-stage kernel or a per-row scratch buffer.</span>

<span style="font-size: 14px;">The fused Triton form reads $x$, $dy$, and $\gamma$ once each, writes $dx$ once, and atomically accumulates $d\gamma$ and $d\beta$. HBM traffic on the input side drops by roughly $2 \times$ relative to a moderately-unfused decomposition. The atomic-add primitive is what enables the fusion: without it, the parameter gradients would need a separate cross-row reduction kernel and a temporary $(M, N)$ buffer.</span>

<span style="font-size: 14px;">For very large $M$, the official Triton tutorial uses a two-stage parallel reduction for $d\gamma$ and $d\beta$ instead of atomics. Each program writes its row's partial gradients to a scratch buffer of shape $(\text{num\_programs}, N)$, and a second kernel reduces along the program dimension. This trades atomic contention for an extra kernel launch and an extra HBM round-trip, and it pays off when $M$ is large enough that atomic serialization on individual $d\gamma[j]$ slots becomes the bottleneck. The atomic form is simpler and sufficient for the sizes here.</span>

<span style="font-size: 14px;">A third strategy, used in some highly-optimized production kernels, groups programs by row range and uses a hierarchical reduction: programs in the same group combine their row contributions in shared memory before a single atomic-add per group writes the group's partial sum into the global buffer. That cuts the atomic-add count from $M$ to $M / \text{group\_size}$ at the cost of intra-group synchronization. None of the three strategies changes the gradient math; they differ only in how the cross-row sum is combined.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $M = 1$, $N = 4$, $\texttt{BLOCK\_SIZE} = 4$, $\epsilon = 10^{-5}$, $\gamma = [1, 1, 1, 1]$, $x = [1, 2, 3, 4]$, $dy = [0.1, 0.2, 0.3, 0.4]$. The launch grid is $(1,)$.</span>

<span style="font-size: 14px;">**Program 0**: recompute $\mu = 2.5$, $\sigma^2 = 1.25$, $r_{\text{std}} \approx 0.8944$, $\hat{x} \approx [-1.342, -0.447, 0.447, 1.342]$. $dy_{\text{norm}} = dy \cdot \gamma = [0.1, 0.2, 0.3, 0.4]$. $c_1 = \text{mean}(dy_{\text{norm}}) = 0.25$. $c_2 = \text{mean}(dy_{\text{norm}} \cdot \hat{x}) \approx (-0.134 - 0.089 + 0.134 + 0.537) / 4 = 0.112$. $dx = r_{\text{std}} \cdot (dy_{\text{norm}} - c_1 - \hat{x} \cdot c_2) \approx 0.8944 \cdot ([0.1, 0.2, 0.3, 0.4] - 0.25 - [-0.150, -0.050, 0.050, 0.150]) = 0.8944 \cdot [0.00, 0.00, 0.00, 0.00] \approx [0, 0, 0, 0]$.</span>

<span style="font-size: 14px;">The $dx$ is approximately zero in this constructed case because $dy$ is itself proportional to $\hat{x}$ after the affine, which is the kernel of LayerNorm's gradient. The atomic adds: $d\gamma[j] \mathrel{+}= dy[j] \cdot \hat{x}[j] \approx [-0.134, -0.089, 0.134, 0.537]$ and $d\beta[j] \mathrel{+}= dy[j] = [0.1, 0.2, 0.3, 0.4]$. With only one program in the grid, there is no contention; with $M = 1000$, the same two atomic-adds would happen $1000$ times for each $j$ slot, serialized but summing to the correct cross-row total.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to pre-zero $d\gamma$ and $d\beta$.** $\texttt{tl.atomic\_add}$ accumulates onto whatever the buffer already holds. Without $\texttt{dgamma\_out.zero\_()}$ and $\texttt{dbeta\_out.zero\_()}$ before the launch, the kernel adds the row contributions to whatever residual the buffers contained. $dx$ does not need zeroing because it is written, not accumulated.</span>
* <span style="font-size: 14px;">**Skipping the mask on the atomic-adds.** The $\texttt{cols} < N$ mask must gate the $\texttt{tl.atomic\_add}$ on $d\gamma$ and $d\beta$ at the tail, otherwise the kernel writes into addresses past the length-$N$ buffers. Forgetting this is silent corruption rather than a crash, because the writes are atomic and the addresses are valid memory (just not the right tensor).</span>
* <span style="font-size: 14px;">**Materializing $\hat{x}$ to HBM.** A reasonable-looking decomposition stores $\hat{x}$ to an intermediate buffer between the recomputation and the $c_2$ reduction. This doubles the HBM traffic on a tensor the size of $x$ and gives back most of the win of fusion. $\hat{x}$ must stay in registers across all of its uses: the $c_2$ accumulation, the $dx$ assembly, and the $d\gamma$ atomic.</span>
* <span style="font-size: 14px;">**Tightening the tolerance below the atomic noise.** Atomic-add ordering is nondeterministic, and the row reductions accumulate float32 rounding differently from the PyTorch reference. Combined $\texttt{atol} = 10^{-2}, \texttt{rtol} = 10^{-2}$ is the realistic tolerance for all three outputs; anything tighter will flake on $N = 1024$ even for a correct implementation.</span>

---