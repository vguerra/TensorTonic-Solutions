# <span style="font-size: 20px;">LayerNorm Forward</span>

<span style="font-size: 14px;">LayerNorm is the first kernel in the curriculum where **fusion** stops being a slogan and starts being the whole point. The forward pass needs four logically distinct steps: a mean, a variance, a normalize-and-affine, and a write back. A naive implementation chains four passes through HBM; the Triton version collapses all four into one program per row, computes both reduction statistics from a single read of the row, and writes the affined output without ever spilling intermediates to memory.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For each row $i$ of $x \in \mathbb{R}^{M \times N}$, the kernel computes the row mean and variance and applies the per-feature affine $(\gamma, \beta)$:</span>

$$
\mu_i = \frac{1}{N} \sum_{j=0}^{N-1} x[i, j], \qquad \sigma_i^2 = \frac{1}{N} \sum_{j=0}^{N-1} (x[i, j] - \mu_i)^2
$$

$$
\texttt{out}[i, j] = \frac{x[i, j] - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \cdot \gamma[j] + \beta[j]
$$

<span style="font-size: 14px;">The variance is computed from centered values. Since the row remains in registers after the mean reduction, the kernel performs the centered-variance reduction without rereading the row from HBM. This avoids the cancellation in $E[x^2] - E[x]^2$. $\gamma$ and $\beta$ are length-$N$ shared parameters, indexed by the column rather than the row, so they participate in the affine but not in either reduction.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">Rows are independent: row $i$'s mean and variance depend only on $x[i, :]$, and row $i$'s output depends only on row $i$'s statistics plus the shared $(\gamma, \beta)$. The launch grid is $(M,)$, one **program** per row. Each program reads its row index from $\texttt{tl.program\_id(0)}$, computes its row offset as $\texttt{row} \cdot \texttt{stride\_x\_row}$, and operates on the whole row in registers. There is no cross-program communication and no atomic accumulation. The kernel is embarrassingly parallel at the row level, with the work inside each program structured as a small fused pipeline.</span>

<span style="font-size: 14px;">This decomposition is what makes the LayerNorm forward "easy" relative to the backward. A reduction lives inside the program; the cross-row dimension is purely a grid index. The full row of width $N$ fits in registers because production transformer dimensions ($768$, $1024$, $4096$) are well within the per-program register budget of modern GPUs when held as a single fp32 tile.</span>

<span style="font-size: 14px;">The pattern is identical to fused softmax in shape: one program per row, one reduction inside the program, no cross-program combine. The difference is that LayerNorm needs **two** reductions and an affine, where softmax needs a max, an exp-sum, and a divide. The structural similarity is why both kernels appear early in any Triton curriculum: once a reader has internalized "one program per row, full row in registers, reduce in place", every per-row reduction kernel becomes a small variation on the same template.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The tile shape is one-dimensional: each program holds a $(\texttt{BLOCK\_SIZE},)$ tile of fp32. Because the row width $N$ is a runtime value but Triton tile shapes must be **constexpr**, the launcher picks $\texttt{BLOCK\_SIZE} = \texttt{triton.next\_power\_of\_2}(N)$, declared $\texttt{tl.constexpr}$. For $N = 768$ this is $1024$; for $N = 1024$ it is itself; for $N = 257$ it is $512$.</span>

<span style="font-size: 14px;">The tail mask $\texttt{cols} < N$ disables the overshoot. The $x$ load uses $\texttt{other} = 0.0$ so masked lanes contribute zero to the mean reduction. After subtracting the mean, padded lanes would contain $-\mu$, so $\texttt{tl.where(mask, x - mean, 0.0)}$ restores them to zero before the variance reduction. Stores are guarded by the same mask so out-of-range lanes never touch the output.</span>

<span style="font-size: 14px;">The same constexpr discipline applies to $\gamma$ and $\beta$: both are length $N$, both are loaded with the same $\texttt{cols} < N$ mask. They could in principle be cached across programs, but with one program per row and no cross-program reuse pattern available, each program simply reloads them, and the L2 absorbs the duplicate traffic for free.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">The row of $x$ is loaded once from HBM and held in registers for the rest of the program: it participates in the mean reduction, the centered subtraction, the centered-square variance reduction, and the final affine transform. That single load is the whole memory cost on the input side, $4N$ bytes per row. The output is written once, another $4N$ bytes. $\gamma$ and $\beta$ contribute $8N$ bytes per row in the worst case (no L2 reuse) or near zero in the best case (warm L2).</span>

<span style="font-size: 14px;">The critical reuse fact is **temporal**, not spatial: the tile of $x$ is reused throughout the program for the mean reduction, centering, centered-square variance reduction, and affine transform, so the kernel pays one HBM load per element of $x$. A non-fused chain would make several HBM round-trips for the same arithmetic. Fusion avoids those additional reads and writes while keeping the full calculation in one program.</span>

<span style="font-size: 14px;">There is no shared memory in the picture because the row reductions happen entirely inside one program's registers. $\texttt{tl.sum}$ collapses the tile to a scalar without spilling: internally the compiler emits a tree reduction across warps with a small SRAM staging area, but the author writes one symbol and gets the right code.</span>

<span style="font-size: 14px;">The reduction depth is the binary log of the tile width: a $\texttt{BLOCK\_SIZE} = 1024$ tile collapses to a scalar in $\log_2(1024) = 10$ tree levels. Each level halves the number of partial sums, and the compiler issues each level as a few instructions over the surviving lanes. The cost of the reduction is therefore logarithmic in the tile width, vanishingly small compared to the HBM load that fed the tile in the first place. Reductions are "free" on a memory-bound kernel precisely because the FLOP cost is dwarfed by the byte cost.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output element the kernel performs roughly $6$ to $8$ FLOPs (the two reductions amortize to two FMAs per element, plus a subtract, a multiply by $\texttt{rstd}$, a multiply by $\gamma$, and an add of $\beta$). It moves $4$ bytes of $x$ in, $4$ bytes of $\gamma$, $4$ bytes of $\beta$, and $4$ bytes of $\texttt{out}$ back to HBM, for a total of about $16$ bytes per element. Arithmetic intensity:</span>

$$
\frac{\approx 7 \text{ FLOPs}}{\approx 16 \text{ bytes}} \approx 0.4 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">That is hard **memory-bound** on every modern accelerator (the roofline crossover sits around $10$ FLOPs/byte). The only optimizations that move the runtime are the ones that reduce HBM traffic: the fusion above (which cuts $x$ reads from four to one), keeping $\gamma$ and $\beta$ in the L2 across programs, and avoiding any redundant write of intermediate normalized values.</span>

<span style="font-size: 14px;">A useful comparison: vector add sits at $\approx 0.08$ FLOPs/byte, fused softmax at $\approx 0.3$, LayerNorm at $\approx 0.4$. All three are memory-bound, all three live on the same vertical of the roofline, and all three are limited by HBM bandwidth rather than the FMA pipeline. The fact that LayerNorm has more arithmetic per element than vector add does not move it off the bandwidth ceiling; the work is just denser.</span>

<span style="font-size: 14px;">The practical consequence is that LayerNorm benchmarks track HBM bandwidth almost exactly. A well-written kernel on a $1$ TB/s memory subsystem processes about $1 / 16 \approx 60$ GB worth of output per second, or roughly $1.5 \cdot 10^{10}$ output elements per second. Any benchmark reporting substantially more is hitting the L2 cache (small enough tensors); any reporting substantially less has a memory-traffic bug, almost always an unfused intermediate.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author chooses:** the grid ($(M,)$, one program per row), the constexpr block size as $\texttt{next\_power\_of\_2}(N)$, the tail mask $\texttt{cols} < N$, the $\texttt{other}$ value on the initial load, the explicit centered-value mask, the stable variance reduction $E[(x - \mu)^2]$, and the use of $\texttt{tl.math.rsqrt}$ (or $1 / \texttt{tl.sqrt}$) for the inverse standard deviation.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering $\texttt{tl.sum}$ to a tree reduction across the tile, picking the vector width for the row load, allocating registers for the row tile, deciding how the tile is sharded across warps inside the program, and inserting any synchronization needed by the internal tree reduction. The author never writes a shuffle, never names a warp, never declares shared memory.</span>

<span style="font-size: 14px;">The important numerical choice is how variance is computed. The centered form $E[(x - \mu)^2]$ requires a reduction after the mean is known, but the original row is still in registers, so it does not require another HBM read. The alternative $E[x^2] - E[x]^2$ can suffer catastrophic cancellation when the two terms are large and close. The author must select the stable centered form because the compiler cannot repair an unstable formula.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive eager-PyTorch decomposition is roughly: one kernel to compute the mean ($\sum x$, then divide), one to compute the centered values ($x - \mu$), one to compute the variance ($\sum (x - \mu)^2$, then divide), one to compute the reciprocal standard deviation, one to normalize, and one to apply the affine. That is six kernel launches and six round-trips through HBM for the row $x$.</span>

<span style="font-size: 14px;">The fused Triton version reads $x$ once, reads $(\gamma, \beta)$ once, and writes $\texttt{out}$ once. HBM traffic on $x$ drops by $6\times$ relative to the naive chain. The arithmetic is unchanged (it has to do the same FLOPs), but because the kernel is memory-bound, dropping the HBM traffic translates almost directly into a $\sim 6\times$ speedup on the same hardware. The compiler-managed tree reduction inside $\texttt{tl.sum}$ is doing what a hand-written CUDA author would write as a warp shuffle followed by a block-level combine; the author gets that for free.</span>

<span style="font-size: 14px;">A further optimization, used in some production kernels, is to fuse the next layer's operation (typically a matmul) into the same kernel. That removes the round-trip on $\texttt{out}$ entirely. The standalone LayerNorm here is the canonical first step on that path.</span>

<span style="font-size: 14px;">A second axis of optimization is precision. Training pipelines run inputs in bf16 or fp16 but accumulate the mean and centered-square reductions in fp32. The Triton form expresses this naturally: load the lower-precision row, cast to fp32 immediately on entry to the kernel, do every reduction and the affine in fp32, and cast back only at the $\texttt{tl.store}$. The compiler emits the casts as cheap register operations; the HBM traffic is paid in the storage precision regardless. Accumulator precision matters here because each reduction can contain hundreds of contributions.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $M = 2$, $N = 4$, $\texttt{BLOCK\_SIZE} = 4$, $\epsilon = 10^{-5}$, $\gamma = [1, 1, 1, 1]$, $\beta = [0, 0, 0, 0]$, and the input row $x[0, :] = [1, 2, 3, 4]$.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{row} = 0$): the tile loads as $[1, 2, 3, 4]$ with all lanes in bounds. $\texttt{tl.sum}(x) = 10$, so the mean is $\mu = 10 / 4 = 2.5$. The centered values are $[-1.5, -0.5, 0.5, 1.5]$, whose squared sum is $5$. Therefore $\sigma^2 = 5 / 4 = 1.25$. The inverse standard deviation is $r_{\text{std}} = 1 / \sqrt{1.25 + 10^{-5}} \approx 0.8944$. The normalized row is $(x - 2.5) \cdot 0.8944 = [-1.342, -0.447, 0.447, 1.342]$. With unit $\gamma$ and zero $\beta$, that is also the output row.</span>

<span style="font-size: 14px;">If instead $N = 5$ and the tile were $\texttt{BLOCK\_SIZE} = 8$, the mask $\texttt{cols} < 5$ would be $[T, T, T, T, T, F, F, F]$. Lanes $5..7$ would load as $0$ and contribute $0$ to the mean reduction. After the mean is computed, $\texttt{tl.where}$ must zero those padded lanes again because subtracting the mean would otherwise turn them into $-\mu$. The divisor is always $N = 5$, not $\texttt{BLOCK\_SIZE} = 8$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Dividing by $\texttt{BLOCK\_SIZE}$ instead of $N$.** The block size is the compile-time tile width, not the runtime row length. Dividing the sums by $\texttt{BLOCK\_SIZE}$ silently produces wrong means and variances on every row whose width is not a power of two. The divisor must be $N$ in both $\mu$ and $\sigma^2$.</span>
* <span style="font-size: 14px;">**Masking only the initial row load.** The initial $\texttt{mask} = \texttt{cols} < N$ and $\texttt{other} = 0.0$ keep padded lanes neutral for the mean. After centering, those lanes become $-\mu$ unless they are explicitly zeroed with $\texttt{tl.where}$. Failing to mask centered values corrupts the variance whenever $N$ is not a power of two.</span>
* <span style="font-size: 14px;">**Skipping $\epsilon$ inside the square root.** Writing $r_{\text{std}} = 1 / \sqrt{\sigma^2}$ rather than $1 / \sqrt{\sigma^2 + \epsilon}$ produces a NaN whenever a row happens to be constant (zero variance), which is rare in real data but easy to trigger in tests.</span>
* <span style="font-size: 14px;">**Recomputing the affine in a second kernel.** A reasonable-looking decomposition that splits the normalize and the affine into two kernels doubles the HBM traffic on the output and gives back most of the win of fusion. The whole point of one program per row is that everything fits in registers; do not write intermediates to memory.</span>

---