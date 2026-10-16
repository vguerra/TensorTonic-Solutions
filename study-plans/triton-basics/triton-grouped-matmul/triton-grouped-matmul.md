# <span style="font-size: 20px;">Grouped Program-ID Matmul</span>

<span style="font-size: 14px;">Grouped-pid matmul is the same tiled GEMM the matmul section already built, with one surgical change to the launch grid: collapse the 2D grid to 1D, then derive $(\text{pid\_m}, \text{pid\_n})$ from the linear $\text{pid}$ with a $\texttt{GROUP\_SIZE\_M}$ remap that walks the output in M-major super-blocks. The arithmetic does not move, but the order in which the hardware visits output tiles does, and that order is what determines L2 hit rate on the $A$ and $B$ operands. The lesson is that **program order is a knob**, and the right remap turns the L2 cache from a coincidence into a structural part of the kernel's bandwidth budget.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">The numerical contract is unchanged from the baseline matmul:</span>

$$
C[i, j] = \sum_{k=0}^{K-1} A[i, k] \cdot B[k, j]
$$

<span style="font-size: 14px;">$A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, $C \in \mathbb{R}^{M \times N}$, all row-major fp32. Each output tile is still $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N}) = (64, 64)$ and the inner $K$-reduction is still $\texttt{BLOCK\_K} = 32$. The only thing this problem rewrites is which program ID gets which output tile.</span>

---

## <span style="font-size: 16px;">Baseline: 2D Grid, Hardware-Chosen Order</span>

<span style="font-size: 14px;">In the baseline kernel the launch grid is two-dimensional: $\text{grid} = (\lceil M / \texttt{BLOCK\_M} \rceil, \lceil N / \texttt{BLOCK\_N} \rceil)$ with $\texttt{pid\_m} = \texttt{tl.program\_id(0)}$ and $\texttt{pid\_n} = \texttt{tl.program\_id(1)}$. The hardware scheduler is free to walk those two axes in any order it likes, and on current GPUs it walks them in a row-major pattern that touches all $\texttt{num\_pid\_n}$ programs in row $\texttt{pid\_m} = 0$ before moving to row $\texttt{pid\_m} = 1$.</span>

<span style="font-size: 14px;">That order is the worst case for L2 reuse on the $A$ side. Two adjacent programs in execution order differ in $\texttt{pid\_n}$, which means they share the same row-block of $A$ but want completely different column-blocks of $B$. The shared $A$ row-block does see some L2 reuse, but the $A$ side is small to begin with; the bulk of the bandwidth on a typical tall matmul is $B$, and $B$ tiles get loaded fresh on every program. After one row of programs finishes, the schedule moves to the next $\texttt{pid\_m}$, where it now re-streams the entire $B$ matrix because every $B$ column-block was evicted while $A$'s row swept across them.</span>

---

## <span style="font-size: 16px;">The Change: 1D Grid with Grouped Remap</span>

<span style="font-size: 14px;">Collapse the grid to one axis of total size $\texttt{num\_pid\_m} \cdot \texttt{num\_pid\_n}$ and decode the program ID in the kernel. The decode interprets $\text{pid}$ as a position inside a $\texttt{GROUP\_SIZE\_M} \times \texttt{num\_pid\_n}$ super-block, M-major within the super-block. Concretely:</span>

<span style="font-size: 14px;">1. **Locate the super-block**: $\text{group\_id} = \text{pid} / (\texttt{GROUP\_SIZE\_M} \cdot \texttt{num\_pid\_n})$.</span>

<span style="font-size: 14px;">2. **Find the first M-row of this super-block**: $\text{first\_pid\_m} = \text{group\_id} \cdot \texttt{GROUP\_SIZE\_M}$, then clamp the effective group width with $\text{group\_size\_m} = \min(\texttt{num\_pid\_m} - \text{first\_pid\_m}, \texttt{GROUP\_SIZE\_M})$.</span>

<span style="font-size: 14px;">3. **Decode $(\text{pid\_m}, \text{pid\_n})$**: $\text{pid\_m} = \text{first\_pid\_m} + (\text{pid} \bmod \text{group\_size\_m})$ and $\text{pid\_n} = (\text{pid} \bmod (\texttt{GROUP\_SIZE\_M} \cdot \texttt{num\_pid\_n})) / \text{group\_size\_m}$.</span>

<span style="font-size: 14px;">After this remap, incrementing $\text{pid}$ by one inside a super-block holds $\text{pid\_n}$ fixed and advances $\text{pid\_m}$. The schedule walks $\texttt{GROUP\_SIZE\_M}$ consecutive M-rows of the output for the same $B$ column-block, then advances $\text{pid\_n}$ by one and walks the same $\texttt{GROUP\_SIZE\_M}$ M-rows for the next $B$ column-block.</span>

---

## <span style="font-size: 16px;">L2 Reuse: Where the Payoff Lives</span>

<span style="font-size: 14px;">L2 is the cache that adjacent-in-time programs implicitly share. When $\texttt{GROUP\_SIZE\_M} = 8$ consecutive programs all hit the same $B$ column-block, the first program pays the HBM cost to bring that $(\texttt{BLOCK\_K}, \texttt{BLOCK\_N}) = (32, 64)$ slab through L2 once, and the next seven programs find it warm. Across the inner $K$-loop, that single $B$ slab is touched $\lceil K / \texttt{BLOCK\_K} \rceil$ times by each of the eight programs, and L2 absorbs all but the first set of loads. Per super-block, $B$-side HBM traffic drops by roughly a factor of $\texttt{GROUP\_SIZE\_M}$.</span>

<span style="font-size: 14px;">The $A$ side gets a smaller, symmetric win. Inside one super-block, the eight programs walk eight different M-rows of $A$ but share a single $B$ column-block. Once the super-block advances to the next $\texttt{pid\_n}$, the schedule revisits the same eight $A$ row-blocks against a fresh $B$ slab, and the L2 captures whatever still fits. The net pattern is that $B$ benefits across-program (within a super-block) and $A$ benefits across-super-block (within a row of super-blocks).</span>

<span style="font-size: 14px;">A useful sanity check: for a square $M = N = K = 1024$ matmul with $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$, $\texttt{BLOCK\_K} = 32$, the baseline schedule reads each $B$ column-block once per M-row of the output, for $16$ reads. With $\texttt{GROUP\_SIZE\_M} = 8$ the same column-block is reused eight times in a row before the next one is touched, and the inner-loop $B$ load stream is essentially L2-resident across the eight programs. That is the entire optimization, expressed as a remap of the integer $\text{pid}$.</span>

<span style="font-size: 14px;">The reason the win has to come from L2 and not from the kernel itself is that each program already does the optimal amount of register-level reuse on its own tile: the inner $K$-loop loads each $A$ tile of shape $(\texttt{BLOCK\_M}, \texttt{BLOCK\_K})$ once and reuses it across all $\texttt{BLOCK\_N}$ columns of the output via $\texttt{tl.dot}$. The compiler stages those tiles into shared memory automatically. The only remaining traffic is the across-program kind, where two different programs ask HBM for the same operand region, and that is where L2 sits in the hierarchy. Grouped scheduling is the trick that turns redundant across-program loads into L2 hits.</span>

---

## <span style="font-size: 16px;">Tile Shapes and Masking</span>

<span style="font-size: 14px;">Nothing about the tile shapes changes. Each program still owns one $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N})$ output tile held in an fp32 accumulator and streams $A$ and $B$ tiles of shape $(\texttt{BLOCK\_M}, \texttt{BLOCK\_K})$ and $(\texttt{BLOCK\_K}, \texttt{BLOCK\_N})$ through the inner loop. The masks are unchanged: $\texttt{offs\_m} < M$ on the $A$ load and the $C$ store, $\texttt{offs\_n} < N$ on the $B$ load and the $C$ store, and $(\text{k} + \texttt{offs\_k}) < K$ on the inner $K$-loop bound. Mask discipline is the price of compile-time block sizes meeting runtime dimensions, and grouped scheduling does not relieve it.</span>

<span style="font-size: 14px;">The clamp on the partial last super-block is the one new piece of correctness machinery. When $\texttt{num\_pid\_m}$ is not a multiple of $\texttt{GROUP\_SIZE\_M}$, the last super-block has fewer than $\texttt{GROUP\_SIZE\_M}$ M-rows. The $\text{group\_size\_m}$ minimum keeps the decoded $\text{pid\_m}$ inside $[0, \texttt{num\_pid\_m})$. Without the clamp, the last super-block produces $\text{pid\_m}$ values past the grid, the M-mask hides the bad stores, but each rogue program still walks the inner $K$-loop and burns time.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Tiled matmul sits on the compute-bound side of the roofline for any reasonable block size. The arithmetic intensity of one $\texttt{tl.dot}$ is</span>

$$
\frac{2 \cdot \texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N} \cdot \texttt{BLOCK\_K}}{4 \cdot (\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_K} + \texttt{BLOCK\_K} \cdot \texttt{BLOCK\_N})}
$$

<span style="font-size: 14px;">which simplifies to roughly $\texttt{BLOCK\_K} / 4$ FLOPs per byte for square tiles. At $\texttt{BLOCK\_K} = 32$ that is $\approx 8$ FLOPs per byte, comfortably across the crossover point on modern accelerators. The kernel is bound by tensor-core throughput, not HBM bandwidth, provided HBM can supply tiles fast enough to keep the cores fed. Grouped scheduling is exactly the trick that keeps HBM out of the critical path: by absorbing the inner-loop $B$ loads into L2, it removes the only term that would have pushed the kernel back into memory-bound territory for skinny or oddly-sized inputs.</span>

---

## <span style="font-size: 16px;">Choosing GROUP_SIZE_M</span>

<span style="font-size: 14px;">$\texttt{GROUP\_SIZE\_M}$ is the only new knob. Eight is the canonical value used by Triton's own matmul tutorial and works on essentially every modern accelerator without retuning. The reasoning is straightforward: the L2 cache holds tens of megabytes on contemporary GPUs, and one super-block's working set is roughly $\texttt{GROUP\_SIZE\_M} \cdot \texttt{BLOCK\_M} \cdot K + \texttt{BLOCK\_N} \cdot K$ bytes of operand data for $A$ and $B$. With $\texttt{GROUP\_SIZE\_M} = 8$, $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$, and a reasonable $K$, that working set is comfortably inside L2 on any current device.</span>

<span style="font-size: 14px;">Pushing $\texttt{GROUP\_SIZE\_M}$ higher gives diminishing returns: the L2 win plateaus once the super-block fully fits in cache, and pushing past that point starts evicting earlier programs' tiles before the schedule wraps around. Pushing it lower (down to $1$) collapses the grouped schedule back into a straight column-major walk and loses most of the reuse. Inputs where one dimension is much smaller than the other can occasionally benefit from a different value, but tuning $\texttt{GROUP\_SIZE\_M}$ is a small effect compared to picking the right tile shape, and is not usually included in autotune configs.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Compiler handles:** the lowering of $\texttt{tl.dot}$ to tensor-core MMA instructions, the staging of $A$ and $B$ tiles into shared memory inside the dot epilogue, the swizzling that matches the MMA fragment layout, and the software pipelining that overlaps the next $K$-tile load with the current accumulate. None of these change when the grid switches from 2D to 1D.</span>

<span style="font-size: 14px;">**Author handles:** the integer math of the remap, the choice of $\texttt{GROUP\_SIZE\_M}$ (eight is the standard value from the official matmul tutorial; smaller values give less L2 benefit, larger values increase the chance that a super-block's working set spills out of L2), the declaration of $\texttt{GROUP\_SIZE\_M}$ as $\texttt{tl.constexpr}$ so the modulo and division compile to constants, and the partial-group clamp. The compiler cannot infer the schedule from the kernel body; the schedule is a property of the integer the kernel reads from $\texttt{tl.program\_id(0)}$.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $M = 256$, $N = 256$, $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$, $\texttt{GROUP\_SIZE\_M} = 4$. Then $\texttt{num\_pid\_m} = \texttt{num\_pid\_n} = 4$ and the total grid is $16$ programs. The super-block is $4 \times 4 = 16$ tiles, so the entire output is one super-block in this small example.</span>

<span style="font-size: 14px;">Programs in execution order:</span>

<span style="font-size: 14px;">1. $\text{pid} = 0 \to (\text{pid\_m}, \text{pid\_n}) = (0, 0)$, $\text{pid} = 1 \to (1, 0)$, $\text{pid} = 2 \to (2, 0)$, $\text{pid} = 3 \to (3, 0)$. All four share $B$'s first column-block.</span>

<span style="font-size: 14px;">2. $\text{pid} = 4 \to (0, 1)$, $\text{pid} = 5 \to (1, 1)$, $\text{pid} = 6 \to (2, 1)$, $\text{pid} = 7 \to (3, 1)$. All four share $B$'s second column-block.</span>

<span style="font-size: 14px;">And so on for $\text{pid} = 8 \dots 15$. In the baseline 2D schedule, the order would have been $(0,0), (0,1), (0,2), (0,3), (1,0), \dots$ and the four $B$ column-blocks would have been swept through once per M-row, eight more times than they need to be loaded.</span>

<span style="font-size: 14px;">Counting HBM traffic for $B$: each column-block has size $K \cdot \texttt{BLOCK\_N} \cdot 4$ bytes. In the baseline schedule, each block is loaded $\texttt{num\_pid\_m} = 4$ times across the matmul (once per M-row). In the grouped schedule, each block is loaded just once per super-block, and the super-block contains $\texttt{GROUP\_SIZE\_M} = 4$ M-rows. The factor of $4$ disappears into L2.</span>

<span style="font-size: 14px;">Scaling the same arithmetic up to $M = N = 1024$ with $\texttt{GROUP\_SIZE\_M} = 8$: $\texttt{num\_pid\_m} = \texttt{num\_pid\_n} = 16$, the grid has $256$ programs in $32$ super-blocks (each $8 \times 16$). Per super-block, the eight M-rows share each of the $16$ $B$ column-blocks, so the $B$-side L2 traffic is the equivalent of $16$ first-touch loads per super-block instead of $8 \cdot 16 = 128$ raw HBM loads. The $8\times$ reuse factor is the headline number on every grouped-matmul benchmark.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Skipping the partial-group clamp.** Without $\min(\texttt{num\_pid\_m} - \text{first\_pid\_m}, \texttt{GROUP\_SIZE\_M})$, the last super-block produces $\text{pid\_m}$ values past $\texttt{num\_pid\_m} - 1$. The M-mask hides the bad stores, so tests pass, but each rogue program still loops over the full $K$ dimension and burns the time it was supposed to save.</span>

* <span style="font-size: 14px;">**Writing $\text{pid\_m} = \text{pid} / \texttt{num\_pid\_n}$ and $\text{pid\_n} = \text{pid} \bmod \texttt{num\_pid\_n}$.** This is the naive row-major decode and silently runs the same schedule as the 2D grid. The kernel still produces the correct result, the L2 win evaporates, and benchmarks look identical to the baseline.</span>

* <span style="font-size: 14px;">**Sizing the grid as $M \cdot N$ instead of $\texttt{num\_pid\_m} \cdot \texttt{num\_pid\_n}$.** A common typo: the launcher spawns $M \cdot N$ programs (one per output element) rather than one per output tile, overspawning by a factor of $\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N} = 4096$.</span>

* <span style="font-size: 14px;">**$\texttt{GROUP\_SIZE\_M}$ passed as a runtime int.** Without $\texttt{tl.constexpr}$, the divisions and moduli in the decode become runtime ops and the compiler cannot fold them. The decode block also forces a recompile per distinct value.</span>

---