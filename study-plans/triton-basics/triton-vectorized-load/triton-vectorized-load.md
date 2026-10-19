# <span style="font-size: 20px;">Vectorized Vector Add</span>

<span style="font-size: 14px;">Vectorized vector add is the smallest possible optimization study in the Triton track: same kernel as the foundational vector add, same masks, same load-add-store triplet, with a single number changed at launch time. The baseline used $\texttt{BLOCK\_SIZE} = 1024$. This kernel uses $\texttt{BLOCK\_SIZE} = 4096$. Each program now owns four times the data, the launch grid shrinks four-fold, and the compiler is free to emit wider memory transactions. The lesson is that **tile size is itself a performance knob**, not just a number that has to be a power of two: changing it shifts the launch-overhead vs register-pressure vs in-flight-load balance, and the right value depends on the kernel and the hardware.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">The operation is unchanged:</span>

$$
\texttt{out}[i] = x[i] + y[i], \quad 0 \le i < N
$$

<span style="font-size: 14px;">$x, y, \texttt{out} \in \mathbb{R}^N$, all contiguous fp32, all on the GPU. The kernel body is identical to a standard vector add: one program reads $\texttt{tl.program\_id(0)}$, builds offsets $\texttt{offs} = \text{pid} \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$, masks the tail with $\texttt{mask} = \texttt{offs} < N$, loads from $x$ and $y$, adds, stores into $\texttt{out}$.</span>

---

## <span style="font-size: 16px;">Baseline: BLOCK_SIZE = 1024</span>

<span style="font-size: 14px;">The standard vector-add kernel uses $\texttt{BLOCK\_SIZE} = 1024$. The choice is a balance: $1024$ is a power of two so the compiler can vectorize the load cleanly; it is large enough that the per-program launch overhead is amortized over a thousand additions; and the per-program register footprint is small enough that the kernel runs at high occupancy on every modern accelerator. For a one-million-element add the launch grid is $\lceil 10^6 / 1024 \rceil = 977$ programs, more than enough to fill all the streaming multiprocessors several times over.</span>

<span style="font-size: 14px;">The baseline gets close to peak HBM bandwidth on long arrays. The exact achieved fraction depends on the hardware, but $80\%$ of peak is a reasonable rule of thumb. The remaining headroom comes from two places: the per-program fixed cost (dispatch, register setup, the mask check on every load) and the width of the memory transactions the compiler emits. The first is amortized by reducing the program count; the second is helped by giving the compiler more contiguous indices per load.</span>

---

## <span style="font-size: 16px;">The Change: BLOCK_SIZE = 4096</span>

<span style="font-size: 14px;">Set $\texttt{BLOCK\_SIZE} = 4096$, update the grid formula to $\lceil N / 4096 \rceil$, and leave everything else alone. For the same one-million-element add, the grid is now $\lceil 10^6 / 4096 \rceil = 245$ programs. Each program owns four times as much data and walks through it in registers across a load-add-store sequence the compiler is free to unroll.</span>

<span style="font-size: 14px;">Two things change in the generated code. The compiler sees a $4096$-element contiguous offset tile and can fold the load into wider PTX vector instructions. On Ampere and later, the native vector width tops out at $128$ bits ($\text{ld.global.v4.f32}$, four fp32 lanes per instruction), so a $4096$-element load is decomposed into $1024$ four-lane instructions across the warps of the program. The baseline $1024$-element load is decomposed into $256$ such instructions per program. The throughput-per-instruction is the same; what changes is how many independent loads the warp scheduler has to choose from.</span>

<span style="font-size: 14px;">More independent loads in flight per program lets the warp scheduler overlap memory latency with itself. While one warp's load is stalled waiting on HBM, the scheduler issues another warp's load, then another, then arithmetic on a tile whose load has already returned. The latency-hiding mechanism does not depend on $\texttt{BLOCK\_SIZE}$ being big, but it gets more headroom as $\texttt{BLOCK\_SIZE}$ grows, because the per-program pool of independent operations grows.</span>

---

## <span style="font-size: 16px;">Where the Speedup Comes From</span>

<span style="font-size: 14px;">Three terms move when $\texttt{BLOCK\_SIZE}$ goes from $1024$ to $4096$:</span>

<span style="font-size: 14px;">1. **Launch overhead per element** drops by $4\times$. On a modern GPU, kernel launch overhead is on the order of $5\!-\!10$ microseconds per program, and program startup (register allocation, mask setup, the first instruction issue) is a few hundred cycles. For a $10^6$-element add, $977$ vs $245$ programs is a difference of roughly $700$ program startups not paid, which at $1$ TB/s of bandwidth corresponds to a noticeable fraction of the total kernel time.</span>

<span style="font-size: 14px;">2. **In-flight loads per program** scale linearly with $\texttt{BLOCK\_SIZE}$. The warp scheduler has more independent work to issue, which hides HBM latency more effectively. The effect saturates once the program has enough in-flight loads to cover the round-trip time to HBM (a few hundred cycles, or a few dozen independent loads on current hardware), so the improvement from $1024$ to $4096$ is real but bounded.</span>

<span style="font-size: 14px;">3. **Mask amortization** improves. The mask is built once per program and applied on every load and store in that program. With $4\times$ more data per program, the mask construction cost (one $\texttt{tl.arange}$ plus one comparison) is paid one quarter as often per element. For shapes that are multiples of $\texttt{BLOCK\_SIZE}$, the compiler may even prove the mask is always true and elide it entirely.</span>

<span style="font-size: 14px;">The net effect on a $1$M-element fp32 add is in the $5\!-\!15\%$ range on most modern accelerators. The kernel is already running near peak HBM bandwidth at $\texttt{BLOCK\_SIZE} = 1024$, so the improvement is the last few percentage points to the roofline rather than a multiplier on throughput.</span>

---

## <span style="font-size: 16px;">The Tradeoff: Register Pressure and Occupancy</span>

<span style="font-size: 14px;">A bigger tile is not free. The compiler has to hold the full $\texttt{BLOCK\_SIZE}$-wide tile in registers across the load-add-store sequence, and at fp32 that is $4 \cdot \texttt{BLOCK\_SIZE} = 16{,}384$ bytes per program for a single live tile. For vector add the live tiles are $x$, $y$, and briefly the sum, so the working set is a couple of times that. Total per-program register usage scales linearly with $\texttt{BLOCK\_SIZE}$.</span>

<span style="font-size: 14px;">Register usage matters because it bounds occupancy. Each streaming multiprocessor has a fixed register file of around $64$K registers in total, shared among all the programs assigned to it. A program that needs more registers gets fewer co-resident neighbors, and the warp scheduler has fewer warps to swap between when latency stalls hit. For vector add this is rarely a binding constraint because the per-element work is small and the compiler does not need many scratch registers, but for hotter kernels (with larger accumulators, more state per element, or fp32 intermediates around lower-precision inputs) pushing $\texttt{BLOCK\_SIZE}$ to $4096$ can drop the achievable $\texttt{num\_warps}$ and lower occupancy.</span>

<span style="font-size: 14px;">Pushing $\texttt{BLOCK\_SIZE}$ past $4096$ on a register-heavy kernel runs into spills to local memory, which is global memory the compiler uses as overflow storage for the register file. Local-memory spills are several orders of magnitude slower than register accesses, and throughput collapses faster than the per-program overhead is recovered. The practical ceiling for $\texttt{BLOCK\_SIZE}$ on a register-heavy kernel is a few thousand; for a register-light kernel like vector add, $4096$ or even $8192$ is fine.</span>

---

## <span style="font-size: 16px;">When the Optimization Helps</span>

<span style="font-size: 14px;">The win from a larger $\texttt{BLOCK\_SIZE}$ is conditional on three things. First, the array must be long enough that the grid actually shrinks: for $N < 4096$, both versions launch a single program and most of the block is masked off either way. The optimization matters only when $N$ is in the tens of thousands or larger. Second, the kernel must be memory-bound (which vector add definitely is); for compute-bound kernels, the per-program work scales with $\texttt{BLOCK\_SIZE}$ and the launch-overhead savings are a smaller fraction of total time. Third, the register pressure has to leave room: a kernel that is already pushing register limits at $\texttt{BLOCK\_SIZE} = 1024$ gets worse at $4096$, not better.</span>

<span style="font-size: 14px;">Vector add satisfies all three. It is the cleanest case for showing that block-size tuning is a real lever; for more complex kernels the same intuition applies but the optimal value lives further inside an autotune search rather than at a single fixed point.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Vector add stays firmly memory-bound at any block size. The arithmetic intensity is</span>

$$
\frac{1 \text{ FLOP}}{12 \text{ bytes}} \approx 0.083 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">which is orders of magnitude below the roofline crossover on every modern accelerator. The kernel is bound by how fast HBM can deliver bytes; arithmetic effectively costs nothing. The block-size change does not move the kernel across the roofline. What it does is raise the achieved fraction of peak bandwidth by reducing constant overheads and letting the warp scheduler hide latency more effectively. Both versions of the kernel run in HBM-throughput territory; the optimized version sits a few percent closer to the ceiling.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Compiler handles:** picking the PTX vector width from the contiguous-offset tile, allocating registers for the tile, sharding the tile across the warps of one program (the default $\texttt{num\_warps} = 4$ gives each warp $\texttt{BLOCK\_SIZE} / 4 / 32 = 32$ lanes per warp at $\texttt{BLOCK\_SIZE} = 4096$), scheduling the load-add-store sequence, and emitting any pipelining. None of this is different from the baseline kernel; the compiler just has a larger tile to work with.</span>

<span style="font-size: 14px;">**Author handles:** the constexpr block-size value, the matching grid formula at the launch site, and the awareness that $\texttt{BLOCK\_SIZE}$ is bounded above by register pressure on the kernel's working set. The block size must remain a power of two so the compiler can vectorize; non-power-of-two values force scalar fallback for partial vectors.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 10{,}000$ and compare $\texttt{BLOCK\_SIZE} = 1024$ vs $4096$.</span>

<span style="font-size: 14px;">**Baseline grid**: $\lceil 10000 / 1024 \rceil = 10$ programs. Programs $0\!-\!8$ are fully in-bounds. Program $9$ has $\texttt{offs} = 9216 + [0, 1, \dots, 1023]$, of which the last $240$ lanes ($\texttt{offs} \ge 10000$) are masked off. Per-program work: $1024$ loads, $1024$ adds, $1024$ stores. Total program work: $10240$ load-add-store triples scheduled, $240$ of them no-ops.</span>

<span style="font-size: 14px;">**Vectorized grid**: $\lceil 10000 / 4096 \rceil = 3$ programs. Programs $0$ and $1$ are fully in-bounds. Program $2$ has $\texttt{offs} = 8192 + [0, 1, \dots, 4095]$, of which the last $2288$ lanes are masked off. Per-program work: $4096$ load-add-store triples scheduled, $2288$ of them no-ops in the last program.</span>

<span style="font-size: 14px;">The wasted-lane count is roughly the same in both versions ($240$ vs $2288$ at the very tail), but the program count drops from $10$ to $3$. On a small-array case like this the launch overhead is a meaningful fraction of total time, and the larger block wins on overhead. On long arrays the wasted-lane count is negligible in both versions and the win comes from the in-flight-load and mask-amortization effects described above.</span>

<span style="font-size: 14px;">Scaling the same example to $N = 10^6$: the baseline launches $977$ programs and the vectorized version launches $245$. Only the last program in each version has a non-trivial mask; the other $976$ or $244$ programs run with the mask fully true. At a typical HBM bandwidth of $1$ TB/s, the kernel moves $12$ MB of traffic and takes $\approx 12$ microseconds at peak. The launch overhead at $977$ programs is a few microseconds in total; at $245$ programs it is a fraction of that. The visible speedup tracks how much of the total time was launch-bound to begin with.</span>

---

## <span style="font-size: 16px;">Comparing With Autotuning</span>

<span style="font-size: 14px;">Fixing $\texttt{BLOCK\_SIZE} = 4096$ as a hard-coded constant is the simplest possible response to the tuning question. The next step up is wrapping the kernel with $\texttt{@triton.autotune}$ over a set of candidate block sizes (and $\texttt{num\_warps}$, $\texttt{num\_stages}$ where applicable), letting the compiler benchmark each configuration on the first call with a given input shape and remember the winner. For vector add the autotune space is small and the win over a sensible hard-coded value is marginal; for hotter kernels like matmul the autotune approach is mandatory because the optimal block configuration shifts with the input shape and is not learnable from inspection alone.</span>

<span style="font-size: 14px;">The optimization in this problem sits in the middle: it commits to one specific value of $\texttt{BLOCK\_SIZE}$ and asks the reader to understand why that value beats the baseline on long arrays. The honest framing is that this is a baseline tuning lesson, not a universal recommendation; for production code the autotune wrapper is the more general answer.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**$\texttt{BLOCK\_SIZE}$ past register limits.** Pushing past $4096\!-\!8192$ on a register-heavy kernel causes spills to local memory and throughput collapses. The first sign of a spill is a sudden drop in achieved bandwidth that does not recover by tuning $\texttt{num\_warps}$; check the compiler's register-usage report when the change is suspect.</span>

* <span style="font-size: 14px;">**Grid formula not updated.** Common bug: the launcher hard-codes $((n + 1023) // 1024,)$ in one place and $\texttt{BLOCK\_SIZE} = 4096$ in another. The grid is sized for the old block, the kernel reads past the end of the array, and the masks on the bad programs cannot save it because $\texttt{offs}$ values are computed assuming the new block size.</span>

* <span style="font-size: 14px;">**$\texttt{BLOCK\_SIZE}$ passed as a runtime int.** Without $\texttt{tl.constexpr}$, the compiler cannot size registers or unroll the load-add-store sequence and falls back to scalar codegen. The kernel runs but loses most of its speed.</span>

* <span style="font-size: 14px;">**Treating the optimization as a free win on every kernel.** Big $\texttt{BLOCK\_SIZE}$ helps memory-bound kernels with light register footprints on long arrays. For compute-bound or register-heavy kernels, or for short arrays, the same change can hurt; autotuning is the right way to pick block sizes in those cases.</span>

---