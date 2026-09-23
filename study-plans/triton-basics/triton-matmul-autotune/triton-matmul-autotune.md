# <span style="font-size: 20px;">Autotuned Matrix Multiplication</span>

<span style="font-size: 14px;">The same tiled matmul kernel runs an order of magnitude apart depending on the block shape and pipeline depth, and the optimal choice shifts with the input dimensions. A $(64, 64, 32)$ tile that is excellent on a $(4096, 4096, 4096)$ matmul leaves performance on the floor for a $(128, 8192, 8192)$ skinny-tall problem, and vice versa. **Autotune** is Triton's mechanism for sweeping a small config space at first launch, benchmarking each candidate on the actual inputs, and caching the winner keyed by problem shape. The kernel body is unchanged; the entire optimization is in the decorator.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, the kernel computes:</span>

$$
C[i, j] = \sum_{k=0}^{K-1} A[i, k] \cdot B[k, j]
$$

<span style="font-size: 14px;">Identical to the plain tiled matmul: same masks, same fp32 accumulator, same $\texttt{tl.dot}$ inner step. The kernel is decorated with $\texttt{@triton.autotune}$ over a small set of configs that vary $\texttt{BLOCK\_M}, \texttt{BLOCK\_N}, \texttt{BLOCK\_K}, \texttt{num\_warps}, \texttt{num\_stages}$, and the grid is expressed as a callable so the launcher can resolve it after the winner is picked.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is two-dimensional, $\lceil M / \texttt{BLOCK\_M} \rceil \times \lceil N / \texttt{BLOCK\_N} \rceil$, with each **program** owning a $\texttt{BLOCK\_M} \times \texttt{BLOCK\_N}$ output tile. The autotuner does not change the parallel decomposition; it changes which $\texttt{BLOCK\_M}$ and $\texttt{BLOCK\_N}$ values fill the grid for a given shape, and therefore how many programs end up in flight.</span>

<span style="font-size: 14px;">Because the grid dimensions depend on values that are only known after the autotuner picks a config, the launcher passes a **callable grid**: a lambda that takes the resolved meta-dictionary and returns the grid tuple. Triton invokes the lambda after running its benchmark sweep, with the winning config's $\texttt{BLOCK\_M}$ and $\texttt{BLOCK\_N}$ substituted in. The author never has to read those values explicitly; the grid is a function of the meta-dict, not a closed-over Python value.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The tile-shape constexprs are no longer fixed at the kernel call site. Each $\texttt{triton.Config}$ in the autotune list pins a different $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N}, \texttt{BLOCK\_K})$ triple along with a $\texttt{num\_warps}$ and a $\texttt{num\_stages}$. A representative space for the harness is three configs: $(64, 64, 32)$ for a balanced square workload, $(128, 64, 32)$ for tall-skinny outputs that benefit from a wider $M$ tile, and $(64, 128, 32)$ for the symmetric short-fat case. Five configs is the upper end of what fits inside the harness's per-test subprocess timeout, because the first call with a new shape pays the full compile-and-benchmark cost for every candidate.</span>

<span style="font-size: 14px;">Mask discipline is identical to the plain matmul: the $K$-loop mask $(k + \texttt{offs\_k}) < K$ guards the $A$ and $B$ loads with $\texttt{other} = 0.0$, and the store mask $(\texttt{offs\_m}[:, \texttt{None}] < M) \,\&\, (\texttt{offs\_n}[\texttt{None}, :] < N)$ guards the write into $C$. Switching configs does not change which lanes are masked, only how many of them belong to one tile. A correctness bug in the mask shows up regardless of which config the autotuner picks.</span>

<span style="font-size: 14px;">A subtle interaction worth naming: configs whose $\texttt{BLOCK\_K}$ exceeds the runtime $K$ are still valid, because the per-iteration $K$ mask zeros the out-of-range columns. They run one $K$-loop iteration that touches the whole problem and store the result. Configs where $\texttt{BLOCK\_M}$ or $\texttt{BLOCK\_N}$ exceeds $M$ or $N$ are equally valid because the store mask discards the overshoot. The autotuner will measure these configs alongside the better-fit ones and rank them by wall time, so the mask discipline that makes the kernel correct in the plain case also makes the autotune search space robust to wildly mismatched block shapes.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">The memory pattern matches the plain tiled matmul: HBM holds $A$, $B$, $C$; SRAM stages the $\texttt{tl.dot}$ operands; registers hold the fp32 accumulator across the $K$-loop. What autotune changes is the **size** of each region's resident tile. A larger $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N})$ raises register footprint per program and pushes more programs into the same SM cluster; a larger $\texttt{BLOCK\_K}$ raises SRAM footprint for staging and lengthens the per-iteration MMA work; a larger $\texttt{num\_stages}$ multiplies the SRAM footprint by the pipeline depth in exchange for better HBM-latency hiding.</span>

<span style="font-size: 14px;">Reuse factors scale with the block dimensions: each $A$ slab loaded inside one iteration is used $\texttt{BLOCK\_N}$ times, each $B$ slab is used $\texttt{BLOCK\_M}$ times. A config with $(128, 64)$ blocks reuses each $A$ element $64$ times and each $B$ element $128$ times, asymmetric in a way that favors HBM bandwidth on $B$ at the cost of $A$. The autotuner does not reason about this analytically; it runs each config and measures end-to-end time, so the winner is the one whose tradeoff best matches the actual hardware's bandwidth-versus-compute balance for the actual shape.</span>

<span style="font-size: 14px;">Register pressure is the binding constraint at the top end of the search space. A $(128, 128)$ accumulator tile of fp32 occupies $128 \cdot 128 \cdot 4 = 65{,}536$ bytes per program, which is on the order of the entire register file of a modern SM. Configs that overshoot the register budget compile but spill to local memory, collapsing throughput. The autotuner observes the collapse as a measured slowdown and ranks the offending config last; the author does not need to compute the spill threshold by hand.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Tiled matmul is compute-bound for any reasonable block size on modern accelerators (arithmetic intensity $\frac{\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N}}{2(\texttt{BLOCK\_M} + \texttt{BLOCK\_N})}$ FLOPs per byte, which evaluates to $16$ at $(64, 64)$ and grows larger as the tile grows). Autotune does not move the kernel across the roofline; it moves it closer to the compute ceiling by choosing configs that maximize tensor-core utilization for the specific $(M, N, K)$ shape.</span>

<span style="font-size: 14px;">The shape sensitivity comes from two effects. **Tile coverage** matters when the matrix is small enough that the number of tiles is comparable to the SM count: a $(64, 64)$ block on a $(128, 128)$ output gives $4$ tiles, far too few to saturate a modern GPU with $100\!+\!$ SMs, so a $(32, 32)$ block that produces $16$ tiles wins despite worse per-tile reuse. **Wave quantization** matters when the tile count is just over a multiple of the SM count: an extra wave of programs hides behind the last wave's tail, and the winner is the config whose tile count is closest to a clean multiple. The autotuner discovers both effects by measuring; the author does not have to predict them.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">$\texttt{@triton.autotune}$ takes a list of $\texttt{triton.Config}$ objects and a $\texttt{key}$ argument naming the runtime parameters that determine the optimal config. The compiler compiles every config in the list on the first call with a given key tuple, runs each one on the actual inputs, and caches the winner. Subsequent calls with the same key skip the search and dispatch the cached kernel. The cache lives in process memory and is per-process; a fresh Python process repeats the full sweep.</span>

<span style="font-size: 14px;">The author chooses the search space, which is the only substantive decision. Three considerations dominate. The space must be **small enough** to fit inside the user's compile-and-benchmark budget (three to five entries for an interactive harness, twenty or more for a one-time precompilation step). The space must be **diverse enough** to cover the geometry the kernel will see in deployment, including different aspect ratios and at least one config with a larger $\texttt{num\_stages}$ for the latency-bound case. The $\texttt{key}$ must capture every runtime parameter the optimal config depends on; for matmul that is $(M, N, K)$, while for a kernel whose work also depends on a stride or a dtype the key would include those too.</span>

<span style="font-size: 14px;">$\texttt{num\_warps}$ and $\texttt{num\_stages}$ are autotune knobs the compiler exposes for tuning. $\texttt{num\_warps}$ controls how the compiler shards each tile across warps inside the program: more warps spread the tile thinner per warp and raise occupancy, fewer warps keep the tile concentrated and reduce intra-program coordination. $\texttt{num\_stages}$ controls the **software pipeline** depth on the $K$-loop: at depth $s$ the compiler issues the next $s - 1$ iterations' loads ahead of the current iteration's $\texttt{tl.dot}$, double- or triple-buffering the SRAM staging area to hide HBM latency. Both knobs trade resources (registers, SRAM) for latency hiding, and the right balance is shape-dependent.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">"Naive" here means a single hand-picked config that the author guessed from a single benchmark on a single shape. The autotuned kernel does measurably better whenever the deployed shapes diverge from the one the guess was based on, and the cost is concentrated at the first call rather than amortized across every launch. For a workload that hits a handful of stable shapes (a transformer with fixed batch and hidden size, for example), the cache hit rate after warmup is essentially $1$ and the per-launch autotune cost is zero.</span>

<span style="font-size: 14px;">Beyond a basic autotune, two further mechanisms apply. **Persistent autotune caches** ($\texttt{cache\_results}$ or environment variables that pin a config) skip the compile-and-benchmark sweep across process restarts by serializing the winner. **Grouped autotune** uses a $\texttt{prune\_configs\_by}$ filter that drops obviously dominated configs (for example, anything with $\texttt{BLOCK\_K} > K$) before benchmarking, cutting the first-call cost without sacrificing coverage. Both are layered on top of the same decorator.</span>

<span style="font-size: 14px;">A third pattern is **conditional configuration** based on input properties. A kernel deployed on both Ampere-class and Hopper-class hardware benefits from different tile shapes (Hopper's larger tensor-core MMA shapes favor wider blocks); the autotuner can express this by including hardware-appropriate configs in the search list and letting the benchmarking step pick. The key here is that the author is not writing two separate kernels and dispatching on hardware; the same kernel handles every shape and every device, and the decorator's measurement loop discovers the right config per deployment.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Suppose the kernel is called first on $(M, N, K) = (1024, 1024, 1024)$ with three configs in the autotune list. On this first call, Triton compiles all three kernels, runs each one on the actual $1024^3$ inputs, and records wall-clock times of, say, $0.42$ ms for $(64, 64, 32)$, $0.51$ ms for $(128, 64, 32)$, and $0.58$ ms for $(64, 128, 32)$. The autotuner caches $(64, 64, 32)$ under the key $(1024, 1024, 1024)$ and dispatches it for that call.</span>

<span style="font-size: 14px;">A subsequent call on the same key $(1024, 1024, 1024)$ skips the sweep entirely and dispatches the cached $(64, 64, 32)$ config in well under a microsecond of dispatch overhead. A call on a new key $(2048, 256, 1024)$ misses the cache and triggers a fresh sweep, which on this skinny shape might select $(64, 128, 32)$ as the winner because the wider $N$ tile better matches the $N = 256$ dimension's tile count. The same kernel source is now serving two shapes with two different machine-code variants, both held in the autotune cache.</span>

<span style="font-size: 14px;">Counting cost: a three-config sweep on a $1024^3$ matmul takes on the order of tens of milliseconds (compile time plus three benchmark runs), recovered in tens of subsequent dispatches that are now optimal. A twenty-config sweep on the same shape would take hundreds of milliseconds, which is fine for a server warmup but disastrous for a per-test interactive harness. The choice of search-space size is a tradeoff between first-call latency and steady-state performance, and the key argument controls how often the cost is paid: a key that omits the dimension that actually drives the optimal config will under-cache (forcing re-tunes that should not happen), while a key that includes a dimension the optimal config does not depend on will over-cache (forcing re-tunes that should hit).</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Search space too wide for the budget.** Twenty-plus configs on a $60$-second subprocess timeout exhausts the budget compiling, never reaches the benchmark phase, and the first call appears to hang. The remedy is to keep the space to three to five configs inside the harness and rely on a separate offline tuning pass to feed any larger space.</span>

* <span style="font-size: 14px;">**Static grid tuple instead of a callable.** Writing $\texttt{grid} = (\texttt{cdiv}(M, \texttt{BLOCK\_M}), \texttt{cdiv}(N, \texttt{BLOCK\_N}))$ at the call site raises a $\texttt{NameError}$ because $\texttt{BLOCK\_M}$ has not been resolved yet; the autotuner has not picked a config. The grid must be a lambda over the meta-dict so Triton can substitute the winning block sizes after the sweep.</span>

* <span style="font-size: 14px;">**Missing key arguments.** Forgetting to pass $M$, $N$, or $K$ as kwargs to the kernel call makes the autotuner unable to match the cached entry by name; every call looks like a fresh shape and triggers a re-tune. The fix is to call the kernel with explicit kwargs ($\texttt{M=M, N=N, K=K}$) so the autotuner's key resolution succeeds.</span>

* <span style="font-size: 14px;">**Per-process cache surprise.** The autotune cache is process-local. A test harness that spawns a subprocess per test pays the full sweep cost for every test even when the shape repeats. The remedy is a persistent cache file or warming the cache once in a long-lived process.</span>

---