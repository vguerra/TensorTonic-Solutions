# <span style="font-size: 20px;">Tiled Transpose</span>

<span style="font-size: 14px;">Transpose is the cleanest demonstration of why **coalesced stores** matter as much as coalesced loads. A scalar transpose that reads row-contiguous from $A$ and writes column-strided into $\texttt{out}$ runs at a fraction of HBM bandwidth because the stores hit one element per cache line instead of a full line per transaction. The Triton tile model finesses this by loading a full $\texttt{BLOCK\_M} \times \texttt{BLOCK\_N}$ tile into registers in row layout and storing it back from the same register tile in transposed layout, so each issued store can still produce a contiguous burst along the destination's fast axis.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given $A \in \mathbb{R}^{M \times N}$, the kernel writes $\texttt{out} \in \mathbb{R}^{N \times M}$ such that:</span>

$$
\texttt{out}[j, i] = A[i, j], \quad 0 \le i < M,\ 0 \le j < N
$$

<span style="font-size: 14px;">No arithmetic, no reduction, just a permutation of indices. Every element moves once, total HBM traffic is $4 (M N + M N) = 8 M N$ bytes for fp32, and the kernel is hard memory-bound. The interesting variable is not the arithmetic but the effective fraction of peak bandwidth.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is two-dimensional, $\lceil M / \texttt{BLOCK\_M} \rceil \times \lceil N / \texttt{BLOCK\_N} \rceil$, with each **program** owning one $\texttt{BLOCK\_M} \times \texttt{BLOCK\_N}$ tile of the input. Program $(\texttt{pid\_m}, \texttt{pid\_n})$ reads $A[i, j]$ for $i \in [\texttt{pid\_m} \cdot \texttt{BLOCK\_M}, \texttt{pid\_m} \cdot \texttt{BLOCK\_M} + \texttt{BLOCK\_M})$ and $j \in [\texttt{pid\_n} \cdot \texttt{BLOCK\_N}, \texttt{pid\_n} \cdot \texttt{BLOCK\_N} + \texttt{BLOCK\_N})$, and writes the corresponding tile of $\texttt{out}$ at $\texttt{out}[j, i]$ for the same indices. There is no $K$-loop, no reduction, no cross-program coordination: every output tile depends on exactly one input tile.</span>

<span style="font-size: 14px;">The parallel pattern is a **2D tile map without reduction**, the lightest weight 2D kernel in the Triton vocabulary. Each program runs one load, one in-register layout reinterpretation, and one store. The launcher waits for all programs to finish; correctness does not depend on any execution order.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">A standard choice is $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 32$, both declared $\texttt{tl.constexpr}$. The tile is square so the in-register layout swap is symmetric; a $32 \times 32$ fp32 tile is $4096$ bytes, well within the register budget of a single program. Powers of two let the compiler emit wide vector loads and stores for the row-contiguous side of each access.</span>

<span style="font-size: 14px;">A single shared mask handles both ends. The expression $(\texttt{offs\_m}[:, \texttt{None}] < M) \,\&\, (\texttt{offs\_n}[\texttt{None}, :] < N)$ marks the lanes that fall inside $A$'s bounds; the same predicate equally marks the lanes that fall inside $\texttt{out}$'s bounds, because the lane $(i, j)$ is valid in $A$ exactly when the lane $(j, i)$ is valid in $\texttt{out}$. The kernel computes the mask once, uses it on the load with $\texttt{other} = 0.0$, and reuses it on the store. Building two separate masks would compile but double the broadcast cost without changing correctness.</span>

<span style="font-size: 14px;">The transpose itself is encoded in the **stride math** on the store, not in a data shuffle. The load pointer is $a\_ptr + \texttt{offs\_m}[:, \texttt{None}] \cdot \texttt{stride\_am} + \texttt{offs\_n}[\texttt{None}, :] \cdot \texttt{stride\_an}$; the store pointer is $\texttt{out\_ptr} + \texttt{offs\_n}[\texttt{None}, :] \cdot \texttt{stride\_om} + \texttt{offs\_m}[:, \texttt{None}] \cdot \texttt{stride\_on}$, with the row and column index expressions swapped on the destination side. The same register tile, written through transposed pointers, lands in the transposed layout. Triton's broadcasting makes this almost a one-liner; the index swap is what does the transpose.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Transpose has zero reuse: every fp32 of $A$ is read once and written once, no element appears in two outputs, no two programs see the same byte. The tile lives in **registers** for the duration of one program (between the $\texttt{tl.load}$ and the $\texttt{tl.store}$), and nothing is staged into SRAM by the kernel because there is no $\texttt{tl.dot}$ for the compiler to back. The compiler decides how the register tile is sharded across warps inside the program; that internal layout is the mechanism by which the row-contiguous load and the column-contiguous store both end up issuing wide HBM transactions.</span>

<span style="font-size: 14px;">For $A$ row-major and $\texttt{out}$ row-major (the standard PyTorch case), the load reads contiguous bytes along $A$'s columns inside each row of the tile and the store writes contiguous bytes along $\texttt{out}$'s columns inside each row of the destination tile. Both ends are coalesced. A scalar transpose that wrote element-by-element would issue $\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N}$ separate stores, each touching a different cache line of $\texttt{out}$ because the address stride is $M$ fp32s between adjacent stores; the per-store amortization would be terrible. The tile form issues $\texttt{BLOCK\_M}$ vector stores of $\texttt{BLOCK\_N}$ fp32s each, every one of them contiguous.</span>

<span style="font-size: 14px;">A more aggressive transpose stages the tile through **shared memory** with explicit padding to avoid bank conflicts, allowing both the load and the store to issue at full bandwidth even when the destination's fast axis is not the load's fast axis. The kernel here does not take that step; the in-register swap is enough for the correctness scope. The shared-memory variant is the natural follow-up when peak bandwidth on the transpose itself is critical.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">The kernel performs zero floating-point operations: every output is a copy of an input. Arithmetic intensity is exactly $0$ FLOPs per byte, the floor of the roofline. The runtime is bounded entirely by HBM bandwidth and how well the kernel's load and store patterns convert that bandwidth into useful work. Optimization is exclusively about effective bandwidth utilization: coalescing on both ends, large enough block sizes to amortize launch overhead, and enough programs to saturate the SMs.</span>

<span style="font-size: 14px;">Compare with matmul, where intensity grows with $\texttt{BLOCK\_K}$ and tiling moves the kernel into compute-bound territory. Transpose is the opposite extreme: tiling does not change the intensity (still zero), but it does change the achieved bandwidth, sometimes by a factor of four or more between the scalar and the tiled form. The lesson is that for the lowest-intensity kernels, the entire performance story is bandwidth efficiency, and the entire purpose of the tile model is to keep both the load and the store coalesced.</span>

<span style="font-size: 14px;">A useful number for ground truth: at fp32 on a $128$-byte cache line, a perfectly coalesced load or store moves $32$ fp32s per transaction. A strided access with stride $M$ between consecutive elements moves $1$ fp32 of useful data per $32$-fp32 transaction, dropping effective bandwidth to $\frac{1}{32}$ of peak. The tile model's job for transpose is to make sure the kernel never lands in that regime; the in-register layout swap is precisely what keeps the strided-access pathology out of the store.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">The author chooses the grid, the tile shape as $\texttt{tl.constexpr}$, the shared mask, and the swapped index expressions on the load and store. The decomposition is fully specified by the pair of pointer expressions; nothing in the kernel body says "transpose". The compiler reads the index expressions, infers the tile layout, and emits a load-and-store sequence whose internal warp-level layout converts the row-major load and the row-major (in $\texttt{out}$'s frame) store into coalesced HBM transactions on both ends.</span>

<span style="font-size: 14px;">The compiler also picks how the in-register tile is laid out across the warp lanes inside one program. For a square $32 \times 32$ tile on hardware with $32$-lane warps, the natural layout puts each row of the tile in one lane group, which makes the load coalesced but the store strided in registers. The compiler resolves this either with **register-level shuffles** between the load and the store (effectively a warp-shuffle transpose, no shared memory involved) or by laying the tile out so both ends are coalesced from the start. Triton authors do not name either mechanism; they write the index swap and let the compiler produce the right code.</span>

<span style="font-size: 14px;">Two compiler-internal mechanisms are worth naming even though the kernel does not invoke them by hand. **Vector width selection** for the load and the store is automatic: the compiler picks $128$-bit loads when the tile shape and the stride allow, $64$-bit when alignment is weaker, scalar fallbacks only as a last resort. **Synchronization** between the load and the store would be required if the compiler routed the tile through shared memory; for this kernel it does not, so no barriers appear in the emitted PTX.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A scalar transpose, written in CUDA or as a Triton kernel with $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 1$, hits the canonical strided-store pathology: each store goes to an address $M$ fp32s away from the previous one, every store touches a new cache line, and HBM transactions carry one useful element per line. Effective bandwidth lands at $1 / 16$ of peak for fp32 on a $128$-byte cache line. The tiled form moves the address stride inside the tile, so consecutive stores hit consecutive lanes of the same destination row; effective bandwidth approaches peak.</span>

<span style="font-size: 14px;">The further optimization is the shared-memory tiled transpose with bank-conflict padding. The kernel stages the tile through a small $\texttt{BLOCK\_M} \times (\texttt{BLOCK\_N} + 1)$ shared-memory region; the load writes the tile row-by-row into shared memory, and the store reads the tile column-by-column out of shared memory, with the $+1$ pad breaking the bank-conflict alignment that would otherwise serialize the column reads. This variant achieves the absolute peak bandwidth the hardware offers for a transpose, but the simple in-register form already gets within a factor close to $1.5$ on modern accelerators, sufficient for most workloads that compose transpose with a downstream matmul.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $M = 5$, $N = 3$, $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 4$. The launch grid is $\lceil 5/4 \rceil \times \lceil 3/4 \rceil = 2 \times 1 = 2$ programs.</span>

<span style="font-size: 14px;">**Program $(0, 0)$**: $\texttt{offs\_m} = [0, 1, 2, 3]$, $\texttt{offs\_n} = [0, 1, 2, 3]$. The mask $(\texttt{offs\_m}[:, \texttt{None}] < 5) \,\&\, (\texttt{offs\_n}[\texttt{None}, :] < 3)$ produces a $4 \times 4$ predicate that is true for the $4 \times 3$ valid sub-tile and false for the rightmost column (lane $\texttt{offs\_n} = 3$). The load pulls $A[0{:}4, 0{:}3]$ into a $4 \times 4$ register tile with the rightmost column zeroed. The store writes to $\texttt{out\_ptr} + \texttt{offs\_n}[\texttt{None}, :] \cdot M + \texttt{offs\_m}[:, \texttt{None}]$ under the same mask, placing the loaded values at $\texttt{out}[0, 0{:}4]$, $\texttt{out}[1, 0{:}4]$, $\texttt{out}[2, 0{:}4]$ and skipping the column-$3$ entries that the mask filtered out.</span>

<span style="font-size: 14px;">**Program $(1, 0)$**: $\texttt{offs\_m} = [4, 5, 6, 7]$, $\texttt{offs\_n} = [0, 1, 2, 3]$. The $M$-side of the mask zeros lanes $5$, $6$, $7$; the $N$-side zeros lane $3$. Only one row of the tile ($\texttt{offs\_m} = 4$) and three of its columns are valid. The store places $A[4, 0]$, $A[4, 1]$, $A[4, 2]$ at $\texttt{out}[0, 4]$, $\texttt{out}[1, 4]$, $\texttt{out}[2, 4]$, finishing the transpose. Counting HBM stores: program $(0, 0)$ issues four vector stores of four fp32s each (one per row of $\texttt{out}$), with the mask filtering out invalid lanes inside each store; program $(1, 0)$ issues the same four stores, mostly masked off, with only one useful element each. Total useful traffic is $5 \cdot 3 = 15$ fp32s in and $15$ fp32s out, exactly the number of elements in $A$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to swap the stride axes on the store.** Writing $\texttt{out\_ptrs} = \texttt{out\_ptr} + \texttt{offs\_m}[:, \texttt{None}] \cdot \texttt{stride\_om} + \texttt{offs\_n}[\texttt{None}, :] \cdot \texttt{stride\_on}$ on the store side produces a copy of $A$, not its transpose. The bug passes on square inputs that happen to be symmetric and fails everywhere else; the fix is to swap the row and column index expressions explicitly on the destination pointer.</span>

* <span style="font-size: 14px;">**Using two separate masks for load and store.** Computing both $(\texttt{offs\_m}[:, \texttt{None}] < M) \,\&\, (\texttt{offs\_n}[\texttt{None}, :] < N)$ and $(\texttt{offs\_n}[\texttt{None}, :] < N) \,\&\, (\texttt{offs\_m}[:, \texttt{None}] < M)$ produces identical predicates and doubles the broadcast cost in the emitted PTX. Reuse the single mask across both memory operations.</span>

* <span style="font-size: 14px;">**Assuming the simple kernel hits peak bandwidth.** The in-register tile transpose is a substantial improvement over a scalar transpose but still loses some bandwidth to non-coalesced stores when $\texttt{out}$ is row-major and the tile shape interacts poorly with the cache-line size. Reaching peak requires the shared-memory padded variant, which is a separate kernel.</span>

* <span style="font-size: 14px;">**Block sizes that mismatch the cache line.** A $\texttt{BLOCK\_N}$ smaller than the number of fp32s per cache line (typically $16$ or $32$) makes the loads sub-line, halving or quartering the effective HBM bandwidth. The standard choice $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 32$ matches one cache line of fp32 on most hardware and saturates the load side; smaller tiles starve the bandwidth, larger ones raise register pressure without further gain on the load.</span>

---