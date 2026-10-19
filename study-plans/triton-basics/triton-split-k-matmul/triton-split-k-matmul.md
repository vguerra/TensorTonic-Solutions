# <span style="font-size: 20px;">Split-K Matmul</span>

<span style="font-size: 14px;">Split-K is the kernel that finally takes seriously what most readers have been ignoring since the matmul section: the $K$ axis is a reduction, and reductions are something parallel hardware can split. The baseline 2D matmul launches exactly one program per output tile and loops the entire $K$ dimension inside that program, which is a fine choice when the output is large enough to keep every streaming multiprocessor busy. When it isn't, the device sits idle while one program at a time grinds through $K$. Split-K cuts the $K$ axis into $\texttt{SPLIT\_K}$ contiguous slices, launches one program per (output tile, slice) pair, has each program compute a partial sum, and combines the partials in the output buffer with $\texttt{tl.atomic\_add}$. The lesson is that **parallelism along the reduction axis is available but costs an atomic combine**, and the tradeoff is favorable exactly when the standard grid is starved.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">The numerical contract is unchanged from the baseline matmul:</span>

$$
C[i, j] = \sum_{k=0}^{K-1} A[i, k] \cdot B[k, j]
$$

<span style="font-size: 14px;">$A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, $C \in \mathbb{R}^{M \times N}$, all row-major fp32. What changes is how the sum on the right is broken across programs. The mathematical sum is associative, so partitioning the $K$ range into disjoint slices and adding the slice-wise partials gives the same answer up to floating-point rounding from the different reduction order.</span>

---

## <span style="font-size: 16px;">Why the Baseline Grid Underutilizes the Device</span>

<span style="font-size: 14px;">The baseline 2D-grid matmul builds a grid of $\texttt{num\_pid\_m} \cdot \texttt{num\_pid\_n}$ programs, one per output tile, where $\texttt{num\_pid\_m} = \lceil M / \texttt{BLOCK\_M} \rceil$ and $\texttt{num\_pid\_n} = \lceil N / \texttt{BLOCK\_N} \rceil$. For a square $M = N = 4096$ matmul with $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$, that is $64 \cdot 64 = 4096$ programs, more than enough to fill the device with several waves of work.</span>

<span style="font-size: 14px;">The problem appears in tall-and-skinny shapes: $M = 128$, $N = 128$, $K = 16384$, the kind of GEMM that shows up in the residual stream of an LLM matmul or a fused gating layer. Now $\texttt{num\_pid\_m} \cdot \texttt{num\_pid\_n} = 2 \cdot 2 = 4$ programs, and a modern accelerator has roughly $100$ streaming multiprocessors. The grid covers $4$ of them and the other $\approx 96$ sit idle for the entire duration of the kernel. The reduction inside each of the four programs is enormous ($16384 / 32 = 512$ $K$-tile iterations), but spreading that reduction across the idle hardware is exactly what the baseline grid refuses to do.</span>

---

## <span style="font-size: 16px;">The Change: Third Grid Axis on K</span>

<span style="font-size: 14px;">Split-K multiplies the grid by an integer $\texttt{SPLIT\_K}$ and assigns each of the $\texttt{SPLIT\_K}$ new programs per output tile a contiguous slice of $K$. With $\texttt{SPLIT\_K} = 4$ on the example above, the grid grows from $4$ to $16$ programs, and the per-program $K$-loop shrinks from $512$ iterations to $128$. The decode that the kernel runs is:</span>

<span style="font-size: 14px;">1. **Linearized program ID**: $\text{pid} = \texttt{tl.program\_id(0)}$ over a 1D grid of size $\texttt{num\_pid\_m} \cdot \texttt{num\_pid\_n} \cdot \texttt{SPLIT\_K}$.</span>

<span style="font-size: 14px;">2. **Decode the K slice**: $\text{pid\_k} = \text{pid} \bmod \texttt{SPLIT\_K}$, then $\text{pid\_mn} = \text{pid} / \texttt{SPLIT\_K}$, then $(\text{pid\_m}, \text{pid\_n}) = (\text{pid\_mn} / \texttt{num\_pid\_n}, \text{pid\_mn} \bmod \texttt{num\_pid\_n})$.</span>

<span style="font-size: 14px;">3. **Compute K bounds**: $k_\text{start} = \text{pid\_k} \cdot \lceil K / \texttt{SPLIT\_K} \rceil$ and $k_\text{end} = \min(k_\text{start} + \lceil K / \texttt{SPLIT\_K} \rceil, K)$. The ceiling division lets $\texttt{SPLIT\_K}$ not divide $K$ evenly; the last slice simply gets fewer iterations.</span>

<span style="font-size: 14px;">Each program walks $A$ and $B$ from $k_\text{start}$ to $k_\text{end}$, accumulates partial $\texttt{tl.dot}$ products into an fp32 register tile, and at the end combines its partial into $C$.</span>

---

## <span style="font-size: 16px;">Atomic Combine and Pre-Zeroing</span>

<span style="font-size: 14px;">Multiple programs now write to the same output tile: all $\texttt{SPLIT\_K}$ programs that share $(\text{pid\_m}, \text{pid\_n})$. The only sanctioned way for distinct Triton programs to combine partials into the same memory location is $\texttt{tl.atomic\_add}$, which lowers to a hardware atomic on global memory. The final store becomes</span>

$$
\texttt{tl.atomic\_add}(C[\text{tile}], \text{acc}, \texttt{mask})
$$

<span style="font-size: 14px;">instead of the plain $\texttt{tl.store}$ the baseline used. Hardware atomics on fp32 are read-modify-write operations that the memory subsystem serializes per address, so two programs that target the same output element take their adds in some sequential order. The mathematical sum is associative so the answer is the same up to rounding, but the actual rounding sequence is non-deterministic across runs.</span>

<span style="font-size: 14px;">Because the kernel adds into $C$ rather than overwriting it, the contents of $C$ before the first atomic matter. The launcher must pre-zero the output: $\texttt{out.zero\_()}$ on the host side before $\text{split\_k\_matmul\_kernel}$ is invoked. The standard $\texttt{torch.empty}$ allocation returns whatever bits were already in the allocator's free pool, and atomically adding the partials into garbage produces output that looks roughly right and is wrong by a constant offset that grows with $\texttt{SPLIT\_K}$.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">Tile shapes are unchanged from the baseline. Each program owns a $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N})$ output tile in an fp32 register accumulator and streams $A$ tiles of shape $(\texttt{BLOCK\_M}, \texttt{BLOCK\_K})$ and $B$ tiles of shape $(\texttt{BLOCK\_K}, \texttt{BLOCK\_N})$ through the inner loop. The masks gain one detail: the $K$-bound is now $k_\text{end}$ instead of $K$, since each program is responsible for a single slice. The inner load uses $(k + \texttt{offs\_k}) < k_\text{end}$ rather than $< K$, with $\texttt{other} = 0.0$ so masked-off lanes contribute zero to the partial $\texttt{tl.dot}$ and do not poison the sum.</span>

<span style="font-size: 14px;">The $M$ and $N$ masks on the load and the atomic-add are unchanged; they protect against the same partial tiles at the edges of the output. The atomic-add mask is what prevents two different output tiles from racing on the same edge address, which would happen if the mask were skipped and the atomic touched whatever memory followed the output buffer.</span>

---

## <span style="font-size: 16px;">Atomic Contention as the Cost</span>

<span style="font-size: 14px;">The atomic combine is not free. Each of the $\texttt{BLOCK\_M} \cdot \texttt{BLOCK\_N}$ elements in the output tile is touched by $\texttt{SPLIT\_K}$ atomic adds, and the memory subsystem must serialize the adds per address. With $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$ and $\texttt{SPLIT\_K} = 4$, that is $16{,}384$ atomic operations per output tile, distributed evenly so that no single address sees more than $\texttt{SPLIT\_K}$ contenders. Contention is bounded by $\texttt{SPLIT\_K}$, which is the lever that controls the cost.</span>

<span style="font-size: 14px;">The right way to think about the tradeoff: split-K is worthwhile when the parallelism gained from the new grid dimension exceeds the time spent on the atomic combine. For the example above (4 output tiles, $\texttt{SPLIT\_K} = 4$, 16 programs total), the gain is going from $4\%$ to $16\%$ device occupancy on a 100-SM accelerator, and the atomic cost is a single $\text{atomic\_add}$ on each of the $4096$ output elements at the end of each program. The reduction work inside each program drops by $4\times$, and the kernel time drops correspondingly. For a tile that already saturates the device on its own ($M = N = 4096$ with $\texttt{SPLIT\_K} = 1$), adding split-K does nothing useful: the device is already busy, the atomics just add overhead.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Split-K does not change the kernel's roofline placement. The tile-level $\texttt{tl.dot}$ still runs at compute-bound intensity, $\approx \texttt{BLOCK\_K} / 4$ FLOPs per byte, because each operand tile inside the inner $K$-loop is reused across the full output tile via tensor-core MMA. What split-K does change is the bandwidth pattern: each $K$-slice loads only its own $\lceil K / \texttt{SPLIT\_K} \rceil$ tiles of $A$ and $B$, so the per-program HBM traffic shrinks by $\texttt{SPLIT\_K}$, but the total HBM traffic across all programs is the same as the baseline (every $A$ and $B$ tile still gets loaded exactly $\texttt{num\_pid\_n}$ or $\texttt{num\_pid\_m}$ times). The extra cost is the atomic-write traffic on $C$, which is the only new term.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Compiler handles:** the $\texttt{tl.dot}$ lowering and shared-memory staging are identical to the baseline. The $\texttt{tl.atomic\_add}$ instruction lowers to a hardware $\text{red.global.add}$ on global memory, which the compiler emits as one atomic per output lane.</span>

<span style="font-size: 14px;">**Author handles:** the choice of $\texttt{SPLIT\_K}$, the integer decode of $\text{pid}$ into $(\text{pid\_m}, \text{pid\_n}, \text{pid\_k})$, the $k_\text{start}$ / $k_\text{end}$ bounds with ceiling division so the partial last slice is handled, the swap of $\texttt{tl.store}$ for $\texttt{tl.atomic\_add}$, and (critically) the host-side $\texttt{out.zero\_()}$. The kernel cannot infer that the output needs pre-zeroing because the kernel has no concept of "before the first launch" - that contract lives in the launcher.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $M = N = 64$, $K = 16$, $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$, $\texttt{BLOCK\_K} = 4$, $\texttt{SPLIT\_K} = 4$. The output is a single $(64, 64)$ tile, so the baseline grid is one program. The split-K grid is $1 \cdot 1 \cdot 4 = 4$ programs, all targeting the same output tile.</span>

<span style="font-size: 14px;">Per-program slice assignments with $\lceil K / \texttt{SPLIT\_K} \rceil = 4$:</span>

<span style="font-size: 14px;">1. $\text{pid} = 0 \to \text{pid\_k} = 0$, $k_\text{start} = 0$, $k_\text{end} = 4$. The K-loop runs one iteration over $k = 0, 1, 2, 3$. The program computes the partial $\sum_{k=0}^{3} A[:, k] \cdot B[k, :]$ into a register tile, then atomic-adds it into $C[0:64, 0:64]$.</span>

<span style="font-size: 14px;">2. $\text{pid} = 1 \to \text{pid\_k} = 1$, $k_\text{start} = 4$, $k_\text{end} = 8$. The K-loop runs one iteration over $k = 4, 5, 6, 7$. Same atomic-add into $C$.</span>

<span style="font-size: 14px;">3. $\text{pid} = 2$ and $\text{pid} = 3$ handle $k = 8 \dots 11$ and $k = 12 \dots 15$ respectively.</span>

<span style="font-size: 14px;">The four atomic-adds combine into the same $C$ buffer (which the launcher already zeroed) and the final contents of $C[i, j]$ are $\sum_{k=0}^{15} A[i, k] \cdot B[k, j]$, the correct matmul. The atomics serialize per address but proceed in parallel across addresses, so the four programs do not block each other on tile-level work, only on the final write of each output lane.</span>

---

## <span style="font-size: 16px;">Choosing SPLIT_K</span>

<span style="font-size: 14px;">$\texttt{SPLIT\_K}$ is the new tuning knob. The useful range is bounded on both sides. The floor is $1$ (no split, identical to the baseline). The ceiling is roughly $K / \texttt{BLOCK\_K}$, the number of $K$-tile iterations available to partition; pushing past that gives some programs an empty K-loop and they still pay the atomic-add cost for no real work. In practice $\texttt{SPLIT\_K} \in \{2, 4, 8\}$ covers nearly every case, and $4$ is the value the Triton matmul tutorial uses as a default for skinny GEMMs. Beyond $8$, the atomic contention starts to dominate and the kernel slows down.</span>

<span style="font-size: 14px;">For the canonical skinny shape ($M$ or $N$ in the low hundreds, $K$ in the tens of thousands), the right $\texttt{SPLIT\_K}$ is whatever brings the total grid size up to roughly the SM count of the device times a factor of $2$ to $4$ (so the scheduler has a small queue per SM). For square matmuls where the baseline grid is already huge, $\texttt{SPLIT\_K} = 1$ wins and split-K is dead weight.</span>

<span style="font-size: 14px;">There is a less obvious tuning interaction with $\texttt{BLOCK\_K}$. A larger $\texttt{BLOCK\_K}$ shrinks the inner-loop trip count but raises register pressure inside each program. With split-K, each program already has a shorter inner loop because its $K$-slice is smaller, so the case for a large $\texttt{BLOCK\_K}$ weakens. A typical split-K config keeps $\texttt{BLOCK\_K}$ at the same value as the baseline ($32$) and gets its parallelism gain purely from the third grid axis.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting $\texttt{out.zero\_()}$ in the launcher.** The first atomic-add reads whatever bits the allocator handed back from $\texttt{torch.empty}$. Tests see numbers in the right ballpark but with a random constant offset, and the offset grows linearly with $\texttt{SPLIT\_K}$. This is the most common split-K bug.</span>

* <span style="font-size: 14px;">**Using $\texttt{tl.store}$ instead of $\texttt{tl.atomic\_add}$.** Without the atomic, the last-arriving program clobbers everyone else's partial sums and only its own $K$-slice survives. Tests look wrong by a factor of $\texttt{SPLIT\_K}$ in the wrong direction (they undershoot, not overshoot).</span>

* <span style="font-size: 14px;">**Non-deterministic output across runs.** Hardware atomics commit in whatever order the memory subsystem picks, and the floating-point sum of partials depends on that order. The output is correct but bit-different across runs, so equality-tolerance tests need $\texttt{atol} \sim 10^{-2}$ rather than $10^{-5}$.</span>

* <span style="font-size: 14px;">**$\texttt{SPLIT\_K}$ larger than $K / \texttt{BLOCK\_K}$.** Most $\text{pid\_k}$ programs run an empty K-loop and the only thing they do is the atomic-add at the end. The kernel pays the launch and atomic costs for no productive work; clamp $\texttt{SPLIT\_K}$ at the host side to at most $K / \texttt{BLOCK\_K}$.</span>

---