# <span style="font-size: 20px;">Block-Pointer Matmul</span>

<span style="font-size: 14px;">Block-pointer matmul keeps the entire numerical and structural design of the baseline tiled GEMM and replaces only the addressing layer. Where the baseline kernel builds 2D tensors of raw integer offsets with $\texttt{tl.arange}$ broadcasts, multiplies them by strides, and writes per-axis boolean masks by hand, this version packages all of that into a single descriptor object via $\texttt{tl.make\_block\_ptr}$ and steps through the $K$ axis with $\texttt{tl.advance}$. The IR the compiler sees changes, the generated PTX gets cleaner, and on Hopper-class hardware the compiler can lower the descriptor to TMA (Tensor Memory Accelerator) instructions. The lesson is that **how addresses are expressed determines what the compiler can do with them**, even when the user-level semantics are identical.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">The numerical contract is unchanged from the baseline matmul:</span>

$$
C[i, j] = \sum_{k=0}^{K-1} A[i, k] \cdot B[k, j]
$$

<span style="font-size: 14px;">$A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, $C \in \mathbb{R}^{M \times N}$, row-major fp32. The tile shapes ($\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$, $\texttt{BLOCK\_K} = 32$) and the launch grid (2D, $\lceil M / \texttt{BLOCK\_M} \rceil \times \lceil N / \texttt{BLOCK\_N} \rceil$) are unchanged. The output of this kernel is bit-identical (up to associativity in the reduction) to the baseline.</span>

---

## <span style="font-size: 16px;">Baseline: Raw Pointer Arithmetic</span>

<span style="font-size: 14px;">The baseline tiled matmul builds its load addresses element by element. The author writes</span>

$$
a\_ptrs = a\_ptr + \texttt{offs\_m}[:, \text{None}] \cdot \text{stride\_am} + \texttt{offs\_k}[\text{None}, :] \cdot \text{stride\_ak}
$$

<span style="font-size: 14px;">which materializes a $(\texttt{BLOCK\_M}, \texttt{BLOCK\_K})$ tile of pointers, one per element. The inner loop advances the pointer tile by $\texttt{BLOCK\_K} \cdot \text{stride\_ak}$ on each iteration, and every $\texttt{tl.load}$ takes an explicit two-axis mask of the form $(\texttt{offs\_m}[:, \text{None}] < M) \,\&\, (k\_mask[\text{None}, :])$. The compiler sees a tile of integer offsets and a separate boolean mask tile, and it has to reverse-engineer the access pattern from that representation.</span>

<span style="font-size: 14px;">This works and runs fast on Ampere and earlier. On Hopper, the bounds-and-stride reasoning that the compiler has to redo from raw integer offsets is what blocks the lowering to TMA descriptors. TMA wants a single piece of metadata that says "here is a logical tensor of shape $(M, K)$, here are its strides, here is the tile I want, please prefetch this whole thing into SRAM"; the per-element pointer tile cannot be cleanly reconstructed back into that form.</span>

---

## <span style="font-size: 16px;">The Change: Descriptor-Based Addressing</span>

<span style="font-size: 14px;">$\texttt{tl.make\_block\_ptr}$ takes six arguments and returns a single descriptor:</span>

<span style="font-size: 14px;">1. **base** - the raw pointer to element zero of the underlying tensor.</span>

<span style="font-size: 14px;">2. **shape** - the logical shape, a tuple of runtime values used for bounds checks ($M, K$ for $A$).</span>

<span style="font-size: 14px;">3. **strides** - element strides on each axis, also runtime values.</span>

<span style="font-size: 14px;">4. **offsets** - the top-left corner of the current tile, $(\text{pid\_m} \cdot \texttt{BLOCK\_M}, 0)$ for the first iteration on $A$.</span>

<span style="font-size: 14px;">5. **block_shape** - the $\texttt{tl.constexpr}$ tile size, $(\texttt{BLOCK\_M}, \texttt{BLOCK\_K})$.</span>

<span style="font-size: 14px;">6. **order** - the memory-traversal order, where $(1, 0)$ means axis $1$ is the contiguous axis. For row-major $A$ of shape $(M, K)$ the $K$ axis is contiguous, so $\text{order} = (1, 0)$.</span>

<span style="font-size: 14px;">The descriptor carries everything the compiler needs to reason about the access pattern: the logical tensor, the strides, the current window, the tile shape, and the contiguous direction. Boundary handling is centralized: $\texttt{tl.load}(\text{block\_ptr}, \text{boundary\_check}=(0, 1), \text{padding\_option}=\text{'zero'})$ checks both axes of the descriptor against the logical shape and pads out-of-bounds elements with zero, identical in effect to the per-axis masks of the baseline.</span>

---

## <span style="font-size: 16px;">Walking K with tl.advance</span>

<span style="font-size: 14px;">$\texttt{tl.advance}(\text{block\_ptr}, (0, \texttt{BLOCK\_K}))$ returns a new descriptor with its offsets shifted by $(0, \texttt{BLOCK\_K})$. The delta is required to be $\texttt{tl.constexpr}$ on both axes, which is satisfied because $\texttt{BLOCK\_K}$ is a meta-parameter. The inner $K$-loop becomes:</span>

$$
a\_ptr \leftarrow \texttt{tl.advance}(a\_ptr, (0, \texttt{BLOCK\_K}))
$$

$$
b\_ptr \leftarrow \texttt{tl.advance}(b\_ptr, (\texttt{BLOCK\_K}, 0))
$$

<span style="font-size: 14px;">The user never writes $\text{stride\_ak} \cdot \texttt{BLOCK\_K}$ arithmetic anywhere. The descriptor knows the strides and the compiler folds the advance into the address computation directly. This matters because $\texttt{tl.advance}$ returns a new descriptor rather than mutating in place; the loop body must reassign $a\_ptr$ each iteration. Forgetting to reassign leaves the kernel stuck on the first $K$-block, a bug the type system does not catch and that produces silently wrong output.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The accumulator and tile shapes are unchanged: an fp32 $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N})$ accumulator, $A$ and $B$ tiles of $(\texttt{BLOCK\_M}, \texttt{BLOCK\_K})$ and $(\texttt{BLOCK\_K}, \texttt{BLOCK\_N})$. Masking moves from per-load boolean tensors to a single $\text{boundary\_check}$ argument that names which axes of the descriptor to bounds-check. $(0, 1)$ means "check both axes of the 2D tile"; the descriptor's $\text{shape}$ argument supplies the actual limits.</span>

<span style="font-size: 14px;">The store uses the same mechanism: $\texttt{tl.store}(c\_ptr, \text{acc}, \text{boundary\_check}=(0, 1))$ writes only the in-bounds part of the tile. The $C$ descriptor is built once near the end of the kernel with $\text{offsets} = (\text{pid\_m} \cdot \texttt{BLOCK\_M}, \text{pid\_n} \cdot \texttt{BLOCK\_N})$ and a $(\texttt{BLOCK\_M}, \texttt{BLOCK\_N})$ block shape. There is no need to advance $C$ because each program writes its tile exactly once.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and What the Compiler Does With the Descriptor</span>

<span style="font-size: 14px;">The descriptor is the input to the compiler's load lowering. Three things change in the generated code compared to the raw-pointer kernel:</span>

<span style="font-size: 14px;">1. **Vector width** - the compiler can prove from the descriptor's $\text{order}$ field which axis is contiguous and emit the widest vector load that fits ($\text{ld.global.v4}$ on most current hardware). With raw pointer tiles, the compiler has to infer contiguity from the offset pattern and may fall back to a narrower load.</span>

<span style="font-size: 14px;">2. **TMA lowering on Hopper** - when the target is Hopper or newer, the descriptor lowers directly to a $\text{cp.async.bulk.tensor}$ TMA instruction. TMA issues a single instruction to bring a 2D tile from HBM into SRAM, with hardware-managed prefetching and address generation, and frees the threads to do other work while the load is in flight. The TMA path is what makes block-pointer matmul measurably faster on H100; on earlier hardware the speedup is smaller and comes from the cleaner IR alone.</span>

<span style="font-size: 14px;">3. **Bounds-check folding** - the compiler centralizes the bounds check in one place per axis. With raw per-load masks, the compiler may emit redundant comparisons; with the descriptor, the comparison happens at most once per axis per load and the masking applies to the entire tile in one swing.</span>

---

## <span style="font-size: 16px;">When TMA Pays</span>

<span style="font-size: 14px;">The TMA instruction on Hopper has two structural advantages over the per-warp load issue of earlier hardware. First, it is asynchronous: the kernel issues the TMA, continues with other work (typically a $\texttt{tl.dot}$ on the previously loaded tiles), and waits for the load only when the next iteration needs the data. The software-pipelining the compiler already does via $\texttt{num\_stages}$ becomes more effective because the load itself is now an asynchronous primitive rather than a stream of synchronous loads that the compiler had to overlap by hand. Second, TMA generates addresses in hardware from the tensor descriptor, freeing the threads from running the offset arithmetic and letting them stay on tensor-core work.</span>

<span style="font-size: 14px;">On pre-Hopper hardware (Ampere, Ada), there is no TMA, and the block-pointer kernel is roughly tied with the raw-pointer kernel on most shapes. The descriptor still gives the compiler cleaner inputs and the generated PTX is shorter, but the practical win is small. The reason to write block-pointer kernels on older hardware is portability: the same source produces a fast kernel on Hopper without rewriting the addressing, and the kernel reads more cleanly to humans because all the bounds reasoning lives in one place.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Roofline placement is unchanged from the baseline tiled matmul. The kernel sits on the compute-bound side at $\approx \texttt{BLOCK\_K} / 4$ FLOPs per byte for square tiles, well past the crossover on every current accelerator. What the descriptor improves is the constant factor on the bandwidth term, by getting closer to the theoretical peak HBM throughput per load through wider vectors and (on Hopper) TMA. The change does not shift the kernel across the roofline; it lifts the achievable fraction of peak on the bandwidth-supply side, which the inner $K$-loop benefits from on every iteration.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Compiler handles:** translating the descriptor into a TMA instruction on Hopper, picking vector widths from $\text{order}$, generating the bounds-check code from $\text{shape}$, staging the loaded tile into shared memory for $\texttt{tl.dot}$, and continuing to do everything it already did in the baseline kernel.</span>

<span style="font-size: 14px;">**Author handles:** the descriptor construction (correct $\text{shape}$, $\text{strides}$, $\text{order}$, and constexpr $\text{block\_shape}$), the choice of $\text{order} = (1, 0)$ vs $(0, 1)$ to match the storage layout, the reassignment of the block pointer after each $\texttt{tl.advance}$, and the $\text{boundary\_check}$ arguments on every load and store. The descriptor is a contract: get the contents wrong and the compiler generates a working but slow kernel with no warning.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $M = 65$, $N = 64$, $K = 33$, $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$, $\texttt{BLOCK\_K} = 32$. The launch grid is $\lceil 65 / 64 \rceil \times \lceil 64 / 64 \rceil = 2 \times 1 = 2$ programs.</span>

<span style="font-size: 14px;">**Program $(0, 0)$** owns output tile $C[0:64, 0:64]$. It builds an $A$ descriptor with $\text{shape} = (65, 33)$, $\text{offsets} = (0, 0)$, $\text{block\_shape} = (64, 32)$. The first $\texttt{tl.load}$ with $\text{boundary\_check} = (0, 1)$ reads $A[0:64, 0:32]$; rows $0\!-\!63$ are in-bounds in the $M$ axis (the descriptor's $\text{shape}[0] = 65$) and columns $0\!-\!31$ are in-bounds in the $K$ axis ($\text{shape}[1] = 33$). $\texttt{tl.advance}$ shifts the descriptor to $\text{offsets} = (0, 32)$. The next $\texttt{tl.load}$ reads $A[0:64, 32:64]$; the $M$ axis is still in-bounds, but only column $32$ is in-bounds in the $K$ axis (column $33$ onward is past the tensor). The descriptor pads columns $33\!-\!63$ with zero from $\text{padding\_option}$, so the $\texttt{tl.dot}$ sees zeros there and the matmul result is unaffected.</span>

<span style="font-size: 14px;">**Program $(1, 0)$** owns output tile $C[64:128, 0:64]$. Its $A$ descriptor has $\text{offsets} = (64, 0)$, and only row $64$ is in-bounds in the $M$ axis (rows $65\!-\!127$ are past). The store uses the same descriptor mechanism on $C$ and writes only $C[64:65, 0:64]$; the out-of-bounds rows are silently dropped.</span>

<span style="font-size: 14px;">In the baseline kernel, all of this bounds reasoning would have been spread across three explicit mask tensors. With the descriptor, the kernel body never names a comparison and never spells out a mask; the bounds live in the descriptor's $\text{shape}$ argument and the compiler applies them.</span>

<span style="font-size: 14px;">Counting the bounds checks the compiler emits for this case: two for $A$ (one on each axis on each load, for $\lceil K / \texttt{BLOCK\_K} \rceil = 2$ loads), two for $B$ similarly, and two on the final $C$ store. Compared to the raw-pointer kernel, which materialized full mask tensors of the same shape as the data tiles on every load, the IR is much smaller and the constant overhead per load drops by enough to be visible in the timeline view.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Not reassigning after $\texttt{tl.advance}$.** $\texttt{tl.advance}$ returns a new descriptor and does not mutate the existing one. Writing $\texttt{tl.advance}(a\_ptr, (0, \texttt{BLOCK\_K}))$ on a line by itself, without $a\_ptr =$ on the left, leaves the loop stuck on the first $K$-block. The kernel compiles and runs and produces wrong output.</span>

* <span style="font-size: 14px;">**Wrong $\text{order}$ for the storage layout.** For row-major $A$ of shape $(M, K)$ the contiguous axis is $K$, so $\text{order} = (1, 0)$. Writing $(0, 1)$ tells the compiler the $M$ axis is contiguous and produces strided loads that are several times slower; on Hopper, the TMA lowering is silently disabled and the kernel falls back to scalar codegen for the descriptor.</span>

* <span style="font-size: 14px;">**Forgetting $\text{boundary\_check}$ on $\texttt{tl.load}$ or $\texttt{tl.store}$.** Without it, the descriptor reads or writes past the end of the tensor when the shapes are not multiples of the block, identical in failure mode to forgetting masks in the raw-pointer kernel. The descriptor does not enforce bounds on its own; bounds checking is opt-in via $\text{boundary\_check}$.</span>

* <span style="font-size: 14px;">**Passing runtime values for $\text{block\_shape}$.** $\texttt{tl.make\_block\_ptr}$ requires $\text{block\_shape}$ to be known at JIT time. Passing $(M, K)$ (runtime ints) instead of $(\texttt{BLOCK\_M}, \texttt{BLOCK\_K})$ (constexprs) fails to compile with a not-always-helpful error message.</span>

---