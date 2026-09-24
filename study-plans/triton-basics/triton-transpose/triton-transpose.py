import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    a_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(axis=0), tl.program_id(axis=1)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs = offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a_ptrs = a_ptr + offs

    a = tl.load(a_ptrs, mask=mask, other=0.0)

    offs_out = offs_n[None, :] * stride_om + offs_m[:, None] * stride_on
    out_ptrs = out_ptr + offs_out
    
    tl.store(out_ptrs, a, mask=mask)


def solve(A: torch.Tensor, out: torch.Tensor) -> None:
    """Launch transpose_kernel: out[j, i] = A[i, j]."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](
        A, out,
        M, N,
        A.stride(0), A.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    print(A.stride(), out.stride())