import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # offs include pid * BLOCK_M plus all indexes to be accessed.
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    # mask makes sure that offsets are within range [0, M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, BLOCK_N)
    offs_x = tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for step_n in range(0, tl.cdiv(N, BLOCK_N)):
        # when computing the set of data pointers to access we need to use the strides
        # that take us from corrdinates to offset of data within flatten vector
        tile_ptrs = a_ptr + offs_m[:, None]*stride_am + offs_n[None, :]*stride_an
        mask_n = offs_n < N
        mask_tile = mask_m[:, None] & mask_n[None, :]

        x_ptrs = x_ptr + offs_x
        mask_x = offs_x < N
        
        tile = tl.load(tile_ptrs, mask=mask_tile, other=0.0)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        acc += tl.sum(tile * x[None, :], axis=1)

        offs_n += BLOCK_N
        offs_x += BLOCK_N

    out_ptrs = out_ptr + offs_m
    tl.store(out_ptrs, acc, mask=mask_m)


def solve(A: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gemv_kernel: out = A @ x."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    gemv_kernel[grid](
        A, x, out,
        M, N,
        A.stride(0), A.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )