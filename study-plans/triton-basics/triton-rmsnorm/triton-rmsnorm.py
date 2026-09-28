import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, gamma_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x_ptrs = x_ptr + pid*stride_x_row + offs    
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0)

    mean_sqrt = tl.sum(x * x, axis=0) / N
    out = tl.rsqrt(mean_sqrt + eps) * x * gamma

    out_ptrs = out_ptr + pid * stride_out_row + offs
    tl.store(out_ptrs, out, mask=mask)


def solve(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the RMSNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_fwd_kernel[grid](
        x, gamma, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )