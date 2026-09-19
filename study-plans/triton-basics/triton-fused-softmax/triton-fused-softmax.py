import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Write code here
    row_idx = tl.program_id(axis=0)
    offs = row_idx * x_row_stride + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    m = tl.max(x)
    x_m = x - m
    x_exp = tl.exp(x_m)
    softmax = x_exp / tl.sum(x_exp)

    out_offs = row_idx * out_row_stride + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + out_offs, softmax, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )