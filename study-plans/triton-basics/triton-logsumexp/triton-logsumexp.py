import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Write code here
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offs = row_id * x_row_stride + cols
    mask = cols < n_cols

    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))

    m = tl.max(x)
    x_m = x - m
    out = m + tl.log(tl.sum(tl.exp(x_m)))

    tl.store(out_ptr + row_id, out)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )