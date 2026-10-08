import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    N, D,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)

    d0 = tl.arange(0, BLOCK_SIZE)
    offs_even = row * D + 2*d0
    offs_odd = row * D + 2 * d0 + 1
    mask = d0 < D/2

    x_even = tl.load(x_ptr + offs_even, mask=mask, other=0.0)
    x_odd = tl.load(x_ptr + offs_odd, mask=mask, other=0.0)

    offs_tables = row * D // 2 + d0
    cos = tl.load(cos_ptr + offs_tables, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + offs_tables, mask=mask, other=0.0)

    outs_even = x_even * cos - x_odd * sin
    outs_odd = x_even * sin + x_odd * cos

    tl.store(out_ptr + offs_even, outs_even, mask=mask)
    tl.store(out_ptr + offs_odd, outs_odd, mask=mask)
    

def solve(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the RoPE kernel: rotate pairs of channels with per-row (cos, sin) tables."""
    N, D = x.shape
    BLOCK_SIZE = 1
    while BLOCK_SIZE < D // 2:
        BLOCK_SIZE *= 2
    if BLOCK_SIZE < 1:
        BLOCK_SIZE = 1
    grid = (N,)
    rope_kernel[grid](x, cos, sin, out, N, D, BLOCK_SIZE=BLOCK_SIZE)