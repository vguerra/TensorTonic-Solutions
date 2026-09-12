import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)
    

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch relu_kernel: out = max(x, 0)."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)