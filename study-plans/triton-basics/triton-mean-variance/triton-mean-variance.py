import torch
import triton
import triton.language as tl


@triton.jit
def mean_var_kernel(x_ptr, sum_ptr, sumsq_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sum = tl.sum(x)
    sumsq = tl.sum(x * x)

    tl.atomic_add(sum_ptr, sum)
    tl.atomic_add(sumsq_ptr, sumsq)


def solve(x: torch.Tensor, mean_out: torch.Tensor, var_out: torch.Tensor) -> None:
    """Launch mean_var_kernel and finalize mean and variance."""
    n = x.numel()
    sum_buf = torch.zeros(1, device='cuda', dtype=torch.float32)
    sumsq_buf = torch.zeros(1, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    mean_var_kernel[grid](x, sum_buf, sumsq_buf, n, BLOCK_SIZE=BLOCK_SIZE)
    mean = sum_buf / n
    var = sumsq_buf / n - mean * mean
    mean_out.copy_(mean)
    var_out.copy_(var)