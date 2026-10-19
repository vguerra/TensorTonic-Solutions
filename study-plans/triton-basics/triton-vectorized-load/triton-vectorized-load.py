import torch
import triton
import triton.language as tl


@triton.jit
def vectorized_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(axis=0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        (n,),
        (1,),
        (pid * BLOCK_SIZE,),
        (BLOCK_SIZE,),
        (0,),
    )

    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        (n,),
        (1,),
        (pid * BLOCK_SIZE,),
        (BLOCK_SIZE,),
        (0,),
    )

    x = tl.load(x_block_ptr, boundary_check=(0,), padding_option='zero')
    y = tl.load(y_block_ptr, boundary_check=(0,), padding_option='zero')

    out_block_ptr = tl.make_block_ptr(
        out_ptr,
        (n,),
        (1,),
        (pid * BLOCK_SIZE,),
        (BLOCK_SIZE,),
        (0,),
    )

    tl.store(out_block_ptr, x + y, boundary_check=(0,))


def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch vectorized_add_kernel with a large BLOCK_SIZE for long arrays."""
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vectorized_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)