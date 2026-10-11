import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    
    m = tl.full((1,), -float('inf'), tl.float32)
    l = tl.zeros((1,), tl.float32)

    d1 = tl.arange(0, BLOCK_SIZE)
    for chunk in range(tl.cdiv(N, BLOCK_SIZE)):
        offs = row * N + d1
        mask = d1 < N
        x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
        m_new = tl.maximum(m, tl.max(x))
        l = l * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0))

        m = m_new
        d1 += BLOCK_SIZE

    d1 = tl.arange(0, BLOCK_SIZE)
    for chunk in range(tl.cdiv(N, BLOCK_SIZE)):
        offs = row * N + d1
        mask = d1 < N
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.exp(x - m)/l, mask=mask)
        d1 += BLOCK_SIZE


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the online softmax kernel row-wise."""
    M, N = x.shape
    BLOCK_SIZE = 1024
    grid = (M,)
    online_softmax_kernel[grid](x, out, M, N, BLOCK_SIZE=BLOCK_SIZE)