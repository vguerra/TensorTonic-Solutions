import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_bwd_kernel(
    x_ptr, gamma_ptr, dy_ptr,
    dx_ptr, dgamma_ptr, dbeta_ptr,
    stride_x_row, stride_dy_row, stride_dx_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    block_start_x = row_id * stride_x_row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + block_start_x + offs, mask=mask, other=0.0)

    block_start_dy = row_id * stride_dy_row
    dy = tl.load(dy_ptr + block_start_dy + offs, mask=mask, other=0.0)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0)
    
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)

    x_hat = centered * rstd

    dy_norm = dy * gamma

    c1 = tl.sum(dy_norm, axis=0) / N
    c2 = tl.sum(dy_norm * x_hat, axis=0) / N

    dx = rstd * (dy_norm - c1 - x_hat * c2)

    block_start_dx = row_id * stride_dx_row
    tl.store(dx_ptr + block_start_dx + offs, dx, mask=mask)

    tl.atomic_add(dgamma_ptr + offs, dy * x_hat, mask=mask)
    tl.atomic_add(dbeta_ptr + offs, dy, mask=mask)
    

def solve(
    x: torch.Tensor, gamma: torch.Tensor, dy: torch.Tensor,
    dx_out: torch.Tensor, dgamma_out: torch.Tensor, dbeta_out: torch.Tensor,
    eps: float,
) -> None:
    """Launch the LayerNorm backward kernel: one program per row, atomic reductions for dgamma and dbeta."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    dgamma_out.zero_()
    dbeta_out.zero_()
    grid = (M,)
    layernorm_bwd_kernel[grid](
        x, gamma, dy,
        dx_out, dgamma_out, dbeta_out,
        x.stride(0), dy.stride(0), dx_out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )