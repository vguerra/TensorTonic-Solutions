import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, loss_out_ptr,
    stride_logits_row,
    B, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * stride_logits_row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C

    logits = tl.load(logits_ptr + block_start + offs, mask=mask, other=-float('inf'))
    target_idx = tl.load(target_ptr + pid)
    target_logit = tl.load(logits_ptr + block_start + target_idx)
    
    logit_max = tl.max(logits, axis=0)
    lse = logit_max + tl.log(tl.sum(tl.exp(logits - logit_max), axis=0))

    tl.atomic_add(loss_out_ptr, lse - target_logit)


def solve(logits: torch.Tensor, target: torch.Tensor, loss_out: torch.Tensor) -> None:
    """Launch the cross-entropy kernel: one program per row, atomic accumulate, then divide by B."""
    B, C = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(C)
    loss_out.zero_()
    grid = (B,)
    cross_entropy_kernel[grid](
        logits, target, loss_out,
        logits.stride(0),
        B, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    loss_out.div_(B)