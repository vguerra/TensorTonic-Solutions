import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, out_ptr,
    N, N_K, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    q_d0 = pid * BLOCK_M + tl.arange(0, BLOCK_M) # row indexes
    q_d1 = tl.arange(0, BLOCK_DMODEL) # col indexes

    offs_q = q_d0[:, None] * D + q_d1[None, :] # offsets for memory layout
    mask_q = (q_d0[:, None] < N) & (q_d1[None, :] < D)

    Q = tl.load(Q_ptr + offs_q, mask=mask_q, other=0.0) * scale
    
    maxs = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    ls = tl.zeros((BLOCK_M,), tl.float32)
    outs = tl.zeros((BLOCK_M, BLOCK_DMODEL), tl.float32)

    k_d0 = tl.arange(0, BLOCK_N)

    for chunk in range(tl.cdiv(N_K, BLOCK_N)):
        K_V_offs = k_d0[:, None] * D + q_d1[None, :]
        mask_k_v = (k_d0[:, None] < N_K) & (q_d1[None, :] < D)
    
        K = tl.load(K_ptr + K_V_offs, mask=mask_k_v, other=0.0)
        V = tl.load(V_ptr + K_V_offs, mask=mask_k_v, other=0.0)

        # loading K transposed is possible and spares the tl.trans here
        s = tl.dot(Q, tl.trans(K), input_precision='ieee')
        s = tl.where(k_d0[None, :] < N_K, s, -float('inf'))
        maxs_new = tl.maximum(maxs, tl.max(s, axis=1))
        e_maxs_diff = tl.exp(maxs - maxs_new)
        p = tl.exp(s - maxs_new[:, None])
        ls = e_maxs_diff * ls + tl.sum(p, axis=1)

        outs = outs * e_maxs_diff[:, None] + tl.dot(p, V, input_precision='ieee')

        maxs = maxs_new    
        k_d0 += BLOCK_N


    outs = outs / ls[:, None]
    
    tl.store(out_ptr + offs_q, outs, mask=mask_q)


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the FlashAttention forward kernel: out = softmax(Q K^T / sqrt(D)) V."""
    N, D = Q.shape
    N_K, _ = K.shape
    scale = 1.0 / (D ** 0.5)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_DMODEL = 1
    while BLOCK_DMODEL < D:
        BLOCK_DMODEL *= 2
    BLOCK_DMODEL = max(BLOCK_DMODEL, 16)
    grid = ((N + BLOCK_M - 1) // BLOCK_M,)
    flash_attention_fwd_kernel[grid](
        Q, K, V, out,
        N, N_K, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )