import torch
import triton
import triton.language as tl


@triton.jit
def matmul_qk_kernel(
    Q_ptr, K_ptr, S_ptr,
    N, N_K, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    q_d0 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_d0 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d1 = tl.arange(0, BLOCK_D)
    offs_q = q_d0[:, None] * D + d1[None, :]
    offs_k = k_d0[:, None] * D + d1[None, :]

    mask_q = (q_d0[:, None] < N) & (d1[None, :] < D)
    mask_k = (k_d0[:, None] < N_K) & (d1[None, :] < D)

    Q = tl.load(Q_ptr + offs_q, mask=mask_q, other=0.0) * scale
    K = tl.load(K_ptr + offs_k, mask=mask_k, other=0.0)

    t = tl.dot(Q, tl.trans(K), input_precision='ieee')

    offs_s = q_d0[:, None] * N_K + k_d0[None, :]
    mask_s = (q_d0[:, None] < N) & (k_d0[None, :] < N_K)

    tl.store(S_ptr + offs_s, t, mask=mask_s)
    

@triton.jit
def softmax_rows_kernel(
    S_ptr, N, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    d0 = tl.arange(0, BLOCK_N)
    offs_s = pid * N + d0
    mask_s = d0 < N

    S_row = tl.load(S_ptr + offs_s, mask=mask_s, other=-float('inf'))
    e_s_m = tl.exp(S_row - tl.max(S_row))
    softmax = e_s_m / tl.sum(e_s_m)

    tl.store(S_ptr + offs_s, softmax, mask=mask_s)


@triton.jit
def matmul_av_kernel(
    S_ptr, V_ptr, out_ptr,
    N, N_V, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_M

    S_d0 = block_start + tl.arange(0, BLOCK_M)
    S_d1 = tl.arange(0, BLOCK_N)

    V_d0 = tl.arange(0, BLOCK_N)
    V_d1 = tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for chunk in range(tl.cdiv(N, BLOCK_N)):
        offs_S = S_d0[:, None] * N + S_d1[None, :]
        mask_S = (S_d0[:, None] < N) & (S_d1[None, :] < N_V)

        S = tl.load(S_ptr + offs_S, mask=mask_S, other=0.0)

        offs_V = V_d0[:, None] * D + V_d1[None, :]
        mask_V = (V_d0[:, None] < N_V) & (V_d1[None, :] < D)

        V = tl.load(V_ptr + offs_V, mask=mask_V, other=0.0)

        acc = tl.dot(S, V, acc=acc, input_precision='ieee')

        S_d1 += BLOCK_N
        V_d0 += BLOCK_N

    out_d0 = block_start + tl.arange(0, BLOCK_M)
    out_d1 = tl.arange(0, BLOCK_D)

    offs_out = out_d0[:, None] * D + out_d1[None, :]
    mask_out = (out_d0[:, None] < N) & (out_d1[None, :] < D)

    tl.store(out_ptr + offs_out, acc, mask=mask_out)


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the three-stage attention: QK^T -> softmax -> @V."""
    N, D = Q.shape
    N_KV, _ = K.shape # Sequence lengths are different for Q and KV (cross attention)
    scale = 1.0 / (D ** 0.5)
    S = torch.empty((N, N_KV), device=Q.device, dtype=torch.float32)

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2
    BLOCK_D = max(BLOCK_D, 16)
    BLOCK_N_PADDED = 1
    while BLOCK_N_PADDED < N_KV:
        BLOCK_N_PADDED *= 2

    grid_qk = ((N + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    matmul_qk_kernel[grid_qk](
        Q, K, S, N, N_KV, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    softmax_rows_kernel[(N,)](S, N_KV, BLOCK_N=BLOCK_N_PADDED)

    grid_av = ((N + BLOCK_M - 1) // BLOCK_M,)
    matmul_av_kernel[grid_av](
        S, V, out, N, N_KV, D,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
    )