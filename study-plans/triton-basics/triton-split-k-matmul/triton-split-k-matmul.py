import torch
import triton
import triton.language as tl


@triton.jit
def split_k_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pids_n = tl.cdiv(N, BLOCK_N)
    num_pids_m = tl.cdiv(M, BLOCK_M)
    chunk_k = tl.cdiv(K, SPLIT_K)

    pid_k = pid % SPLIT_K
    pid_mn = pid // SPLIT_K

    (pid_m, pid_n) = (pid_mn // num_pids_n, pid_mn % num_pids_n)
    
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    k_start = pid_k * chunk_k 
    k_end = tl.minimum(k_start + chunk_k, K)
    offs_k = k_start + tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None]*stride_am + offs_k[None, :]*stride_ak
    b_ptrs = b_ptr + offs_k[:, None]*stride_bk + offs_n[None, :]*stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for chunk in range(tl.cdiv(chunk_k, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_end - chunk * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_end - chunk * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc=acc, input_precision="ieee")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, acc, mask=c_mask)


def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch split_k_matmul_kernel with SPLIT_K K-partitions and atomic_add."""
    M, K = A.shape
    K2, N = B.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    SPLIT_K = 4
    out.zero_()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * SPLIT_K,)
    split_k_matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
    )