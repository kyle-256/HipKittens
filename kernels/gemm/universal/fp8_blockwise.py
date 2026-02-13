"""
FP8 Blockwise GEMM Kernel for AMD MI355X — Forward + Backward

Block-wise quantization (DeepGEMM-style):
  - A_scales: [M, K/128]      per-row, per-K-block scaling
  - B_scales: [N/128, K/128]  2D block scaling for weights
  - block_size = 128 (fixed, matches quantization granularity)

Layout: NT  (A[M,K] @ B[N,K].T -> C[M,N])

Forward:
  C[m,n] = sum_k A_dq[m,k] * B_dq[n,k]
  where A_dq = A_fp8 * A_scales, B_dq = B_fp8 * B_scales

Backward (dgrad):
  dA[m,k] = sum_n grad_out[m,n] * B_dq[n,k]
  (Simplified: uses BF16 for backward since FP8 backward is typically less critical)

Backward (wgrad):
  dB[n,k] = sum_m grad_out[m,n] * A_dq[m,k]

MI355X Optimizations:
  - XCD-aware PID mapping (8 XCDs)
  - Pre-transposed A_scales for coalesced access
  - NN-style dot (B loaded as [K,N]) avoids tl.trans
  - Persistent kernel, num_warps=4, num_stages=2
"""

import torch
import triton
import triton.language as tl

NUM_XCDS = 8


def _get_num_cus():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _set_amd_knobs(use_async=True, scalarize=True):
    if hasattr(triton, 'knobs') and hasattr(triton.knobs, 'amd'):
        triton.knobs.amd.use_async_copy = use_async
        triton.knobs.amd.scalarize_packed_fops = scalarize


@triton.jit
def _chiplet_transform(pid, NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, CHUNK: tl.constexpr):
    chunk_id = pid // CHUNK
    chunk_off = pid % CHUNK
    cta_per_xcd = NUM_SMS // NUM_XCDS
    return (chunk_id % NUM_XCDS) * cta_per_xcd + (chunk_id // NUM_XCDS) * CHUNK + chunk_off


# ============================================================
# Forward Kernel
# ============================================================

@triton.jit
def _blockwise_fp8_fwd_kernel(
    A_ptr, B_ptr, C_ptr,
    A_scales_ptr, B_scales_ptr,
    M, N, K,
    stride_am, stride_bn,
    stride_cm, stride_cn,
    stride_as_k, stride_as_m,
    stride_bs_n, stride_bs_k,
    stride_ak: tl.constexpr, stride_bk: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, CHUNK: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform(pid, NUM_SMS, NUM_XCDS, CHUNK)

    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    total = num_m * num_n
    grp = GROUP_M * num_n

    tl.assume(stride_am > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    for tid in range(pid, total, NUM_SMS):
        gid = tid // grp
        fm = gid * GROUP_M
        gs = min(num_m - fm, GROUP_M)
        pm = fm + (tid % grp) % gs
        pn = (tid % grp) // gs
        tl.assume(pm >= 0)
        tl.assume(pn >= 0)

        rm = tl.max_contiguous(tl.multiple_of((pm * BLOCK_M + tl.arange(0, BLOCK_M)) % M, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of((pn * BLOCK_N + tl.arange(0, BLOCK_N)) % N, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        as_ptrs = A_scales_ptr + rm * stride_as_m
        bs_ptr_base = B_scales_ptr + pn * stride_bs_n

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for ki in range(NUM_K_BLOCKS):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(a_ptrs, (1, 16)), cache_modifier='.ca')
            else:
                a = tl.load(tl.multiple_of(a_ptrs, (16, 1)), cache_modifier='.ca')
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(b_ptrs, (16, 1)), cache_modifier='.ca')
            else:
                b = tl.load(tl.multiple_of(b_ptrs, (1, 16)), cache_modifier='.ca')

            partial = tl.dot(a, b, input_precision='ieee')

            a_s = tl.load(as_ptrs + ki * stride_as_k)
            b_s = tl.load(bs_ptr_base + ki * stride_bs_k)

            scale = a_s * b_s
            acc += partial * scale[:, None]

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask)


# ============================================================
# Backward dgrad kernel: dA[M,K] = grad_out[M,N] @ B[N,K] (with blockwise scales)
# ============================================================

@triton.jit
def _blockwise_fp8_dgrad_kernel(
    GRAD_ptr, B_ptr, DA_ptr,
    B_scales_ptr,
    M, N, K,
    stride_gm, stride_gn,
    stride_bn, stride_bk,
    stride_dam, stride_dak,
    stride_bs_n, stride_bs_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, CHUNK: tl.constexpr,
):
    """dA = grad_out @ B_dq. B_dq needs B_scales applied per block.
    grad_out is [M,N] bf16, B is [N,K] fp8 with B_scales[N/128, K/128].
    dA is [M,K] bf16.

    We tile over M,K and reduce over N (with blockwise B_scales).
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform(pid, NUM_SMS, NUM_XCDS, CHUNK)

    num_m = tl.cdiv(M, BLOCK_M)
    num_k = tl.cdiv(K, BLOCK_K)
    total = num_m * num_k
    grp = GROUP_M * num_k

    tl.assume(stride_gm > 0)
    tl.assume(stride_dam > 0)

    num_n_blocks = tl.cdiv(N, BLOCK_N)

    for tid in range(pid, total, NUM_SMS):
        gid = tid // grp
        fm = gid * GROUP_M
        gs = min(num_m - fm, GROUP_M)
        pm = fm + (tid % grp) % gs
        pk = (tid % grp) // gs
        tl.assume(pm >= 0)
        tl.assume(pk >= 0)

        rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        rk = pk * BLOCK_K + tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for ni in range(num_n_blocks):
            rn = ni * BLOCK_N + tl.arange(0, BLOCK_N)

            # Load grad_out[rm, rn]
            g_ptrs = GRAD_ptr + rm[:, None] * stride_gm + rn[None, :] * stride_gn
            g_mask = (rm[:, None] < M) & (rn[None, :] < N)
            grad_tile = tl.load(g_ptrs, mask=g_mask, other=0.0)

            # Load B[rn, rk] (B is stored as [N,K])
            b_ptrs = B_ptr + rn[:, None] * stride_bn + rk[None, :] * stride_bk
            b_mask = (rn[:, None] < N) & (rk[None, :] < K)
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

            # Apply B_scales for this block and convert to BF16 for fast dot
            b_scale = tl.load(B_scales_ptr + ni * stride_bs_n + pk * stride_bs_k)
            b_tile_dq = (b_tile * b_scale).to(tl.bfloat16)

            # Accumulate: dA += grad_out @ B_dq (BF16 dot with FP32 accum)
            acc += tl.dot(grad_tile.to(tl.bfloat16), b_tile_dq)

        # Store dA
        da_ptrs = DA_ptr + rm[:, None] * stride_dam + rk[None, :] * stride_dak
        da_mask = (rm[:, None] < M) & (rk[None, :] < K)
        tl.store(da_ptrs, acc.to(tl.bfloat16), mask=da_mask)


# ============================================================
# Backward wgrad kernel: dB[N,K] = grad_out[M,N].T @ A_dq[M,K]
# ============================================================

@triton.jit
def _blockwise_fp8_wgrad_kernel(
    GRAD_ptr, A_ptr, DB_ptr,
    A_scales_ptr,
    M, N, K,
    stride_gm, stride_gn,
    stride_am, stride_ak,
    stride_dbn, stride_dbk,
    stride_as_m, stride_as_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, CHUNK: tl.constexpr,
):
    """dB = grad_out^T @ A_dq. A_dq needs A_scales applied per block.
    grad_out is [M,N] bf16, A is [M,K] fp8 with A_scales[M, K/128].
    dB is [N,K] bf16.

    We tile over N,K and reduce over M.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform(pid, NUM_SMS, NUM_XCDS, CHUNK)

    num_n = tl.cdiv(N, BLOCK_N)
    num_k = tl.cdiv(K, BLOCK_K)
    total = num_n * num_k
    grp = GROUP_M * num_k

    tl.assume(stride_gm > 0)
    tl.assume(stride_dbn > 0)

    num_m_blocks = tl.cdiv(M, BLOCK_M)

    for tid in range(pid, total, NUM_SMS):
        gid = tid // grp
        fn = gid * GROUP_M
        gs = min(num_n - fn, GROUP_M)
        pn = fn + (tid % grp) % gs
        pk = (tid % grp) // gs
        tl.assume(pn >= 0)
        tl.assume(pk >= 0)

        rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = pk * BLOCK_K + tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for mi in range(num_m_blocks):
            rm = mi * BLOCK_M + tl.arange(0, BLOCK_M)

            # Load grad_out[rm, rn]
            g_ptrs = GRAD_ptr + rm[:, None] * stride_gm + rn[None, :] * stride_gn
            g_mask = (rm[:, None] < M) & (rn[None, :] < N)
            grad_tile = tl.load(g_ptrs, mask=g_mask, other=0.0)

            # Load A[rm, rk] (A is [M,K])
            a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
            a_mask = (rm[:, None] < M) & (rk[None, :] < K)
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

            # Apply A_scales for this block and convert to BF16
            a_scales_block = tl.load(
                A_scales_ptr + rm * stride_as_m + pk * stride_as_k,
                mask=rm < M, other=0.0
            )
            a_tile_dq = (a_tile * a_scales_block[:, None]).to(tl.bfloat16)

            # Accumulate: dB += grad_out^T @ A_dq => [N, M] @ [M, K] = [N, K]
            # Use BF16 dot with FP32 accumulation for performance
            acc += tl.dot(tl.trans(grad_tile.to(tl.bfloat16)), a_tile_dq)

        # Store dB
        db_ptrs = DB_ptr + rn[:, None] * stride_dbn + rk[None, :] * stride_dbk
        db_mask = (rn[:, None] < N) & (rk[None, :] < K)
        tl.store(db_ptrs, acc.to(tl.bfloat16), mask=db_mask)


# ============================================================
# Public API
# ============================================================

def blockwise_fp8_gemm_forward(
    A: torch.Tensor,        # [M, K] fp8
    B: torch.Tensor,        # [N, K] fp8
    A_scales: torch.Tensor, # [M, K//128] fp32
    B_scales: torch.Tensor, # [N//128, K//128] fp32
    out: torch.Tensor = None,
) -> torch.Tensor:
    """FP8 blockwise GEMM forward: C = (A @ B.T) * block_scales."""
    _set_amd_knobs(True, True)

    assert A.dtype == torch.float8_e4m3fn and A.is_contiguous()
    assert B.dtype == torch.float8_e4m3fn and B.is_contiguous()
    M, K = A.shape
    N = B.shape[0]
    assert B.shape[1] == K

    BLOCK_K = 128
    num_k = K // BLOCK_K
    num_n = (N + BLOCK_K - 1) // BLOCK_K

    if out is None:
        out = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)

    A_scales_t = A_scales.T.contiguous()  # [K//128, M]
    B_t = B.T  # view [K, N]

    num_tiles_m = (M + 127) // 128
    num_tiles_n = (N + 127) // 128
    NUM_SMS = num_tiles_m * num_tiles_n
    CHUNK = 64
    num_xcds = NUM_XCDS
    if NUM_SMS // num_xcds < CHUNK:
        num_xcds = 1

    _blockwise_fp8_fwd_kernel[(NUM_SMS,)](
        A, B_t, out,
        A_scales_t, B_scales,
        M, N, K,
        A.stride(0), B_t.stride(1),
        out.stride(0), out.stride(1),
        A_scales_t.stride(0), A_scales_t.stride(1),
        B_scales.stride(0), B_scales.stride(1),
        stride_ak=1, stride_bk=1,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
        GROUP_M=8, NUM_SMS=NUM_SMS, NUM_XCDS=num_xcds, CHUNK=CHUNK,
        NUM_K_BLOCKS=num_k,
        num_warps=4, num_stages=2,
        matrix_instr_nonkdim=32, waves_per_eu=2,
    )
    return out


def blockwise_fp8_gemm_backward(
    grad_out: torch.Tensor,  # [M, N] bf16
    A: torch.Tensor,         # [M, K] fp8
    B: torch.Tensor,         # [N, K] fp8
    A_scales: torch.Tensor,  # [M, K//128] fp32
    B_scales: torch.Tensor,  # [N//128, K//128] fp32
) -> tuple:
    """FP8 blockwise GEMM backward.

    Returns (dA, dB) both in BF16.
    dA = grad_out @ B_dq  (B_dq = B * B_scales)
    dB = grad_out^T @ A_dq  (A_dq = A * A_scales)
    """
    _set_amd_knobs(True, True)

    M, N = grad_out.shape
    K = A.shape[1]
    BLOCK_SIZE = 128

    num_cus = _get_num_cus()
    CHUNK = 64
    num_xcds = NUM_XCDS

    # --- dgrad: dA[M,K] = grad_out[M,N] @ B_dq[N,K] ---
    dA = torch.empty((M, K), device=grad_out.device, dtype=torch.bfloat16)

    num_tiles_m = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_tiles_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    NUM_SMS_dA = num_tiles_m * num_tiles_k
    num_xcds_dA = num_xcds if NUM_SMS_dA // num_xcds >= CHUNK else 1

    _blockwise_fp8_dgrad_kernel[(NUM_SMS_dA,)](
        grad_out, B, dA,
        B_scales,
        M, N, K,
        grad_out.stride(0), grad_out.stride(1),
        B.stride(0), B.stride(1),
        dA.stride(0), dA.stride(1),
        B_scales.stride(0), B_scales.stride(1),
        BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, BLOCK_K=BLOCK_SIZE,
        GROUP_M=4, NUM_SMS=NUM_SMS_dA, NUM_XCDS=num_xcds_dA, CHUNK=CHUNK,
        num_warps=4, num_stages=2,
        matrix_instr_nonkdim=32, waves_per_eu=2,
    )

    # --- wgrad: dB[N,K] = grad_out^T[N,M] @ A_dq[M,K] ---
    dB = torch.empty((N, K), device=grad_out.device, dtype=torch.bfloat16)

    num_tiles_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    NUM_SMS_dB = num_tiles_n * num_tiles_k
    num_xcds_dB = num_xcds if NUM_SMS_dB // num_xcds >= CHUNK else 1

    _blockwise_fp8_wgrad_kernel[(NUM_SMS_dB,)](
        grad_out, A, dB,
        A_scales,
        M, N, K,
        grad_out.stride(0), grad_out.stride(1),
        A.stride(0), A.stride(1),
        dB.stride(0), dB.stride(1),
        A_scales.stride(0), A_scales.stride(1),
        BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, BLOCK_K=BLOCK_SIZE,
        GROUP_M=4, NUM_SMS=NUM_SMS_dB, NUM_XCDS=num_xcds_dB, CHUNK=CHUNK,
        num_warps=4, num_stages=2,
        matrix_instr_nonkdim=32, waves_per_eu=2,
    )

    return dA, dB


# ============================================================
# Testing
# ============================================================

def test_correctness():
    print("=" * 60)
    print("FP8 Blockwise GEMM Correctness Tests")
    print("=" * 60)

    all_pass = True

    for sz_name, M, N, K in [("Small", 512, 512, 512), ("Medium", 1024, 1024, 1024), ("Large", 2048, 2048, 2048)]:
        BK = BN = 128
        num_k = K // BK
        num_n = N // BN

        A_f32 = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.1
        B_f32 = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.1
        A_scales = torch.randn(M, num_k, device="cuda", dtype=torch.float32).abs() + 0.1
        B_scales = torch.randn(num_n, num_k, device="cuda", dtype=torch.float32).abs() + 0.1

        A_fp8 = (A_f32 / A_scales.repeat_interleave(BK, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        B_scales_exp = B_scales.repeat_interleave(BN, 0).repeat_interleave(BK, 1)
        B_fp8 = (B_f32 / B_scales_exp).clamp(-448, 448).to(torch.float8_e4m3fn)

        # --- Forward Test ---
        C = blockwise_fp8_gemm_forward(A_fp8, B_fp8, A_scales, B_scales)

        # Reference
        C_ref = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        A_dq = A_fp8.float()
        B_dq = B_fp8.float()
        for kb in range(num_k):
            ks, ke = kb * BK, (kb + 1) * BK
            A_block = A_dq[:, ks:ke] * A_scales[:, kb:kb+1]
            for nb in range(num_n):
                ns, ne = nb * BN, (nb + 1) * BN
                B_block = B_dq[ns:ne, ks:ke] * B_scales[nb, kb]
                C_ref[:, ns:ne] += A_block @ B_block.T

        cos = torch.nn.functional.cosine_similarity(
            C.float().reshape(1, -1), C_ref.reshape(1, -1)).item()
        ok = cos > 0.999
        if not ok: all_pass = False
        print(f"  Fwd {sz_name:6s} M={M:4d} N={N:4d} K={K:4d}: cos={cos:.6f} {'PASS' if ok else 'FAIL'}")

        # --- Backward Test ---
        grad_out = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.01
        dA, dB = blockwise_fp8_gemm_backward(grad_out, A_fp8, B_fp8, A_scales, B_scales)

        # Reference dA = grad_out @ B_dq
        B_dq_full = torch.zeros(N, K, device="cuda", dtype=torch.float32)
        for kb in range(num_k):
            ks, ke = kb * BK, (kb + 1) * BK
            for nb in range(num_n):
                ns, ne = nb * BN, (nb + 1) * BN
                B_dq_full[ns:ne, ks:ke] = B_dq[ns:ne, ks:ke] * B_scales[nb, kb]
        dA_ref = grad_out.float() @ B_dq_full

        cos_dA = torch.nn.functional.cosine_similarity(
            dA.float().reshape(1, -1), dA_ref.reshape(1, -1)).item()
        ok_dA = cos_dA > 0.99
        if not ok_dA: all_pass = False
        print(f"  dA  {sz_name:6s} M={M:4d} N={N:4d} K={K:4d}: cos={cos_dA:.6f} {'PASS' if ok_dA else 'FAIL'}")

        # Reference dB = grad_out^T @ A_dq
        A_dq_full = torch.zeros(M, K, device="cuda", dtype=torch.float32)
        for kb in range(num_k):
            ks, ke = kb * BK, (kb + 1) * BK
            A_dq_full[:, ks:ke] = A_dq[:, ks:ke] * A_scales[:, kb:kb+1]
        dB_ref = grad_out.float().T @ A_dq_full

        cos_dB = torch.nn.functional.cosine_similarity(
            dB.float().reshape(1, -1), dB_ref.reshape(1, -1)).item()
        ok_dB = cos_dB > 0.99
        if not ok_dB: all_pass = False
        print(f"  dB  {sz_name:6s} M={M:4d} N={N:4d} K={K:4d}: cos={cos_dB:.6f} {'PASS' if ok_dB else 'FAIL'}")

    print("\n" + ("ALL PASSED" if all_pass else "SOME FAILED"))
    return all_pass


if __name__ == "__main__":
    test_correctness()

