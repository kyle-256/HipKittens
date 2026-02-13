"""
Grouped GEMM Triton Kernels — BF16 + FP8 Per-Tensor, Forward + Backward

For MoE-style grouped GEMM: multiple groups with different M sizes, shared K and N.

Forward:
  out[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B_view[g] [* scale]

Backward (variable-K):
  dB[g] = A_g^T @ grad_out_g [* scale]

Supports:
  - BF16 forward/backward
  - FP8 per-tensor forward/backward

MI355X Optimizations:
  - Persistent kernel, single launch for all groups
  - XCD-aware PID mapping (8 XCDs)
  - GROUP_SIZE_M swizzling for L2 cache
  - Zero CPU synchronization (group mapping computed on GPU)
"""

import torch
import triton
import triton.language as tl

NUM_XCDS = 8


def _get_num_cus():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _set_amd_knobs(enable=True):
    if hasattr(triton, 'knobs') and hasattr(triton.knobs, 'amd'):
        triton.knobs.amd.use_async_copy = enable
        triton.knobs.amd.scalarize_packed_fops = enable


@triton.jit
def _chiplet_transform(pid, NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, CHUNK_SIZE: tl.constexpr):
    if pid > (NUM_SMS // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    local_pid = pid // NUM_XCDS
    chunk_idx = local_pid // CHUNK_SIZE
    pos_in_chunk = local_pid % CHUNK_SIZE
    xcd = pid % NUM_XCDS
    return chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk


# ============================================================
# Forward Kernel (shared by BF16 and FP8)
# ============================================================

@triton.jit()
def _grouped_gemm_fwd_kernel(
    A, B, C,
    A_scale_ptr, B_scale_ptr,
    group_offs_ptr,
    G, N, K,
    stride_am, stride_bg, stride_bn, stride_cm, stride_cn,
    stride_ak: tl.constexpr, stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr, IS_FP8: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Compute total tiles across all groups
    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    if IS_FP8:
        scale = tl.load(A_scale_ptr) * tl.load(B_scale_ptr)

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        # Find group via linear scan
        group_idx: tl.int32 = 0
        tile_start: tl.int32 = 0
        cumsum: tl.int32 = 0
        for _g in range(G):
            m_g_i = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
            tiles_g = tl.cdiv(m_g_i, BLOCK_SIZE_M) * num_pid_n
            new_cumsum = cumsum + tiles_g
            if global_tile_id >= new_cumsum:
                group_idx = _g + 1
                tile_start = new_cumsum
            cumsum = new_cumsum

        local_tile = global_tile_id - tile_start
        m_start_g = tl.load(group_offs_ptr + group_idx)
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - tl.load(group_offs_ptr + group_idx)).to(tl.int32)
        tiles_m_g = tl.cdiv(M_g, BLOCK_SIZE_M)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m_g - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        group_offset_b = group_idx.to(tl.int64) * stride_bg
        A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 1)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier='.ca')
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier='.ca')
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier='.ca')
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier='.ca')
            if IS_FP8:
                acc += tl.dot(a, b, input_precision='ieee')
            else:
                acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + m_start_g * stride_am + rm[:, None] * stride_am + rk_last[None, :] * stride_ak
            B_LAST = B + group_offset_b + rk_last[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_LAST, mask=rk_last[None, :] < K, other=0.0, cache_modifier='.ca')
            b = tl.load(B_LAST, mask=rk_last[:, None] < K, other=0.0, cache_modifier='.ca')
            if IS_FP8:
                acc += tl.dot(a, b, input_precision='ieee')
            else:
                acc += tl.dot(a, b)

        if IS_FP8:
            acc *= scale
        c = acc.to(C.type.element_ty)
        rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
        C_ = C + m_start_g * stride_cm + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


# ============================================================
# Backward Kernel: dB[g] = A_g^T @ grad_out_g
# Variable-K (M_g varies per group), fixed output dims (K x N)
# ============================================================

@triton.jit()
def _grouped_gemm_bwd_kernel(
    LHS, RHS, C,
    LHS_scale_ptr, RHS_scale_ptr,
    group_offs_ptr,
    G, OUT_M, OUT_N,
    stride_lhs_m, stride_rhs_m,
    stride_cg, stride_cm, stride_cn,
    stride_lhs_n: tl.constexpr, stride_rhs_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    tiles_m = tl.cdiv(OUT_M, BLOCK_SIZE_M)
    tiles_n = tl.cdiv(OUT_N, BLOCK_SIZE_N)
    tiles_per_group = tiles_m * tiles_n
    total_tiles = G * tiles_per_group

    tl.assume(stride_lhs_m > 0)
    tl.assume(stride_rhs_m > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    if IS_FP8:
        scale = tl.load(LHS_scale_ptr) * tl.load(RHS_scale_ptr)

    for global_tile in range(pid, total_tiles, NUM_SMS):
        group_idx = global_tile // tiles_per_group
        local_tile = global_tile - group_idx * tiles_per_group

        num_pid_in_group = GROUP_SIZE_M * tiles_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        m_start = tl.load(group_offs_ptr + group_idx)
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - m_start).to(tl.int32)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % OUT_M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % OUT_N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # LHS^T[rm, rk] = LHS[m_start + rk, rm]
        LHS_BASE = LHS + m_start * stride_lhs_m + rm[:, None] * stride_lhs_n + rk[None, :] * stride_lhs_m
        RHS_BASE = RHS + m_start * stride_rhs_m + rk[:, None] * stride_rhs_m + rn[None, :] * stride_rhs_n

        loop_k = tl.cdiv(M_g, BLOCK_SIZE_K)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(loop_k):
            k_start = k * BLOCK_SIZE_K
            mask_k = (k_start + tl.arange(0, BLOCK_SIZE_K)) < M_g
            if stride_lhs_n == 1:
                a = tl.load(tl.multiple_of(LHS_BASE, (16, 1)), mask=mask_k[None, :], other=0.0, cache_modifier='.ca')
            else:
                a = tl.load(tl.multiple_of(LHS_BASE, (1, 16)), mask=mask_k[None, :], other=0.0, cache_modifier='.ca')
            if stride_rhs_n == 1:
                b = tl.load(tl.multiple_of(RHS_BASE, (1, 16)), mask=mask_k[:, None], other=0.0, cache_modifier='.ca')
            else:
                b = tl.load(tl.multiple_of(RHS_BASE, (16, 1)), mask=mask_k[:, None], other=0.0, cache_modifier='.ca')
            if IS_FP8:
                acc += tl.dot(a, b, input_precision='ieee')
            else:
                acc += tl.dot(a, b)
            LHS_BASE += BLOCK_SIZE_K * stride_lhs_m
            RHS_BASE += BLOCK_SIZE_K * stride_rhs_m

        if IS_FP8:
            acc *= scale
        c = acc.to(C.type.element_ty)
        rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s % OUT_N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < OUT_M) & (rn_s[None, :] < OUT_N)
        C_ = C + group_idx.to(tl.int64) * stride_cg + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)


# ============================================================
# Public API — BF16 Grouped GEMM
# ============================================================

def grouped_gemm_bf16_forward(
    a: torch.Tensor,       # [M_total, K] bf16
    b: torch.Tensor,       # [G, K, N] or [G, N, K] bf16
    group_offs: torch.Tensor,  # [G+1] int64
    trans_b: bool = False,
) -> torch.Tensor:
    """BF16 grouped GEMM forward."""
    _set_amd_knobs(True)
    M_total, K_a = a.shape
    G = b.shape[0]
    if trans_b:
        N, K_b = b.shape[1], b.shape[2]
        stride_bk = b.stride(2)
        stride_bn = b.stride(1)
    else:
        K_b, N = b.shape[1], b.shape[2]
        stride_bk = b.stride(1)
        stride_bn = b.stride(2)
    assert K_a == K_b
    K = K_a

    out = torch.empty((M_total, N), device=a.device, dtype=a.dtype)
    num_sms = _get_num_cus()
    even_k = K % 64 == 0
    dummy_scale = torch.empty(1, device=a.device, dtype=torch.float32)

    _grouped_gemm_fwd_kernel[(num_sms,)](
        a, b, out, dummy_scale, dummy_scale,
        group_offs, G, N, K,
        a.stride(0), b.stride(0), stride_bn, out.stride(0), out.stride(1),
        stride_ak=a.stride(1), stride_bk=stride_bk,
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4, NUM_SMS=num_sms, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=32,
        EVEN_K=even_k, IS_FP8=False,
        num_warps=8, num_stages=2, waves_per_eu=2,
        matrix_instr_nonkdim=16, kpack=1,
    )
    return out


def grouped_gemm_bf16_backward(
    lhs: torch.Tensor,        # [M_total, K] bf16  (e.g. input a)
    rhs: torch.Tensor,        # [M_total, N] bf16  (e.g. grad_out)
    group_offs: torch.Tensor,  # [G+1] int64
) -> torch.Tensor:
    """BF16 grouped GEMM backward: dB[g] = lhs_g^T @ rhs_g.

    Returns [G, K, N] tensor.
    """
    _set_amd_knobs(True)
    OUT_M = lhs.shape[1]  # K
    OUT_N = rhs.shape[1]  # N
    G = group_offs.shape[0] - 1

    out = torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=lhs.dtype)
    num_sms = _get_num_cus()
    dummy_scale = torch.empty(1, device=lhs.device, dtype=torch.float32)

    _grouped_gemm_bwd_kernel[(num_sms,)](
        lhs, rhs, out, dummy_scale, dummy_scale,
        group_offs, G, OUT_M, OUT_N,
        lhs.stride(0), rhs.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        stride_lhs_n=lhs.stride(1), stride_rhs_n=rhs.stride(1),
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4, NUM_SMS=num_sms, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=32,
        IS_FP8=False,
        num_warps=8, num_stages=2, waves_per_eu=2,
        matrix_instr_nonkdim=16, kpack=1,
    )
    return out


# ============================================================
# Public API — FP8 Per-Tensor Grouped GEMM
# ============================================================

def grouped_gemm_fp8_forward(
    a: torch.Tensor,       # [M_total, K] fp8
    b: torch.Tensor,       # [G, K, N] or [G, N, K] fp8
    a_scale: torch.Tensor, # scalar fp32
    b_scale: torch.Tensor, # scalar fp32
    group_offs: torch.Tensor,  # [G+1] int64
    trans_b: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP8 per-tensor grouped GEMM forward."""
    _set_amd_knobs(True)
    M_total, K_a = a.shape
    G = b.shape[0]
    if trans_b:
        N, K_b = b.shape[1], b.shape[2]
        stride_bk = b.stride(2)
        stride_bn = b.stride(1)
    else:
        K_b, N = b.shape[1], b.shape[2]
        stride_bk = b.stride(1)
        stride_bn = b.stride(2)
    assert K_a == K_b
    K = K_a

    out = torch.empty((M_total, N), device=a.device, dtype=out_dtype)
    num_sms = _get_num_cus()
    blk_k = 128 if (a.stride(1) == 1 and stride_bk == 1) else 64
    even_k = K % blk_k == 0

    _grouped_gemm_fwd_kernel[(num_sms,)](
        a, b, out, a_scale, b_scale,
        group_offs, G, N, K,
        a.stride(0), b.stride(0), stride_bn, out.stride(0), out.stride(1),
        stride_ak=a.stride(1), stride_bk=stride_bk,
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=blk_k,
        GROUP_SIZE_M=4, NUM_SMS=num_sms, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=32,
        EVEN_K=even_k, IS_FP8=True,
        num_warps=8, num_stages=2, waves_per_eu=0,
        matrix_instr_nonkdim=16, kpack=1,
    )
    return out


def grouped_gemm_fp8_backward(
    lhs: torch.Tensor,        # [M_total, K] fp8
    rhs: torch.Tensor,        # [M_total, N] fp8
    lhs_scale: torch.Tensor,  # scalar fp32
    rhs_scale: torch.Tensor,  # scalar fp32
    group_offs: torch.Tensor,  # [G+1] int64
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP8 per-tensor grouped GEMM backward: dB[g] = lhs_g^T @ rhs_g * scale.

    Returns [G, K, N] tensor.
    """
    _set_amd_knobs(True)
    OUT_M = lhs.shape[1]  # K
    OUT_N = rhs.shape[1]  # N
    G = group_offs.shape[0] - 1

    out = torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=out_dtype)
    num_sms = _get_num_cus()

    _grouped_gemm_bwd_kernel[(num_sms,)](
        lhs, rhs, out, lhs_scale, rhs_scale,
        group_offs, G, OUT_M, OUT_N,
        lhs.stride(0), rhs.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        stride_lhs_n=lhs.stride(1), stride_rhs_n=rhs.stride(1),
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4, NUM_SMS=num_sms, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=32,
        IS_FP8=True,
        num_warps=8, num_stages=2, waves_per_eu=0,
        matrix_instr_nonkdim=16, kpack=1,
    )
    return out


# ============================================================
# Testing
# ============================================================

def test_correctness():
    print("=" * 60)
    print("Grouped GEMM Correctness Tests")
    print("=" * 60)

    device = 'cuda'
    G, M_per_group, K, N = 4, 512, 4096, 4096

    all_pass = True

    # --- BF16 Forward ---
    print("\n--- BF16 Forward ---")
    a_bf16 = torch.randn(G * M_per_group, K, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(G, K, N, device=device, dtype=torch.bfloat16)
    offs = torch.arange(0, (G + 1) * M_per_group, M_per_group, device=device, dtype=torch.int64)

    out = grouped_gemm_bf16_forward(a_bf16, b_bf16, offs, trans_b=False)

    # Reference
    for g in range(G):
        ref = a_bf16[offs[g]:offs[g+1]].float() @ b_bf16[g].float()
        cos = torch.nn.functional.cosine_similarity(
            out[offs[g]:offs[g+1]].float().reshape(1, -1), ref.reshape(1, -1)).item()
        ok = cos > 0.999
        if not ok: all_pass = False
        print(f"  Group {g}: cos={cos:.6f} {'PASS' if ok else 'FAIL'}")

    # --- BF16 Backward ---
    print("\n--- BF16 Backward ---")
    grad_out = torch.randn(G * M_per_group, N, device=device, dtype=torch.bfloat16)
    dB = grouped_gemm_bf16_backward(a_bf16, grad_out, offs)

    for g in range(G):
        ref = a_bf16[offs[g]:offs[g+1]].float().T @ grad_out[offs[g]:offs[g+1]].float()
        cos = torch.nn.functional.cosine_similarity(
            dB[g].float().reshape(1, -1), ref.reshape(1, -1)).item()
        ok = cos > 0.999
        if not ok: all_pass = False
        print(f"  Group {g}: cos={cos:.6f} {'PASS' if ok else 'FAIL'}")

    # --- FP8 Forward ---
    print("\n--- FP8 Per-Tensor Forward ---")
    a_fp8 = torch.randn(G * M_per_group, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b_fp8 = torch.randn(G, K, N, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    a_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    b_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    out_fp8 = grouped_gemm_fp8_forward(a_fp8, b_fp8, a_scale, b_scale, offs, trans_b=False)

    for g in range(G):
        ref = a_fp8[offs[g]:offs[g+1]].float() @ b_fp8[g].float()
        cos = torch.nn.functional.cosine_similarity(
            out_fp8[offs[g]:offs[g+1]].float().reshape(1, -1), ref.reshape(1, -1)).item()
        ok = cos > 0.999
        if not ok: all_pass = False
        print(f"  Group {g}: cos={cos:.6f} {'PASS' if ok else 'FAIL'}")

    # --- FP8 Backward ---
    print("\n--- FP8 Per-Tensor Backward ---")
    lhs_fp8 = a_fp8
    rhs_fp8 = torch.randn(G * M_per_group, N, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    dB_fp8 = grouped_gemm_fp8_backward(lhs_fp8, rhs_fp8, a_scale, b_scale, offs)

    for g in range(G):
        ref = lhs_fp8[offs[g]:offs[g+1]].float().T @ rhs_fp8[offs[g]:offs[g+1]].float()
        cos = torch.nn.functional.cosine_similarity(
            dB_fp8[g].float().reshape(1, -1), ref.reshape(1, -1)).item()
        ok = cos > 0.999
        if not ok: all_pass = False
        print(f"  Group {g}: cos={cos:.6f} {'PASS' if ok else 'FAIL'}")

    print("\n" + ("ALL PASSED" if all_pass else "SOME FAILED"))
    return all_pass


if __name__ == "__main__":
    test_correctness()

