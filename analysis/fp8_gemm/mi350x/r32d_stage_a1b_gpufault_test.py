"""R32 Dev D — Stage A1b GPU-fault test for rect-V2 RCR fastpath.

Calls the rect-V2 RCR fastpath kernel with intentionally over-allocated scale
buffers so that even an incorrectly-sized rect slab geometry does not address
out-of-bounds. Goal: verify the rect kernel does NOT GPU-fault, segfault, or
kernel-timeout.

Numerical correctness is OUT OF SCOPE for Stage A1b — the host preshuffle is
the square layout (R29 b_pack_count=2, blk=256), but the rect kernel uses
pack_count=1 + blk_n=128. Numerics will be wrong; this test only validates
"no fault".

Usage: HIP_VISIBLE_DEVICES=3 python3 r32d_stage_a1b_gpufault_test.py
"""
import math
import os
import sys
import time

import torch

torch.manual_seed(0)

import tk_mxfp8_layouts

# 4096^3 — the structural target for V2-RCR rect (R31 Reviewer ratio 0.9229,
# 16% CUs idle at BLK=256). Rect BLK_N=128 -> 16x32 = 512 tiles vs square 256.
M, N, K = 4096, 4096, 4096
print(f"[r32d-A1b] starting GPU-fault test M={M} N={N} K={K}", flush=True)


def gen_fp8(rows, cols):
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda") * 0.05
    return x.to(torch.float8_e4m3fn)


def gen_scale(rows, kb):
    return torch.randint(-2, 3, (rows, kb), dtype=torch.int8, device="cuda")


def encode_raw(s):
    raw = (s.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(s == -128, torch.full_like(raw, 0xFF), raw)


# Square preshuffle (intentionally — we are NOT testing correctness).
# Reused from R29A bench (same layout used for V2-RCR/V2-CRR A side).
def preshuffle_v2_a(scale_exp, blk=256, hb=128, rbm=64, warps_m=2):
    pack_count = 4
    rows, kb = scale_exp.shape
    padded_rows = math.ceil(rows / blk) * blk
    padded_kb = math.ceil(kb / 8) * 8
    num_ctiles = padded_rows // blk
    num_slabs = num_ctiles * warps_m
    rgs_per_ctile = blk // 32
    pack_a = rbm // 32
    raw = torch.full((padded_rows, padded_kb), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = encode_raw(scale_exp)
    perm = torch.empty_like(raw)
    rg_view = raw.view(num_ctiles, rgs_per_ctile, 32, padded_kb)
    perm_view = perm.view(num_ctiles, num_slabs // num_ctiles, pack_count, 32, padded_kb)
    for wm in range(warps_m):
        rg_base = wm * (rbm // 32)
        rg_hi = (hb // 32) + rg_base
        for pidx in range(pack_a):
            perm_view[:, wm, 2*pidx, :, :] = rg_view[:, rg_base + pidx, :, :]
            perm_view[:, wm, 2*pidx + 1, :, :] = rg_view[:, rg_hi + pidx, :, :]
    kp_count = padded_kb // 8
    rv = perm.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    sh = rv.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return sh.view(num_slabs, pack_count * 32 * padded_kb)


# Rect-tuned over-allocated B preshuffle: use a buffer of the rect kernel's
# expected size (num_slabs_rect * pack_count_rect * 32 * padded_kb) — even
# filled with zeros, the rect kernel will read in-bounds → no GPU fault.
def preshuffle_v2_b_overalloc_for_rect(scale_exp, blk_n_rect=128, warps_n=4):
    rows, kb = scale_exp.shape  # rows = N, kb = K/32
    padded_rows = math.ceil(rows / blk_n_rect) * blk_n_rect
    padded_kb = math.ceil(kb / 8) * 8
    num_ctiles_rect = padded_rows // blk_n_rect
    num_slabs_rect = num_ctiles_rect * warps_n  # N=4096 BLK_N=128 -> 32*4=128 slabs
    pack_count_rect = 1
    out = torch.zeros(num_slabs_rect, pack_count_rect * 32 * padded_kb,
                      dtype=torch.uint8, device=scale_exp.device)
    return out


k_blocks = (K + 31) // 32
# RCR: A is row-layout (M x K), B is col-layout (K x N).
A = torch.randn(M, K, dtype=torch.float32, device="cuda").mul_(0.05).to(torch.float8_e4m3fn)
B = torch.randn(K, N, dtype=torch.float32, device="cuda").mul_(0.05).to(torch.float8_e4m3fn)
Ase = gen_scale(M, k_blocks)
Bse = gen_scale(N, k_blocks)

As = preshuffle_v2_a(Ase)
Bs = preshuffle_v2_b_overalloc_for_rect(Bse)
print(f"[r32d-A1b] As.shape={As.shape} Bs.shape={Bs.shape}", flush=True)
print(f"[r32d-A1b] As.numel={As.numel()} Bs.numel={Bs.numel()}", flush=True)

C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()

print(f"[r32d-A1b] calling gemm_rcr_pq_v2 (rect kernel via R32D dispatcher)...", flush=True)
t0 = time.time()
try:
    tk_mxfp8_layouts.gemm_rcr_pq_v2(A, B, As, Bs, C)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[r32d-A1b] PASS — kernel completed in {elapsed:.3f}s, no GPU fault.", flush=True)
    # Print small slice of C to confirm it actually ran (not zeroed)
    print(f"[r32d-A1b] C[0, :8] = {C[0, :8].tolist()}", flush=True)
    print(f"[r32d-A1b] C abs sum = {C.abs().sum().item():.4f}", flush=True)
    print(f"[r32d-A1b] STAGE_A1b_RESULT: NO_FAULT", flush=True)
    sys.exit(0)
except Exception as e:
    print(f"[r32d-A1b] FAIL — exception: {e}", flush=True)
    print(f"[r32d-A1b] STAGE_A1b_RESULT: FAULT", flush=True)
    sys.exit(1)
