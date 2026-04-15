#!/usr/bin/env python3
"""Compare 128-tile kernel output against reference 256-tile kernel.

Both kernels compiled with K_DIM=4096, N_DIM=32768.
Test with M values that are multiples of LCM(128, 256) = 256.
"""
import math, sys, os, torch
torch.manual_seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import tk_mxfp4_128tile
import tk_mxfp4_ref

N, K = 32768, 4096
k_blocks = K // 32

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo

def preshuffle_mfma16_merged(scale_exp):
    rows, kb = scale_exp.shape
    pr = math.ceil(rows / 64) * 64
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    sh = sh.view(pr // 32, pk * 32)
    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4)
    sh = sh.permute(0, 2, 1, 3).contiguous()
    return sh.view(pr // 64, pk * 64)

# Test with M values divisible by 256 (works for both kernels)
for M in [256, 512, 1024, 4096]:
    print(f"\nTesting M={M}, N={N}, K={K}")

    A = gen_fp4(M, K)
    B = gen_fp4(N, K)
    sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
    sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
    A_sc = preshuffle_mfma16_merged(sc_exp_a)
    B_sc = preshuffle_mfma16_merged(sc_exp_b)

    C_ref = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    C_128 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    tk_mxfp4_ref.gemm_rcr(A, B, A_sc, B_sc, C_ref)
    tk_mxfp4_128tile.gemm_rcr(A, B, A_sc, B_sc, C_128)
    torch.cuda.synchronize()

    # Compare
    diff = (C_ref.float() - C_128.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Ref range: [{C_ref.min().item():.2f}, {C_ref.max().item():.2f}]")
    print(f"  128 range: [{C_128.min().item():.2f}, {C_128.max().item():.2f}]")

    # Check exact match (both produce bf16 output, should be identical)
    exact_match = (C_ref == C_128).all().item()
    print(f"  Exact match: {exact_match}")

    if max_diff > 0.5:
        print(f"  FAIL: max diff too large!")
        idx = diff.argmax()
        row = idx.item() // N
        col = idx.item() % N
        print(f"  Location: [{row}, {col}]")
        print(f"  Ref: {C_ref[row, col].item():.4f}, 128: {C_128[row, col].item():.4f}")
    elif not exact_match:
        print(f"  WARN: not exact match but close")
        mismatch_count = (C_ref != C_128).sum().item()
        print(f"  Mismatched elements: {mismatch_count} / {M*N}")
    else:
        print(f"  PASS (exact match)")

print("\nDone!")
