#!/usr/bin/env python3
"""Debug the 128-tile kernel correctness issue."""
import math, sys, os, torch
torch.manual_seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import tk_mxfp4_128tile
import tk_mxfp4_ref

N, K = 32768, 4096
M = 256
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

# Test 1: Unit scales (all exponent = 0)
print("=== Test 1: Unit scales ===")
A = gen_fp4(M, K)
B = gen_fp4(N, K)
sc_exp_a = torch.zeros(M, k_blocks, dtype=torch.int8, device="cuda")
sc_exp_b = torch.zeros(N, k_blocks, dtype=torch.int8, device="cuda")
A_sc = preshuffle_mfma16_merged(sc_exp_a)
B_sc = preshuffle_mfma16_merged(sc_exp_b)

C_ref = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
C_128 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

tk_mxfp4_ref.gemm_rcr(A, B, A_sc, B_sc, C_ref)
tk_mxfp4_128tile.gemm_rcr(A, B, A_sc, B_sc, C_128)
torch.cuda.synchronize()

diff = (C_ref.float() - C_128.float()).abs()
max_diff = diff.max().item()
print(f"  Max diff: {max_diff:.6f}")
print(f"  Exact match: {(C_ref == C_128).all().item()}")

if max_diff > 0.5:
    idx = diff.argmax()
    row = idx.item() // N
    col = idx.item() % N
    print(f"  Max diff at [{row}, {col}]: ref={C_ref[row,col].item():.4f}, 128={C_128[row,col].item():.4f}")

    # Check which quadrant has errors
    for mh in range(2):
        for nh in range(2):
            r0, r1 = mh * 128, (mh+1) * 128
            c0, c1 = nh * (N//2), (nh+1) * (N//2)
            if r1 > M: continue
            block_diff = diff[r0:r1, c0:c1].max().item()
            block_match = (C_ref[r0:r1, c0:c1] == C_128[r0:r1, c0:c1]).all().item()
            print(f"  Block mh={mh},nh={nh} [{r0}:{r1},{c0}:{c1}]: max_diff={block_diff:.4f}, match={block_match}")

    # Check finer blocks (32x32 warp blocks)
    print("  Per-warp blocks (32x32):")
    for i in range(M // 32):
        for j in range(min(4, N // 32)):  # First 4 B blocks
            r0, r1 = i * 32, (i+1) * 32
            c0, c1 = j * 32, (j+1) * 32
            block_diff = diff[r0:r1, c0:c1].max().item()
            if block_diff > 0.5:
                print(f"    [{i}][{j}] rows {r0}-{r1} cols {c0}-{c1}: max_diff={block_diff:.4f}")

# Test 2: With varying scales
print("\n=== Test 2: Non-trivial scales ===")
sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
A_sc = preshuffle_mfma16_merged(sc_exp_a)
B_sc = preshuffle_mfma16_merged(sc_exp_b)

C_ref.zero_()
C_128.zero_()

tk_mxfp4_ref.gemm_rcr(A, B, A_sc, B_sc, C_ref)
tk_mxfp4_128tile.gemm_rcr(A, B, A_sc, B_sc, C_128)
torch.cuda.synchronize()

diff = (C_ref.float() - C_128.float()).abs()
max_diff = diff.max().item()
print(f"  Max diff: {max_diff:.6f}")
print(f"  Exact match: {(C_ref == C_128).all().item()}")

if max_diff > 0.5:
    # Check quadrants
    for mh in range(2):
        for nh in range(2):
            r0, r1 = mh * 128, min((mh+1) * 128, M)
            # For 128-tile: N split at HB=64 within each tile
            # But tiles are 128 wide, so Bl covers first 64 cols, Br covers next 64
            # Actually the column split is at N/2 (HB of the B dimension)
            # With BLK=128 tile, each tile covers 128 columns
            # Within each tile: Bl=cols[0:64], Br=cols[64:128]
            c0, c1 = nh * (N//2), (nh+1) * (N//2)
            if r1 > M: continue
            block_diff = diff[r0:r1, c0:c1].max().item()
            block_mean = diff[r0:r1, c0:c1].mean().item()
            print(f"  Quadrant mh={mh},nh={nh}: max_diff={block_diff:.4f}, mean_diff={block_mean:.6f}")

# Test 3: Check with deterministic small values
print("\n=== Test 3: All-ones data, unit scales ===")
A = torch.ones(M, K // 2, dtype=torch.uint8, device="cuda") * 0x11  # FP4 val 1 in both nibbles
B = torch.ones(N, K // 2, dtype=torch.uint8, device="cuda") * 0x11
sc_exp_a = torch.zeros(M, k_blocks, dtype=torch.int8, device="cuda")
sc_exp_b = torch.zeros(N, k_blocks, dtype=torch.int8, device="cuda")
A_sc = preshuffle_mfma16_merged(sc_exp_a)
B_sc = preshuffle_mfma16_merged(sc_exp_b)

C_ref.zero_()
C_128.zero_()

tk_mxfp4_ref.gemm_rcr(A, B, A_sc, B_sc, C_ref)
tk_mxfp4_128tile.gemm_rcr(A, B, A_sc, B_sc, C_128)
torch.cuda.synchronize()

diff = (C_ref.float() - C_128.float()).abs()
max_diff = diff.max().item()
print(f"  Max diff: {max_diff:.6f}")
print(f"  Ref[0,0] = {C_ref[0,0].item():.4f}, 128[0,0] = {C_128[0,0].item():.4f}")
print(f"  Ref[128,0] = {C_ref[128,0].item():.4f}, 128[128,0] = {C_128[128,0].item():.4f}")
print(f"  Exact match: {(C_ref == C_128).all().item()}")

print("\nDone!")
