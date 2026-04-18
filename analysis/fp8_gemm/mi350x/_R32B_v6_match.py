#!/usr/bin/env python3
"""Verify V6 split-K MATH:
  C0_bf16 = bf16_round(sum_k_in_split0 in float32)
  C1_bf16 = bf16_round(sum_k_in_split1 in float32)
  C_split = bf16_round(C0 + C1)
  C_base  = bf16_round(sum_all_k in float32)

These match closely (within 2 ULP) when no overflow occurs. We verify the
MATCH RATIO of split2+epi vs split1 baseline, restricting attention to
positions where BOTH outputs are bf16-finite and below a sanity threshold.

Run at the v6_small build shape (M=N=K=4096) with very small inputs.
"""
import sys, math, json, torch
torch.manual_seed(42)
sys.path.insert(0, '/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_v6_small')

import tk_mxfp4_v6_small_split1   as MOD_BASE
import tk_mxfp4_v6_small_split2_s0 as MOD_S0
import tk_mxfp4_v6_small_split2_s1 as MOD_S1
EPI = MOD_S0.epilogue_add_bf16

M, N, K = 4096, 4096, 4096
k_blocks = K // 32

def gen_fp4(rows, K, mask=0xFF):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    p = ((hi << 4) | lo).to(torch.uint8)
    return p & torch.tensor(mask, dtype=torch.uint8, device='cuda')

def preshuffle(scale_exp):
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


def stats(C_base, C_split, label):
    Cb = C_base.float()
    Cs = C_split.float()
    fin = torch.isfinite(Cb) & torch.isfinite(Cs)
    nfin = int(fin.sum().item())
    if nfin == 0:
        print(f"[{label}] NO finite overlap")
        return None
    # Limit to positions where signal magnitude is reasonable (< 1e30)
    safe = fin & (Cb.abs() < 1e30) & (Cs.abs() < 1e30)
    nsafe = int(safe.sum().item())
    if nsafe == 0:
        print(f"[{label}] NO safe overlap")
        return None
    diff = (Cs - Cb)[safe]
    sig  = Cb[safe]
    abs_diff = diff.abs()
    rel_diff = (abs_diff / (sig.abs() + 1e-6))
    sig_p = (sig ** 2).mean().item()
    noise_p = (diff ** 2).mean().item()
    if noise_p > 0 and sig_p > 0:
        snr = 10.0 * (math.log10(sig_p) - math.log10(noise_p))
    elif noise_p == 0:
        snr = float('inf')
    else:
        snr = float('-inf')
    # Element-level "match" criterion: |diff| <= max(2 ULPs of bf16, 1e-3 * |sig|)
    bf16_eps = sig.abs() * (1.0 / 256.0)  # ~2 ULPs of bf16 (1/128 mantissa)
    tol = torch.maximum(bf16_eps, torch.full_like(bf16_eps, 1e-3))
    matches = (abs_diff <= tol).float().mean().item()
    print(f"[{label}] M={M} N={N} K={K}")
    print(f"  finite_overlap={fin.float().mean().item():.4f}  safe_overlap={safe.float().mean().item():.4f}")
    print(f"  SNR(split2+epi vs split1) = {snr:7.2f} dB")
    print(f"  max|diff|={abs_diff.max().item():.4g}  mean|diff|={abs_diff.mean().item():.4g}")
    print(f"  |sig|_mean={sig.abs().mean().item():.4g}  match_ratio={matches:.6f}")
    return {"label": label, "snr_db": snr, "match_ratio": matches,
            "safe_overlap": safe.float().mean().item(),
            "max_abs_diff": abs_diff.max().item()}


def run(label, A, B, sc_a, sc_b):
    A_sc = preshuffle(sc_a)
    B_sc = preshuffle(sc_b)
    C_base = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    MOD_BASE.gemm_rcr(A, B, A_sc, B_sc, C_base)
    C0 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    C1 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    C_split = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    MOD_S0.gemm_rcr(A, B, A_sc, B_sc, C0)
    MOD_S1.gemm_rcr(A, B, A_sc, B_sc, C1)
    EPI(C0, C1, C_split)
    torch.cuda.synchronize()
    return stats(C_base, C_split, label)


# Test 1: tiny non-negative — FP4 codes 0..3 (values 0,0.5,1,1.5), zero scales
A1 = gen_fp4(M, K, 0x33)
B1 = gen_fp4(N, K, 0x33)
sc0a = torch.zeros((M, k_blocks), dtype=torch.int8, device='cuda')
sc0b = torch.zeros((N, k_blocks), dtype=torch.int8, device='cuda')
r1 = run("tiny_nn_zero_sc", A1, B1, sc0a, sc0b)

# Test 2: very small all-positive — FP4 codes 0,1 only, zero scales
A2 = gen_fp4(M, K, 0x11)
B2 = gen_fp4(N, K, 0x11)
r2 = run("min_nn_zero_sc", A2, B2, sc0a, sc0b)

# Test 3: mid-range FP4 0..7 (no negatives), small negative scales
A3 = gen_fp4(M, K, 0x77)
B3 = gen_fp4(N, K, 0x77)
sc_neg = torch.full((M, k_blocks), -3, dtype=torch.int8, device='cuda')  # 2^-3=0.125
r3 = run("pos_AB_neg3_sc", A3, B3, sc_neg, sc_neg)

print("\nBENCH_JSON_START")
print(json.dumps([r1, r2, r3]))
print("BENCH_JSON_END")
