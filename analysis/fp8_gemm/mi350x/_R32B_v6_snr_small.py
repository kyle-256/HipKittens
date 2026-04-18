#!/usr/bin/env python3
"""Small-shape SNR validation for V6 split-K.

At L6 (K=128256) bf16 saturates regardless of split, so we cannot validate
correctness there. Instead validate the split-K MATH at a shape where bf16
holds: K=4096 lets the random-input sum stay finite.

Validates:
  baseline (K_SPLIT=1)  ==  split2_s0 + split2_s1 + epilogue

If this matches at K=4096, the split-K math is correct; the L6 NaN/Inf
pattern is purely bf16-saturation, present in production too.
"""
import gc, json, math, sys, torch
torch.manual_seed(0)
sys.path.insert(0, '/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_v6_small')

import tk_mxfp4_v6_small_split1 as MOD_BASE
import tk_mxfp4_v6_small_split2_s0 as MOD_S0
import tk_mxfp4_v6_small_split2_s1 as MOD_S1
EPI = MOD_S0.epilogue_add_bf16

# Small shape — bf16 safe with random scales
M, N, K = 4096, 4096, 4096
k_blocks = K // 32

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
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


def run_test(label, A, B, sc_a, sc_b):
    A_sc = preshuffle_mfma16_merged(sc_a)
    B_sc = preshuffle_mfma16_merged(sc_b)

    C_base = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    MOD_BASE.gemm_rcr(A, B, A_sc, B_sc, C_base)
    torch.cuda.synchronize()

    C0 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    C1 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    C_split = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    MOD_S0.gemm_rcr(A, B, A_sc, B_sc, C0)
    MOD_S1.gemm_rcr(A, B, A_sc, B_sc, C1)
    EPI(C0, C1, C_split)
    torch.cuda.synchronize()

    Cb = C_base.float()
    Cs = C_split.float()
    fin_b = torch.isfinite(Cb)
    fin_s = torch.isfinite(Cs)
    base_fin_frac = fin_b.float().mean().item()
    split_fin_frac = fin_s.float().mean().item()

    both_fin = fin_b & fin_s
    diff = (Cs - Cb)[both_fin]
    sig = Cb[both_fin]
    sig_p = (sig ** 2).mean().item()
    noise_p = (diff ** 2).mean().item()
    snr_db = 10.0 * math.log10(sig_p / noise_p) if noise_p > 0 else float('inf')
    max_abs = diff.abs().max().item()
    mean_abs_sig = sig.abs().mean().item()

    # Also: compute the per-split halves separately to verify they add up
    c0_mean = C0.float()[both_fin].abs().mean().item()
    c1_mean = C1.float()[both_fin].abs().mean().item()

    print(f"[{label}] M={M} N={N} K={K}")
    print(f"  base_finite={base_fin_frac:.4f}  split_finite={split_fin_frac:.4f}")
    print(f"  SNR(split vs base) = {snr_db:.2f} dB")
    print(f"  max|diff| = {max_abs:.4g}  |C_base|_mean = {mean_abs_sig:.4g}")
    print(f"  |C0|_mean = {c0_mean:.4g}  |C1|_mean = {c1_mean:.4g}  (should sum ≈ |C_base|_mean for non-cancelling case)")

    return {"label": label, "M": M, "N": N, "K": K,
            "snr_db": round(snr_db, 2) if math.isfinite(snr_db) else None,
            "max_abs": max_abs,
            "base_finite_frac": base_fin_frac, "split_finite_frac": split_fin_frac,
            "c0_mean": c0_mean, "c1_mean": c1_mean}


# ── Test 1: random fp4 with random scales ──
A = gen_fp4(M, K)
B = gen_fp4(N, K)
sc_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device='cuda')
sc_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device='cuda')
r1 = run_test("random_fp4_pm2_scales", A, B, sc_a, sc_b)

# ── Test 2: zero scales (=2^0), random fp4 ──
sc0a = torch.zeros((M, k_blocks), dtype=torch.int8, device='cuda')
sc0b = torch.zeros((N, k_blocks), dtype=torch.int8, device='cuda')
r2 = run_test("random_fp4_zero_scales", A, B, sc0a, sc0b)

# ── Test 3: random fp4 with negative scales (smaller magnitudes) ──
sc_neg_a = torch.randint(-4, -1, (M, k_blocks), dtype=torch.int8, device='cuda')
sc_neg_b = torch.randint(-4, -1, (N, k_blocks), dtype=torch.int8, device='cuda')
r3 = run_test("random_fp4_neg_scales", A, B, sc_neg_a, sc_neg_b)

print("BENCH_JSON_START")
print(json.dumps([r1, r2, r3]))
print("BENCH_JSON_END")
