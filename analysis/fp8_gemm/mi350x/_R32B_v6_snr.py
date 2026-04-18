#!/usr/bin/env python3
"""V6 split-K SNR validation for L6 (4096x32768x128256).

Compares (split2_s0 + split2_s1 + epilogue) output vs the K_SPLIT=1 baseline
output on identical inputs. The split sums are mathematically identical to
the single-launch sum.

NOTE: At L6, with random ±N scales, bf16 accumulator output saturates to NaN
on a substantial fraction of positions (consistent with production behavior).
We compare only on finite positions and report:
  (a) the fraction of positions where both paths agree on NaN-ness,
  (b) SNR on the intersection of finite positions.

For DLA1-like patterns with smaller scales, bf16 stays in range.
"""
import gc, json, math, sys, torch
torch.manual_seed(0)
sys.path.insert(0, '/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_v6')

import tk_mxfp4_v6_split1 as MOD_BASE
import tk_mxfp4_v6_split2_s0 as MOD_S0
import tk_mxfp4_v6_split2_s1 as MOD_S1
EPI = MOD_S0.epilogue_add_bf16

M, N, K = 4096, 32768, 128256
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

    # ── Baseline (K_SPLIT=1) ──
    C_base = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    MOD_BASE.gemm_rcr(A, B, A_sc, B_sc, C_base)
    torch.cuda.synchronize()

    # ── Split (K_SPLIT=2, S=0 + S=1 + epilogue) ──
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
    both_fin = fin_b & fin_s
    nan_agree = (fin_b == fin_s).float().mean().item()
    base_fin_frac = fin_b.float().mean().item()
    split_fin_frac = fin_s.float().mean().item()

    # Stats on finite intersection
    if both_fin.sum() > 0:
        diff = (Cs - Cb)[both_fin]
        sig = Cb[both_fin]
        sig_p = (sig ** 2).mean().item()
        noise_p = (diff ** 2).mean().item()
        snr_db = 10.0 * math.log10(sig_p / noise_p) if noise_p > 0 else float('inf')
        max_abs = diff.abs().max().item()
        mean_abs_sig = sig.abs().mean().item()
        mean_abs_split = Cs[both_fin].abs().mean().item()
        c0_mean = C0.float()[both_fin].abs().mean().item()
        c1_mean = C1.float()[both_fin].abs().mean().item()
    else:
        snr_db = float('nan')
        max_abs = float('nan')
        mean_abs_sig = float('nan')
        mean_abs_split = float('nan')
        c0_mean = float('nan')
        c1_mean = float('nan')

    print(f"[{label}]")
    print(f"  base_finite={base_fin_frac:.4f}  split_finite={split_fin_frac:.4f}  "
          f"nan_pattern_agree={nan_agree:.4f}")
    print(f"  SNR(split vs base on finite intersection) = {snr_db:.2f} dB")
    print(f"  max|diff| = {max_abs:.4g}  |C_base|_mean = {mean_abs_sig:.4g}  "
          f"|C_split|_mean = {mean_abs_split:.4g}")
    print(f"  |C0|_mean = {c0_mean:.4g}  |C1|_mean = {c1_mean:.4g}")

    return {"label": label, "snr_db": round(snr_db, 2) if math.isfinite(snr_db) else None,
            "max_abs": max_abs,
            "base_finite_frac": base_fin_frac, "split_finite_frac": split_fin_frac,
            "nan_pattern_agree": nan_agree,
            "c0_mean": c0_mean, "c1_mean": c1_mean}


# ── Bf16-safe inputs for K=128256 ──
# At full random fp4 (0..15) + scale=1, bf16 saturates badly. We need very
# small input magnitudes. Use FP4 nibbles in {0, 1, 2} (values {0, 0.5, 1})
# and small scales. Mathematically the split-K identity is the same regardless
# of input magnitude.
#
# nibble & 0x33 → keep only bits 0,1 of each 4-bit field → values in 0..3
#   (FP4 codes 0,1,2,3 = +0, +0.5, +1, +1.5)
# nibble & 0x11 → keep only bit 0 of each 4-bit field → values in {0, 1} (codes 0,1 = 0, 0.5)
A_small = gen_fp4(M, K) & 0x33
B_small = gen_fp4(N, K) & 0x33
A_tiny = gen_fp4(M, K) & 0x11
B_tiny = gen_fp4(N, K) & 0x11

# Test 1: zero scales (=2^0) on tiny inputs (0 or 0.5) — max sum = 128256/4 ≈ 32K
sc_zero_a = torch.zeros((M, k_blocks), dtype=torch.int8, device='cuda')
sc_zero_b = torch.zeros((N, k_blocks), dtype=torch.int8, device='cuda')
r1 = run_test("tiny_AB_zero_scales", A_tiny, B_tiny, sc_zero_a, sc_zero_b)

# Test 2: small inputs + small +/- scales
sc_small_a = torch.randint(-1, 2, (M, k_blocks), dtype=torch.int8, device='cuda')
sc_small_b = torch.randint(-1, 2, (N, k_blocks), dtype=torch.int8, device='cuda')
r2 = run_test("small_AB_pm1_scales", A_small, B_small, sc_small_a, sc_small_b)

# Test 3: DLA1-like — slightly larger but still bf16-safe
A_d = gen_fp4(M, K) & 0x33
B_d = gen_fp4(N, K) & 0x33
sc_a3 = torch.full((M, k_blocks), -2, dtype=torch.int8, device='cuda')  # 2^-2 = 0.25
sc_b3 = torch.full((N, k_blocks), -2, dtype=torch.int8, device='cuda')
r3 = run_test("DLA1_small_AB_neg2_scales", A_d, B_d, sc_a3, sc_b3)

print("BENCH_JSON_START")
print(json.dumps([r1, r2, r3]))
print("BENCH_JSON_END")
