#!/usr/bin/env python3
"""Compare V6 K_SPLIT=1 vs V6 K_SPLIT=2+epilogue against TORCH REFERENCE.

Use small shape so torch reference fits, and use very small inputs so bf16
saturation doesn't occur.
"""
import gc, json, math, sys, torch
torch.manual_seed(0)
sys.path.insert(0, '/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_v6_small')

import tk_mxfp4_v6_small_split1 as MOD_BASE
import tk_mxfp4_v6_small_split2_s0 as MOD_S0
import tk_mxfp4_v6_small_split2_s1 as MOD_S1
EPI = MOD_S0.epilogue_add_bf16

# Small shape
M, N, K = 4096, 4096, 4096
k_blocks = K // 32

# FP4 e2m1 LUT (16 codes -> float32 values).
# bit 3 = sign, bits 2:1 = exponent (0..3), bit 0 = mantissa
# values: 0=+0, 1=+0.5, 2=+1, 3=+1.5, 4=+2, 5=+3, 6=+4, 7=+6
# 8=-0, 9=-0.5, ... 15=-6
FP4_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def gen_fp4(rows, K, mask=0xFF):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    packed = ((hi << 4) | lo).to(torch.uint8)
    return packed & torch.tensor(mask, dtype=torch.uint8, device='cuda')


def fp4_unpack(packed, K):
    """Unpack FP4 nibble-pairs to float32 [rows, K]."""
    rows, cols = packed.shape
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)
    lut = FP4_LUT.to(packed.device)
    lo_f = lut[lo]
    hi_f = lut[hi]
    out = torch.empty(rows, K, dtype=torch.float32, device=packed.device)
    out[:, 0::2] = lo_f
    out[:, 1::2] = hi_f
    return out


def torch_ref(A_packed, B_packed, sc_a, sc_b, M, N, K):
    """Reference: dequant A and B, scale per 32-block, then bf16 matmul."""
    # Unpack
    A_f = fp4_unpack(A_packed, K)  # [M, K]
    B_f = fp4_unpack(B_packed, K)  # [N, K]
    # Apply per-32 scales: scale = 2^scale_exp
    sc_a_f = (2.0 ** sc_a.to(torch.float32))  # [M, K/32]
    sc_b_f = (2.0 ** sc_b.to(torch.float32))  # [N, K/32]
    # Broadcast to [M, K] by repeating each 32 times
    A_scaled = A_f * sc_a_f.repeat_interleave(32, dim=1)
    B_scaled = B_f * sc_b_f.repeat_interleave(32, dim=1)
    # Cast to bf16 then matmul (mimics kernel pipeline)
    Ab = A_scaled.bfloat16()
    Bb = B_scaled.bfloat16()
    # C = A @ B^T
    Cf = (Ab.float() @ Bb.float().T).bfloat16()
    return Cf


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


def snr_db(ref, test, label):
    fin = torch.isfinite(ref) & torch.isfinite(test)
    n_fin = int(fin.sum().item())
    fin_frac = fin.float().mean().item()
    if n_fin == 0:
        print(f"  {label:>22s}: NO FINITE OVERLAP  fin_frac={fin_frac:.4f}")
        return None
    diff = (test.float() - ref.float())[fin]
    sig = ref.float()[fin]
    sig_p = (sig ** 2).mean().item()
    noise_p = (diff ** 2).mean().item()
    max_d = diff.abs().max().item()
    mean_sig = sig.abs().mean().item()
    if noise_p <= 0:
        s = float('inf')
    elif sig_p <= 0:
        s = float('-inf')
    else:
        # Use log subtraction to avoid underflow
        try:
            s = 10.0 * (math.log10(sig_p) - math.log10(noise_p))
        except ValueError:
            s = float('nan')
    print(f"  {label:>22s}: SNR={s:7.2f} dB  finite={fin_frac:.4f}  max|diff|={max_d:.4g}  |ref|_mean={mean_sig:.4g}")
    return s


def run(label, A, B, sc_a, sc_b):
    print(f"\n[{label}] M={M} N={N} K={K}")
    A_sc = preshuffle_mfma16_merged(sc_a)
    B_sc = preshuffle_mfma16_merged(sc_b)

    C_base = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    MOD_BASE.gemm_rcr(A, B, A_sc, B_sc, C_base)

    C0 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    C1 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    C_split = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    MOD_S0.gemm_rcr(A, B, A_sc, B_sc, C0)
    MOD_S1.gemm_rcr(A, B, A_sc, B_sc, C1)
    EPI(C0, C1, C_split)
    torch.cuda.synchronize()

    Cref = torch_ref(A, B, sc_a, sc_b, M, N, K)

    s_base = snr_db(Cref, C_base, "K_SPLIT=1 vs torch")
    s_split = snr_db(Cref, C_split, "split2+epi vs torch")
    s_match = snr_db(C_base, C_split, "split2+epi vs split1")
    return {"label": label, "snr_base_vs_torch": s_base,
            "snr_split_vs_torch": s_split, "snr_split_vs_base": s_match}


# Test 1: tiny inputs (FP4 codes 0..3, values 0,0.5,1,1.5), zero scales
A1 = gen_fp4(M, K, 0x33)
B1 = gen_fp4(N, K, 0x33)
sc0a = torch.zeros((M, k_blocks), dtype=torch.int8, device='cuda')
sc0b = torch.zeros((N, k_blocks), dtype=torch.int8, device='cuda')
r1 = run("tiny_AB_zero_sc", A1, B1, sc0a, sc0b)

# Test 2: small inputs, small ± scales
sc_pm_a = torch.randint(-1, 2, (M, k_blocks), dtype=torch.int8, device='cuda')
sc_pm_b = torch.randint(-1, 2, (N, k_blocks), dtype=torch.int8, device='cuda')
r2 = run("tiny_AB_pm1_sc", A1, B1, sc_pm_a, sc_pm_b)

# Test 3: full-range FP4, all-negative scales (small magnitudes)
A3 = gen_fp4(M, K)  # full 0..15
B3 = gen_fp4(N, K)
sc_neg_a = torch.full((M, k_blocks), -3, dtype=torch.int8, device='cuda')  # 2^-3 = 0.125
sc_neg_b = torch.full((N, k_blocks), -3, dtype=torch.int8, device='cuda')
r3 = run("full_AB_neg3_sc", A3, B3, sc_neg_a, sc_neg_b)

print("\nBENCH_JSON_START")
print(json.dumps([r1, r2, r3]))
print("BENCH_JSON_END")
