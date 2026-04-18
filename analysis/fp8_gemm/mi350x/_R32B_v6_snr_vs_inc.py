#!/usr/bin/env python3
"""V6 split-K SNR validation at L6 vs INCUMBENT (R32A pattern).

L6 = 4096×32768×128256. Compares (split2_s0 + split2_s1 + epilogue) output
to the L6 incumbent kernel's output, restricted to both-finite entries.

Acceptance gate: SNR ≥ 25 dB (per R32 mission spec, same as R32A).
"""
import gc, json, math, sys, importlib, torch
torch.manual_seed(0)

V6_DIR = '/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_v6'
INC_DIR = '/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_all42'
sys.path.insert(0, V6_DIR)
sys.path.insert(0, INC_DIR)

INC_NAME = "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all"
MOD_BASE = importlib.import_module("tk_mxfp4_v6_split1")
MOD_S0   = importlib.import_module("tk_mxfp4_v6_split2_s0")
MOD_S1   = importlib.import_module("tk_mxfp4_v6_split2_s1")
INC      = importlib.import_module(INC_NAME)
EPI = MOD_S0.epilogue_add_bf16

M, N, K = 4096, 32768, 128256
k_blocks = K // 32

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    return (hi << 4) | lo

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

def snr_db(ref, test, mag_cap=1e15):
    # Restrict to entries where BOTH outputs are finite AND below mag_cap
    # (so rf*rf doesn't overflow to inf in float32). On L6, ~38% of entries pass this.
    fin = torch.isfinite(ref) & torch.isfinite(test)
    safe = fin & (ref.abs().float() < mag_cap) & (test.abs().float() < mag_cap)
    n = int(safe.sum().item())
    if n == 0:
        return float('nan'), 0.0
    rf = ref.float()[safe]; tf = test.float()[safe]
    err = tf - rf
    sp = (rf * rf).mean().item()
    ep = (err * err).mean().item() + 1e-30
    if not (math.isfinite(sp) and math.isfinite(ep)):
        return float('nan'), safe.float().mean().item()
    if sp <= 0:
        return float('-inf'), safe.float().mean().item()
    s = 10.0 * math.log10(sp / ep)
    return s, safe.float().mean().item()

def run(label, A, B, sc_a, sc_b):
    A_sc = preshuffle(sc_a)
    B_sc = preshuffle(sc_b)

    # Reference: L6 incumbent
    C_inc = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    INC.gemm_rcr(A, B, A_sc, B_sc, C_inc)
    torch.cuda.synchronize()
    inc_fin = torch.isfinite(C_inc).float().mean().item()

    # V6 K_SPLIT=1 (sanity: identical math to incumbent except different flag set)
    C_base = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    MOD_BASE.gemm_rcr(A, B, A_sc, B_sc, C_base)
    torch.cuda.synchronize()

    # V6 split2 + epilogue
    C0 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    C1 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    C_split = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    MOD_S0.gemm_rcr(A, B, A_sc, B_sc, C0)
    MOD_S1.gemm_rcr(A, B, A_sc, B_sc, C1)
    EPI(C0, C1, C_split)
    torch.cuda.synchronize()

    s_base, f_base = snr_db(C_inc, C_base)
    s_split, f_split = snr_db(C_inc, C_split)

    print(f"[{label}]")
    print(f"  incumbent finite_frac={inc_fin:.4f}")
    print(f"  V6_split1 SNR vs incumbent = {s_base:7.2f} dB  both_finite={f_base:.4f}")
    print(f"  V6_split2+epi SNR vs incumbent = {s_split:7.2f} dB  both_finite={f_split:.4f}")

    return {"label": label, "inc_finite": inc_fin,
            "snr_v6split1_vs_inc": s_base if math.isfinite(s_base) else None,
            "snr_v6split2_vs_inc": s_split if math.isfinite(s_split) else None,
            "both_fin_split1": f_base, "both_fin_split2": f_split}


# Test 1: same input pattern as R32A (random fp4 + random ±2 scales)
A = gen_fp4(M, K)
B = gen_fp4(N, K)
sc_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device='cuda')
sc_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device='cuda')
r1 = run("R32A_pattern", A, B, sc_a, sc_b)

# Test 2: DLA1-like — small ± scales
sc_a2 = torch.randint(-3, 1, (M, k_blocks), dtype=torch.int8, device='cuda')
sc_b2 = torch.randint(-3, 1, (N, k_blocks), dtype=torch.int8, device='cuda')
r2 = run("DLA1_neg_scales", A, B, sc_a2, sc_b2)

print("\nBENCH_JSON_START")
print(json.dumps([r1, r2]))
print("BENCH_JSON_END")
