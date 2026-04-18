#!/usr/bin/env python3
"""V6 split-K BENCHMARK at L6: 2-launch + epilogue total time vs incumbent.

Per benchmark-rules.md: warmup=200, iters=500, trim=10%.

Compares:
  (a) Incumbent: ts_lgk2_v12_memc_btw_all (5354 TFLOPS = 92.6% of comp 5781)
  (b) V6 K_SPLIT=1 baseline (sanity — should match incumbent, modulo flag-set)
  (c) V6 split2+epi (the candidate)

Total TFLOPS counts ONLY the GEMM FLOPs (2*M*N*K). Epilogue cost is added
to the wall time but not the FLOPs (so split-K with overhead → lower TFLOPS).
"""
import gc, json, math, sys, importlib, statistics, time, torch
torch.manual_seed(0)

V6_DIR = '/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_v6'
INC_DIR = '/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_all42'
sys.path.insert(0, V6_DIR)
sys.path.insert(0, INC_DIR)

INC_NAME = "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all"
INC      = importlib.import_module(INC_NAME)
MOD_BASE = importlib.import_module("tk_mxfp4_v6_split1")
MOD_S0   = importlib.import_module("tk_mxfp4_v6_split2_s0")
MOD_S1   = importlib.import_module("tk_mxfp4_v6_split2_s1")
EPI      = MOD_S0.epilogue_add_bf16

M, N, K = 4096, 32768, 128256
k_blocks = K // 32
WARMUP = 200
ITERS = 500
TRIM_FRAC = 0.10

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

def trimmed_mean(times_us, frac=TRIM_FRAC):
    s = sorted(times_us)
    n = len(s)
    k = int(n * frac)
    return statistics.mean(s[k:n-k])

def bench_one(label, fn):
    """fn() runs ONE iteration. Returns trimmed-mean us per call and TFLOPS."""
    # Warmup
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times_us = []
    for _ in range(ITERS):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        times_us.append(e0.elapsed_time(e1) * 1000.0)  # ms -> us
    tm = trimmed_mean(times_us)
    flops = 2.0 * M * N * K
    tflops = flops / (tm * 1e-6) / 1e12
    print(f"[{label}] trimmed_mean={tm:.2f} us  TFLOPS={tflops:.1f}")
    return {"label": label, "us": tm, "tflops": tflops,
            "p10": sorted(times_us)[int(0.1*len(times_us))],
            "p50": sorted(times_us)[int(0.5*len(times_us))],
            "p90": sorted(times_us)[int(0.9*len(times_us))]}


# Inputs
A = gen_fp4(M, K)
B = gen_fp4(N, K)
sc_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device='cuda')
sc_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device='cuda')
A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
C   = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
C0  = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
C1  = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
Cout= torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

# (a) Incumbent
def f_inc():
    INC.gemm_rcr(A, B, A_sc, B_sc, C)
r_inc = bench_one("INCUMBENT", f_inc)

# (b) V6 K_SPLIT=1 (sanity)
def f_v6_split1():
    MOD_BASE.gemm_rcr(A, B, A_sc, B_sc, C)
r_v6_b = bench_one("V6_split1", f_v6_split1)

# (c) V6 split2+epi (TOTAL: 2 GEMMs + epilogue)
def f_v6_split2_total():
    MOD_S0.gemm_rcr(A, B, A_sc, B_sc, C0)
    MOD_S1.gemm_rcr(A, B, A_sc, B_sc, C1)
    EPI(C0, C1, Cout)
r_v6_t = bench_one("V6_split2+epi (TOTAL)", f_v6_split2_total)

# Component costs
def f_v6_s0():
    MOD_S0.gemm_rcr(A, B, A_sc, B_sc, C0)
r_v6_s0 = bench_one("  V6_split2_s0 only", f_v6_s0)

def f_v6_s1():
    MOD_S1.gemm_rcr(A, B, A_sc, B_sc, C1)
r_v6_s1 = bench_one("  V6_split2_s1 only", f_v6_s1)

def f_epi():
    EPI(C0, C1, Cout)
r_epi = bench_one("  epilogue only", f_epi)

print("\nSummary:")
print(f"  Incumbent:           {r_inc['us']:7.2f} us  {r_inc['tflops']:6.1f} TFLOPS")
print(f"  V6_split1 (sanity):  {r_v6_b['us']:7.2f} us  {r_v6_b['tflops']:6.1f} TFLOPS")
print(f"  V6_split2+epi total: {r_v6_t['us']:7.2f} us  {r_v6_t['tflops']:6.1f} TFLOPS")
print(f"    s0: {r_v6_s0['us']:.2f} us  s1: {r_v6_s1['us']:.2f} us  epi: {r_epi['us']:.2f} us")
print(f"  V6 vs incumbent: {(r_inc['us'] / r_v6_t['us'] - 1) * 100:+.2f}% wall, {(r_v6_t['tflops'] / r_inc['tflops'] - 1) * 100:+.2f}% TFLOPS")

print("\nBENCH_JSON_START")
print(json.dumps({"incumbent": r_inc, "v6_split1": r_v6_b, "v6_split2_total": r_v6_t,
                  "v6_split2_s0": r_v6_s0, "v6_split2_s1": r_v6_s1, "epilogue": r_epi}))
print("BENCH_JSON_END")
