#!/usr/bin/env python3
"""Quick V6 split-K bench at L6 — no long warmup that triggered the GPU fault.
Uses 50 warmup, 50 iters with cuda events for relative comparison.
"""
import sys, math, importlib, statistics, json, torch
torch.manual_seed(0)
sys.path.insert(0, '/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_v6')
sys.path.insert(0, '/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_all42')

INC      = importlib.import_module('tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all')
MOD_BASE = importlib.import_module('tk_mxfp4_v6_split1')
MOD_S2_0 = importlib.import_module('tk_mxfp4_v6_split2_s0')
MOD_S2_1 = importlib.import_module('tk_mxfp4_v6_split2_s1')
MOD_S4_0 = importlib.import_module('tk_mxfp4_v6_split4_s0')
MOD_S4_1 = importlib.import_module('tk_mxfp4_v6_split4_s1')
MOD_S4_2 = importlib.import_module('tk_mxfp4_v6_split4_s2')
MOD_S4_3 = importlib.import_module('tk_mxfp4_v6_split4_s3')
EPI = MOD_S2_0.epilogue_add_bf16

M, N, K = 4096, 32768, 128256
WARMUP = 50
ITERS = 50

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

def bench(label, fn, k=10):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times_us = []
    for _ in range(ITERS):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record()
        torch.cuda.synchronize()
        times_us.append(e0.elapsed_time(e1) * 1000.0)
    s = sorted(times_us)
    tm = statistics.mean(s[k:-k])
    flops = 2.0 * M * N * K
    tflops = flops / (tm * 1e-6) / 1e12
    print(f'[{label:30s}] trim={tm:8.2f} us  TFLOPS={tflops:6.1f}  p10={s[5]:.2f}  p50={s[25]:.2f}  p90={s[45]:.2f}')
    return {"label": label, "us": tm, "tflops": tflops}

A = gen_fp4(M, K); B = gen_fp4(N, K)
sc_a = torch.randint(-2, 3, (M, K // 32), dtype=torch.int8, device='cuda')
sc_b = torch.randint(-2, 3, (N, K // 32), dtype=torch.int8, device='cuda')
A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)

C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
C0 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
C1 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
C2 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
C3 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
Cout = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

results = {}
results['INC'] = bench('INCUMBENT', lambda: INC.gemm_rcr(A, B, A_sc, B_sc, C))
results['V6_split1'] = bench('V6_split1 (sanity)', lambda: MOD_BASE.gemm_rcr(A, B, A_sc, B_sc, C))

def f_split2():
    MOD_S2_0.gemm_rcr(A, B, A_sc, B_sc, C0)
    MOD_S2_1.gemm_rcr(A, B, A_sc, B_sc, C1)
    EPI(C0, C1, Cout)
results['V6_split2_total'] = bench('V6_split2+epi (TOTAL)', f_split2)

def f_split4():
    MOD_S4_0.gemm_rcr(A, B, A_sc, B_sc, C0)
    MOD_S4_1.gemm_rcr(A, B, A_sc, B_sc, C1)
    MOD_S4_2.gemm_rcr(A, B, A_sc, B_sc, C2)
    MOD_S4_3.gemm_rcr(A, B, A_sc, B_sc, C3)
    EPI(C0, C1, Cout)
    EPI(Cout, C2, Cout)
    EPI(Cout, C3, Cout)
results['V6_split4_total'] = bench('V6_split4+epi (TOTAL)', f_split4)

# Per-component costs
results['v6_s2_0'] = bench('  V6_split2_s0', lambda: MOD_S2_0.gemm_rcr(A, B, A_sc, B_sc, C0))
results['v6_s2_1'] = bench('  V6_split2_s1', lambda: MOD_S2_1.gemm_rcr(A, B, A_sc, B_sc, C1))
results['v6_s4_0'] = bench('  V6_split4_s0', lambda: MOD_S4_0.gemm_rcr(A, B, A_sc, B_sc, C0))
results['v6_s4_1'] = bench('  V6_split4_s1', lambda: MOD_S4_1.gemm_rcr(A, B, A_sc, B_sc, C1))
results['v6_s4_2'] = bench('  V6_split4_s2', lambda: MOD_S4_2.gemm_rcr(A, B, A_sc, B_sc, C2))
results['v6_s4_3'] = bench('  V6_split4_s3', lambda: MOD_S4_3.gemm_rcr(A, B, A_sc, B_sc, C3))
results['epi'] = bench('  epilogue_add', lambda: EPI(C0, C1, Cout))

print('\nSummary vs incumbent:')
inc_us = results['INC']['us']
inc_tflops = results['INC']['tflops']
for k, r in results.items():
    if k == 'INC' or k.startswith(' '): continue
    if k.startswith('v6'): continue
    delta_pct = (r['tflops'] / inc_tflops - 1) * 100
    print(f"  {r['label']:30s} {r['us']:8.2f} us  {r['tflops']:6.1f} TFLOPS  ({delta_pct:+.2f}%)")

print('\nBENCH_JSON_START')
print(json.dumps(results))
print('BENCH_JSON_END')
