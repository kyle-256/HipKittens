"""R26 Dev A — 5x A/B for H3: V2 vs V1 CRR-PQ at 4096x1024x8192.

Single process. Preheat + interleaved runs to neutralize drift.
"""
import os, sys, time, statistics, math

import torch
torch.manual_seed(0)

# ====== preheat ======
print("[preheat] starting sustained heavy load...", file=sys.stderr, flush=True)
pa = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
pb = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
t0 = time.time()
while time.time() - t0 < 6.0:
    pc = pa @ pb
torch.cuda.synchronize()
print(f"[preheat] done", file=sys.stderr, flush=True)

import tk_mxfp8_layouts as tk

M, N, K = 4096, 1024, 8192
flops_ref = 2 * M * N * K
WARMUP = 100
ITERS = 200

# ===== input gen (matches test_mxfp8_python.py) =====
def gen_fp8(rows, cols):
    x = (torch.randn(rows, cols, dtype=torch.float32, device="cuda") * 0.05)
    return x.to(torch.float8_e4m3fn)

def gen_scale_raw(rows, kblocks):
    s = torch.randint(low=-2, high=3, size=(rows, kblocks), dtype=torch.int8, device="cuda")
    return (s.to(torch.int16) + 127).to(torch.uint8)

# CRR: A is K x M, B is K x N
A = gen_fp8(K, M)
B = gen_fp8(K, N)
kblocks = K // 32
A_scale_exp = gen_scale_raw(M, kblocks)  # already raw (uint8)
B_scale_exp = gen_scale_raw(N, kblocks)

# ===== preshuffle helpers (copy from test_mxfp8_python.py) =====
import importlib.util, sys as _sys
spec = importlib.util.spec_from_file_location("test_mxfp8_python", os.path.join(os.path.dirname(__file__), "test_mxfp8_python.py"))
# Avoid actually executing the test; just import functions by re-reading source.
src = open(os.path.join(os.path.dirname(__file__), "test_mxfp8_python.py")).read()
# Extract two preshuffle helpers we need
import types
mod = types.ModuleType("tmpmod")
mod.__dict__["torch"] = torch
mod.__dict__["math"] = math
mod.__dict__["os"] = os
# Find function defs
import re
for fname in ("encode_scale_matrix_raw", "preshuffle_scale_matrix_mfma16", "preshuffle_scale_matrix_mfma16_v2_rcr_a", "preshuffle_scale_matrix_mfma16_v2_rcr_b"):
    m = re.search(rf"\ndef {fname}\(.*?\n(?=\n\n|\ndef |\Z)", src, re.S)
    if m:
        exec(m.group(0), mod.__dict__)
preshuffle_v1 = mod.preshuffle_scale_matrix_mfma16
preshuffle_v2_a = mod.preshuffle_scale_matrix_mfma16_v2_rcr_a
preshuffle_v2_b = mod.preshuffle_scale_matrix_mfma16_v2_rcr_b

# Use raw-encoded scales (as test does for PQ path) — convert int8 exponent to raw uint8 first
# test_mxfp8_python uses: scale_exp (int8), then preshuffle, then encode raw. Let's mirror exactly.
def gen_scale_int8(rows, kblocks):
    return torch.randint(low=-2, high=3, size=(rows, kblocks), dtype=torch.int8, device="cuda")

# Re-gen with int8 exponents to match preshuffle expectations
A_scale_exp = gen_scale_int8(M, kblocks)
B_scale_exp = gen_scale_int8(N, kblocks)

# Preshuffle V1 + V2
A_scale_v1 = preshuffle_v1(A_scale_exp)
B_scale_v1 = preshuffle_v1(B_scale_exp)
A_scale_v2 = preshuffle_v2_a(A_scale_exp)
B_scale_v2 = preshuffle_v2_b(B_scale_exp)

C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def bench(fn):
    # warmup
    for _ in range(WARMUP):
        C.zero_()
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        C.zero_()
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times

run_v1 = lambda: tk.gemm_crr_pq(A, B, A_scale_v1, B_scale_v1, C)
run_v2 = lambda: tk.gemm_crr_pq_v2(A, B, A_scale_v2, B_scale_v2, C)

# Quick sanity
print("[smoke] V1...", file=sys.stderr, flush=True); run_v1(); torch.cuda.synchronize()
print("[smoke] V2...", file=sys.stderr, flush=True); run_v2(); torch.cuda.synchronize()

# Interleaved 5x A/B
v1_tflops = []
v2_tflops = []
for trial in range(5):
    if trial % 2 == 0:
        order = (("V2", run_v2, v2_tflops), ("V1", run_v1, v1_tflops))
    else:
        order = (("V1", run_v1, v1_tflops), ("V2", run_v2, v2_tflops))
    for name, fn, lst in order:
        ts = bench(fn)
        avg_ms = sum(ts) / len(ts)
        tf = flops_ref / (avg_ms * 1e9)
        lst.append(tf)
        print(f"[trial {trial} {name}] {avg_ms*1000:.2f} us, {tf:.2f} TFLOPS", flush=True)

def stats(lst, name):
    med = statistics.median(lst)
    mean = statistics.mean(lst)
    sd = statistics.pstdev(lst)
    print(f"  {name}: median={med:.2f}  mean={mean:.2f}  std={sd:.2f}  raw={[f'{x:.1f}' for x in lst]}")
    return med, mean, sd

print("\n=== H3 RESULTS ===")
m_v1, _, sd_v1 = stats(v1_tflops, "V1")
m_v2, _, sd_v2 = stats(v2_tflops, "V2")
delta = m_v1 - m_v2
ratio = (m_v1 / m_v2 - 1.0) * 100
# Welch t
n_v1 = len(v1_tflops); n_v2 = len(v2_tflops)
mean_v1 = statistics.mean(v1_tflops); mean_v2 = statistics.mean(v2_tflops)
var_v1 = statistics.variance(v1_tflops) if n_v1 > 1 else 0.0
var_v2 = statistics.variance(v2_tflops) if n_v2 > 1 else 0.0
se = math.sqrt(var_v1/n_v1 + var_v2/n_v2)
t = (mean_v1 - mean_v2) / se if se > 0 else float("inf")
print(f"  V1 - V2 = {delta:+.2f} TFLOPS ({ratio:+.2f}%)  Welch t={t:.2f}")
