"""R45 Dev A — paired BABA bench for R44A M=2..16 decode fastpath.

Loads two .so files:
  - SO_DECODE: R44A decode .so (R44A predicate fires, hits gemv_m2_16_decode_kernel)
  - SO_BASELINE: V1-LEGACY .so (M_DIM=256 trick, falls through to legacy tail)

Both .so expose gemm_rcr_pq / gemm_rrr_pq / gemm_crr_pq. Paired BABA pattern,
30s+ preheat, sclk-post-preheat / sclk-post-bench reported for 3-gate retry.

Env vars:
  M, N, K           — shape
  SO_DECODE, MOD_DECODE
  SO_BASELINE, MOD_BASELINE
  LAYOUT            — rcr | rrr | crr (default rcr)
  PHYS_GPU          — physical GPU (for rocm-smi)
  N_PAIRS           — paired BABA reps (default 5)
  MXFP8_WARMUP      — warmup iters per bench (default 30)
  MXFP8_ITERS       — measured iters per bench (default 50)
  PREHEAT_S         — preheat seconds (default 30)
  WARMUP_PAIRS      — discarded warmup BABA pairs (default 2)

Output (parsable):
  DECODE   median=... mean=... stdev=... n=...
  BASELINE median=... mean=... stdev=... n=...
  Welch t (BASELINE vs DECODE) = ...   (positive = DECODE faster)
  DELTA_MEDIAN_PCT DECODE_vs_BASELINE = +X.XXX%
  DECODE_TFLOPS_LIST x,y,z,...
  BASELINE_TFLOPS_LIST x,y,z,...
"""
import importlib.util
import math
import os
import statistics
import subprocess
import sys
import time

import torch

torch.manual_seed(0)

M = int(os.environ['M']); N = int(os.environ['N']); K = int(os.environ['K'])
n_pairs = int(os.environ.get('N_PAIRS', '5'))
warmup = int(os.environ.get('MXFP8_WARMUP', '30'))
iters = int(os.environ.get('MXFP8_ITERS', '50'))
phys_gpu = os.environ.get('PHYS_GPU', '0')
so_dec = os.environ['SO_DECODE']; mod_dec_n = os.environ['MOD_DECODE']
so_base = os.environ['SO_BASELINE']; mod_base_n = os.environ['MOD_BASELINE']
layout = os.environ.get('LAYOUT', 'rcr').lower()
preheat_s = float(os.environ.get('PREHEAT_S', '30'))
warmup_pairs = int(os.environ.get('WARMUP_PAIRS', '2'))

assert layout in ('rcr', 'rrr', 'crr'), f"bad LAYOUT={layout}"
assert so_dec != so_base, "SO_DECODE == SO_BASELINE; need two distinct .so"
assert mod_dec_n != mod_base_n, "MOD_DECODE == MOD_BASELINE; need distinct PY_MODULE_NAME"

def load(modname, sopath):
    spec = importlib.util.spec_from_file_location(modname, sopath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read_sclk():
    try:
        out = subprocess.check_output(['rocm-smi', '--showclocks', '-d', phys_gpu],
                                      stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if 'sclk clock level' in line:
                return line.strip()
    except Exception as e:
        return f'sclk-err: {e}'
    return 'unk'

mod_dec = load(mod_dec_n, so_dec)
mod_base = load(mod_base_n, so_base)
assert mod_dec is not mod_base, (
    "PY_MODULE_NAME collision between decode and baseline .so")

print(f'[sclk-pre-preheat] {read_sclk()}', flush=True)
A_h = torch.randn(16384, 16384, device='cuda', dtype=torch.float16)
B_h = torch.randn(16384, 16384, device='cuda', dtype=torch.float16)
t0 = time.time(); i = 0
while time.time() - t0 < preheat_s:
    C_h = A_h @ B_h; i += 1
torch.cuda.synchronize()
print(f'[preheat {preheat_s}s] done {i} iters', flush=True)
del A_h, B_h, C_h
torch.cuda.empty_cache()
print(f'[sclk-post-preheat] {read_sclk()}', flush=True)

def gen_fp8(r, c):
    x = torch.randn(r, c, dtype=torch.float32, device='cuda') * 0.05
    return x.to(torch.float8_e4m3fn)

def gen_scale(r, kb):
    return torch.randint(-2, 3, (r, kb), dtype=torch.int8, device='cuda')

def encode_raw(s):
    raw = (s.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(s == -128, torch.full_like(raw, 0xFF), raw)

def preshuffle_v1(scale_exp):
    rows, kb = scale_exp.shape
    padded_rows = math.ceil(rows / 32) * 32
    padded_kb = math.ceil(kb / 8) * 8
    raw = torch.full((padded_rows, padded_kb), 0x7F, dtype=torch.uint8,
                     device=scale_exp.device)
    raw[:rows, :kb] = encode_raw(scale_exp)
    sh = raw.view(padded_rows // 32, 2, 16, padded_kb // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return sh.view(padded_rows // 32, padded_kb * 32)

k_blocks = (K + 31) // 32

# Build inputs for the chosen layout
if layout == 'rcr':
    A = gen_fp8(M, K); B = gen_fp8(N, K)
    A_se = gen_scale(M, k_blocks); B_se = gen_scale(N, k_blocks)
    A_s = preshuffle_v1(A_se); B_s = preshuffle_v1(B_se)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    fn_dec = lambda: mod_dec.gemm_rcr_pq(A, B, A_s, B_s, C)
    fn_base = lambda: mod_base.gemm_rcr_pq(A, B, A_s, B_s, C)
elif layout == 'rrr':
    A = gen_fp8(M, K); B = gen_fp8(K, N)
    A_se = gen_scale(M, k_blocks); B_se = gen_scale(N, k_blocks)
    A_s = preshuffle_v1(A_se); B_s = preshuffle_v1(B_se)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    fn_dec = lambda: mod_dec.gemm_rrr_pq(A, B, A_s, B_s, C)
    fn_base = lambda: mod_base.gemm_rrr_pq(A, B, A_s, B_s, C)
else:  # crr
    A = gen_fp8(K, M); B = gen_fp8(K, N)
    A_se = gen_scale(M, k_blocks); B_se = gen_scale(N, k_blocks)
    A_s = preshuffle_v1(A_se); B_s = preshuffle_v1(B_se)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    fn_dec = lambda: mod_dec.gemm_crr_pq(A, B, A_s, B_s, C)
    fn_base = lambda: mod_base.gemm_crr_pq(A, B, A_s, B_s, C)

# Correctness sanity: SNR check on one call each
for fn, name in [(fn_dec, 'DECODE'), (fn_base, 'BASELINE')]:
    C.zero_(); fn(); torch.cuda.synchronize()
    # Compute MXFP8 reference
    Af = A.to(torch.float32); Bf = B.to(torch.float32)
    A_sf = (2.0 ** A_se.to(torch.float32))
    B_sf = (2.0 ** B_se.to(torch.float32))
    if layout == 'rrr':
        Kr = Af.shape[1]
        A_sf_full = A_sf.repeat_interleave(32, dim=1)[:, :Kr]
        B_sf_full = B_sf.repeat_interleave(32, dim=1)[:, :Kr]
        Cref = (Af * A_sf_full) @ (Bf * B_sf_full.transpose(0, 1))
    elif layout == 'crr':
        Kr = Af.shape[0]
        A_sf_full = A_sf.repeat_interleave(32, dim=1)[:, :Kr]
        B_sf_full = B_sf.repeat_interleave(32, dim=1)[:, :Kr]
        Cref = (Af * A_sf_full.transpose(0, 1)).transpose(0, 1) @ (Bf * B_sf_full.transpose(0, 1))
    else:
        Kr = Af.shape[1]
        A_sf_full = A_sf.repeat_interleave(32, dim=1)[:, :Kr]
        B_sf_full = B_sf.repeat_interleave(32, dim=1)[:, :Kr]
        Cref = (Af * A_sf_full) @ (Bf * B_sf_full).transpose(0, 1)
    diff = (C[:M, :N].float() - Cref).abs()
    sig = (Cref * Cref).sum().item()
    noise = (diff * diff).sum().item()
    snr = 10 * math.log10(sig / noise) if noise > 0 else float('inf')
    print(f'CORRECTNESS_{name} snr_db={snr:.2f}', flush=True)

def bench_fn(fn):
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        C.zero_(); fn()
    timings = []
    for _ in range(iters):
        C.zero_(); torch.cuda.synchronize()
        se.record(); fn(); ee.record(); torch.cuda.synchronize()
        timings.append(se.elapsed_time(ee))
    return sum(timings) / len(timings)

flops = 2.0 * M * N * K
dec_tf, base_tf = [], []
print(f'[sclk-pre-bench] {read_sclk()}', flush=True)
for w in range(warmup_pairs):
    avg_d = bench_fn(fn_dec); td = flops / (avg_d * 1e9)
    avg_b = bench_fn(fn_base); tb = flops / (avg_b * 1e9)
    print(f'WARMUP {w} DECODE={td:.4f} BASELINE={tb:.4f}', flush=True)
for r in range(n_pairs):
    avg_d = bench_fn(fn_dec); td = flops / (avg_d * 1e9)
    avg_b = bench_fn(fn_base); tb = flops / (avg_b * 1e9)
    avg_d2 = bench_fn(fn_dec); td2 = flops / (avg_d2 * 1e9)
    avg_b2 = bench_fn(fn_base); tb2 = flops / (avg_b2 * 1e9)
    dec_tf.extend([td, td2]); base_tf.extend([tb, tb2])
    print(f'PAIR {r} D1={td:.4f} B1={tb:.4f} D2={td2:.4f} B2={tb2:.4f}', flush=True)
print(f'[sclk-post-bench] {read_sclk()}', flush=True)

def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(sa/na + sb/nb)
    return (ma - mb) / se if se > 0 else float('nan')

print(f'BASELINE median={statistics.median(base_tf):.4f} mean={statistics.mean(base_tf):.4f} '
      f'stdev={statistics.stdev(base_tf):.4f} n={len(base_tf)}')
print(f'DECODE   median={statistics.median(dec_tf):.4f} mean={statistics.mean(dec_tf):.4f} '
      f'stdev={statistics.stdev(dec_tf):.4f} n={len(dec_tf)}')
t = welch(dec_tf, base_tf)
print(f'Welch t (DECODE vs BASELINE) = {t:.3f}  (positive = DECODE faster)')
delta_pct = (statistics.median(dec_tf) - statistics.median(base_tf)) / statistics.median(base_tf) * 100
print(f'DELTA_MEDIAN_PCT DECODE_vs_BASELINE = {delta_pct:+.4f}%')
print(f'DECODE_TFLOPS_LIST {",".join(f"{x:.4f}" for x in dec_tf)}')
print(f'BASELINE_TFLOPS_LIST {",".join(f"{x:.4f}" for x in base_tf)}')
