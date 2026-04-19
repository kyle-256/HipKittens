"""R45 Dev A — FP8 per-tensor reference bench (single .so).

Single .so call to compute MXFP8/FP8 ratio. Mirrors r44a_decode_m2_16_bench.py
but trimmed to only FP8 path (single-call median over 100 iters).

Env vars:
  M, N, K, SO, MOD, LAYOUT (rcr|rrr|crr), PREHEAT_S, MXFP8_WARMUP, MXFP8_ITERS
"""
import importlib.util, math, os, statistics, subprocess, sys, time
import torch
torch.manual_seed(0)

M = int(os.environ['M']); N = int(os.environ['N']); K = int(os.environ['K'])
warmup = int(os.environ.get('MXFP8_WARMUP', '30'))
iters = int(os.environ.get('MXFP8_ITERS', '100'))
preheat_s = float(os.environ.get('PREHEAT_S', '30'))
phys_gpu = os.environ.get('PHYS_GPU', '0')
so = os.environ['SO']; mod_n = os.environ['MOD']
layout = os.environ.get('LAYOUT', 'rcr').lower()

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
    except Exception as e: return f'sclk-err: {e}'
    return 'unk'

mod = load(mod_n, so)
print(f'[sclk-pre-preheat] {read_sclk()}', flush=True)
A_h = torch.randn(16384, 16384, device='cuda', dtype=torch.float16)
B_h = torch.randn(16384, 16384, device='cuda', dtype=torch.float16)
t0 = time.time(); i = 0
while time.time() - t0 < preheat_s:
    C_h = A_h @ B_h; i += 1
torch.cuda.synchronize()
del A_h, B_h, C_h; torch.cuda.empty_cache()
print(f'[sclk-post-preheat] {read_sclk()}', flush=True)

def gen_fp8(r, c):
    x = torch.randn(r, c, dtype=torch.float32, device='cuda') * 0.05
    return x.to(torch.float8_e4m3fn)

if layout == 'rrr':
    A = gen_fp8(M, K); B = gen_fp8(K, N)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    fn = lambda: mod.gemm_rrr(A, B, C, 1.0)
elif layout == 'crr':
    A = gen_fp8(K, M); B = gen_fp8(K, N)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    fn = lambda: mod.gemm_crr(A, B, C, 1.0)
else:
    A = gen_fp8(M, K); B = gen_fp8(N, K)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    fn = lambda: mod.gemm_rcr(A, B, C, 1.0)

# Bench
se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
for _ in range(warmup):
    C.zero_(); fn()
timings = []
for _ in range(iters):
    C.zero_(); torch.cuda.synchronize()
    se.record(); fn(); ee.record(); torch.cuda.synchronize()
    timings.append(se.elapsed_time(ee))
print(f'[sclk-post-bench] {read_sclk()}', flush=True)
flops = 2.0 * M * N * K
ms_med = statistics.median(timings)
tf = flops / (ms_med * 1e9)
print(f'FP8_REF M={M} N={N} K={K} layout={layout} median_ms={ms_med:.6f} tflops={tf:.4f} '
      f'mean_ms={statistics.mean(timings):.6f} stdev_ms={statistics.stdev(timings):.6f}', flush=True)
