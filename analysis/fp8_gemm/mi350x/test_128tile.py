#!/usr/bin/env python3
"""Test correctness and benchmark the 128-tile MXFP4 kernel."""
import gc, json, math, sys, time, os, torch
torch.manual_seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Also build/import the reference kernel
import subprocess, sysconfig

M, N, K = 128256, 32768, 4096
WARMUP = 200
ITERS = 500
TRIM_FRAC = 0.10

k_blocks = K // 32

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo

def preshuffle_mfma16_merged(scale_exp):
    """Merged preshuffle: [M/64, pk*64] -- dwordx2 scale loads."""
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

print(f"Testing 128-tile kernel: M={M}, N={N}, K={K}")

# Import the 128-tile kernel
import tk_mxfp4_128tile

# Generate data
print("Generating data...")
A = gen_fp4(M, K)
B = gen_fp4(N, K)
sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
A_sc = preshuffle_mfma16_merged(sc_exp_a)
B_sc = preshuffle_mfma16_merged(sc_exp_b)
C_128 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

print("Running 128-tile kernel...")
tk_mxfp4_128tile.gemm_rcr(A, B, A_sc, B_sc, C_128)
torch.cuda.synchronize()
print(f"  Result range: [{C_128.min().item():.4f}, {C_128.max().item():.4f}]")
print(f"  Non-zero fraction: {(C_128 != 0).float().mean().item():.4f}")

# Check for NaN/Inf
nan_count = torch.isnan(C_128).sum().item()
inf_count = torch.isinf(C_128).sum().item()
print(f"  NaN count: {nan_count}, Inf count: {inf_count}")

if nan_count > 0 or inf_count > 0:
    print("FAIL: NaN or Inf detected!")
    sys.exit(1)

# Benchmark
print(f"\nBenchmarking: warmup={WARMUP}, iters={ITERS}")
run = lambda: tk_mxfp4_128tile.gemm_rcr(A, B, A_sc, B_sc, C_128)

for _ in range(WARMUP):
    run()
torch.cuda.synchronize()

times_ms = []
for _ in range(ITERS):
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    run()
    end_evt.record()
    torch.cuda.synchronize()
    times_ms.append(start_evt.elapsed_time(end_evt))

times_ms.sort()
trim_count = int(len(times_ms) * TRIM_FRAC)
if trim_count > 0:
    trimmed = times_ms[trim_count:-trim_count]
else:
    trimmed = times_ms
avg_ms = sum(trimmed) / len(trimmed)

tflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12
print(f"\n128-tile kernel:")
print(f"  avg_ms = {avg_ms:.4f}")
print(f"  TFLOPS = {tflops:.1f}")
print(f"  (target: 4400 TFLOPS, baseline: 3904 TFLOPS)")
