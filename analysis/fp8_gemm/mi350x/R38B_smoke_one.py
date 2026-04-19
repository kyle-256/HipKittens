#!/usr/bin/env python3
"""R38B smoke test: bench one CRASH shape with full warmup/iters and check for crash.

Default shape: m32768_n4096_k2048 / variant ts_lgk2_gm6_v12_memc_pfoff4 (the most
reliably-crashing CRASH shape from R37). Uses HIP_VISIBLE_DEVICES=0 by default.
"""
import importlib.util, json, math, os, sys, sysconfig, time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R38B")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GATE = 0.995

M, N, K = 32768, 4096, 2048
PARENT_TAG = "ts_lgk2_gm6_v12_memc_pfoff4"

if len(sys.argv) > 1:
    M, N, K = (int(x) for x in sys.argv[1].split("x"))
if len(sys.argv) > 2:
    PARENT_TAG = sys.argv[2]

torch.manual_seed(0)

module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}_{PARENT_TAG}_R38B"
so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
print(f"Loading: {so_path}")
assert os.path.exists(so_path), f"missing: {so_path}"

spec = importlib.util.spec_from_file_location(module_name, so_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda') << 4) | \
           torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda')


def preshuffle(se):
    r, kb = se.shape
    pr = math.ceil(r / 64) * 64
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=se.device)
    raw[:r, :kb] = (se.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(pr // 32, pk * 32)
    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4).permute(0, 2, 1, 3).contiguous()
    return sh.view(pr // 64, pk * 64)


print(f"Shape: {M}x{N}x{K}, variant: {PARENT_TAG}")
A = gen_fp4(M, K)
B = gen_fp4(N, K)

sc_a_corr = torch.full((M, K // 32), -4, dtype=torch.int8, device='cuda')
sc_b_corr = torch.full((N, K // 32), -4, dtype=torch.int8, device='cuda')
A_sc_corr = preshuffle(sc_a_corr)
B_sc_corr = preshuffle(sc_b_corr)

sc_a = torch.randint(-2, 3, (M, K // 32), dtype=torch.int8, device='cuda')
sc_b = torch.randint(-2, 3, (N, K // 32), dtype=torch.int8, device='cuda')
A_sc = preshuffle(sc_a)
B_sc = preshuffle(sc_b)

C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

# Correctness check
C.zero_()
mod.gemm_rcr(A, B, A_sc_corr, B_sc_corr, C)
torch.cuda.synchronize()
finite_frac = float(torch.isfinite(C.float()).sum().item()) / float(C.numel())
finite_frac = round(finite_frac, 6)
print(f"finite_frac = {finite_frac}  (gate={GATE})")
if finite_frac < GATE:
    print("FAIL: WRONG_OUTPUT")
    sys.exit(2)

print(f"Warmup ({WARMUP})...")
for _ in range(WARMUP):
    mod.gemm_rcr(A, B, A_sc, B_sc, C)
torch.cuda.synchronize()

print(f"Bench ({ITERS} iters)...")
times = []
t0 = time.time()
for i in range(ITERS):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    mod.gemm_rcr(A, B, A_sc, B_sc, C)
    e.record()
    torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
elapsed = time.time() - t0

times.sort()
trim = int(len(times) * TRIM)
if trim > 0:
    times = times[trim:-trim]
avg = sum(times) / len(times)
tflops = 2.0 * M * N * K / (avg * 1e-3) / 1e12

print(f"PASS: tflops={tflops:.1f} avg_ms={avg:.4f} finite={finite_frac:.4f} (wall={elapsed:.1f}s)")
