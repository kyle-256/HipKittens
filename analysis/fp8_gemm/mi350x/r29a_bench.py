"""R29 Dev A: bench harness for rect-V2 cycle.

Supports two paths:
  v2 — V2 dispatch (gemm_crr_pq_v2) with V2 host preshuffle
  v1 — V1 dispatch (gemm_crr_pq) with V1 host preshuffle (used for the Path B
       fallback path when MXFP8_RECT_BLK_N=64 and there is no rect-V2 fastpath)

Mirrors r28a_bench5x.py preheat-then-bench pattern.

Usage: HIP_VISIBLE_DEVICES=0 python3 r29a_bench.py <stage> <layout> M N K
       stage in {v1, v2}
"""
import math
import os
import statistics
import subprocess
import sys
import time

import torch

torch.manual_seed(0)

import tk_mxfp8_layouts

stage = sys.argv[1].lower()
layout = sys.argv[2].lower()
M, N, K = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
n_runs = int(os.environ.get("N_RUNS", "5"))
warmup = int(os.environ.get("MXFP8_WARMUP", "50"))
iters = int(os.environ.get("MXFP8_ITERS", "100"))


def read_sclk():
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showclocks", "-d", "0"], stderr=subprocess.DEVNULL
        ).decode()
        for line in out.splitlines():
            if "sclk clock level" in line:
                return line.strip()
    except Exception as e:
        return f"sclk-read-err: {e}"
    return "sclk-unknown"


print(f"[sclk-pre-preheat] {read_sclk()}", file=sys.stderr, flush=True)
print(f"[preheat] sustained 8s 16k matmul...", file=sys.stderr, flush=True)
A_h = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
B_h = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
t0 = time.time()
i = 0
while time.time() - t0 < 8.0:
    C_h = A_h @ B_h
    i += 1
torch.cuda.synchronize()
print(f"[preheat] done {i} iters", file=sys.stderr, flush=True)
del A_h, B_h, C_h
torch.cuda.empty_cache()
print(f"[sclk-post-preheat] {read_sclk()}", file=sys.stderr, flush=True)


def gen_fp8(rows, cols):
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda") * 0.05
    return x.to(torch.float8_e4m3fn)


def gen_scale(rows, kb):
    return torch.randint(-2, 3, (rows, kb), dtype=torch.int8, device="cuda")


def encode_raw(s):
    raw = (s.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(s == -128, torch.full_like(raw, 0xFF), raw)


def preshuffle_v1(scale_exp):
    rows, kb = scale_exp.shape
    padded_rows = math.ceil(rows / 32) * 32
    padded_kb = math.ceil(kb / 8) * 8
    raw = torch.full((padded_rows, padded_kb), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = encode_raw(scale_exp)
    sh = raw.view(padded_rows // 32, 2, 16, padded_kb // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return sh.view(padded_rows // 32, padded_kb * 32)


def preshuffle_v2_a(scale_exp, blk=256, hb=128, rbm=64, warps_m=2):
    pack_count = 4
    rows, kb = scale_exp.shape
    padded_rows = math.ceil(rows / blk) * blk
    padded_kb = math.ceil(kb / 8) * 8
    num_ctiles = padded_rows // blk
    num_slabs = num_ctiles * warps_m
    rgs_per_ctile = blk // 32
    pack_a = rbm // 32
    raw = torch.full((padded_rows, padded_kb), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = encode_raw(scale_exp)
    perm = torch.empty_like(raw)
    rg_view = raw.view(num_ctiles, rgs_per_ctile, 32, padded_kb)
    perm_view = perm.view(num_ctiles, num_slabs // num_ctiles, pack_count, 32, padded_kb)
    for wm in range(warps_m):
        rg_base = wm * (rbm // 32)
        rg_hi = (hb // 32) + rg_base
        for pidx in range(pack_a):
            perm_view[:, wm, 2*pidx, :, :] = rg_view[:, rg_base + pidx, :, :]
            perm_view[:, wm, 2*pidx + 1, :, :] = rg_view[:, rg_hi + pidx, :, :]
    kp_count = padded_kb // 8
    rv = perm.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    sh = rv.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return sh.view(num_slabs, pack_count * 32 * padded_kb)


def preshuffle_v2_b(scale_exp, blk=256, hb=128, rbn=32, warps_n=4):
    pack_count = 2
    rows, kb = scale_exp.shape
    padded_rows = math.ceil(rows / blk) * blk
    padded_kb = math.ceil(kb / 8) * 8
    num_ctiles = padded_rows // blk
    num_slabs = num_ctiles * warps_n
    rgs_per_ctile = blk // 32
    raw = torch.full((padded_rows, padded_kb), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = encode_raw(scale_exp)
    perm = torch.empty_like(raw)
    rg_view = raw.view(num_ctiles, rgs_per_ctile, 32, padded_kb)
    perm_view = perm.view(num_ctiles, warps_n, pack_count, 32, padded_kb)
    rbn_rg = rbn // 32
    rg_hi_off = hb // 32
    for wn in range(warps_n):
        rg_base = wn * rbn_rg
        perm_view[:, wn, 0, :, :] = rg_view[:, rg_base, :, :]
        perm_view[:, wn, 1, :, :] = rg_view[:, rg_base + rg_hi_off, :, :]
    kp_count = padded_kb // 8
    rv = perm.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    sh = rv.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return sh.view(num_slabs, pack_count * 32 * padded_kb)


def expand_col(s, rows):
    return torch.pow(2.0, s.float()).transpose(0, 1).repeat_interleave(32, dim=0)[:rows, :]


k_blocks = (K + 31) // 32

if layout != "crr":
    raise SystemExit("only crr supported in r29a bench")

A = gen_fp8(K, M)
B = gen_fp8(K, N)
Ase = gen_scale(M, k_blocks)
Bse = gen_scale(N, k_blocks)

if stage == "v2":
    As = preshuffle_v2_a(Ase)
    Bs = preshuffle_v2_b(Bse)
    fn_call = lambda C: tk_mxfp8_layouts.gemm_crr_pq_v2(A, B, As, Bs, C)
elif stage == "v1":
    As = preshuffle_v1(Ase)
    Bs = preshuffle_v1(Bse)
    fn_call = lambda C: tk_mxfp8_layouts.gemm_crr_pq(A, B, As, Bs, C)
else:
    raise SystemExit(f"unknown stage {stage}")

A_ref = A.float() * expand_col(Ase, K)
B_ref = B.float() * expand_col(Bse, K)
C_ref = A_ref.T @ B_ref

C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
fn = lambda: fn_call(C)


def bench_once():
    se_start = torch.cuda.Event(enable_timing=True)
    se_end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        C.zero_(); fn()
    timings = []
    for _ in range(iters):
        C.zero_(); torch.cuda.synchronize()
        se_start.record(); fn(); se_end.record(); torch.cuda.synchronize()
        timings.append(se_start.elapsed_time(se_end))
    return sum(timings) / len(timings)


flops = 2.0 * M * N * K

# Correctness
fn()
torch.cuda.synchronize()
diff = (C[:M, :N].float() - C_ref).abs()
sig_p = (C_ref * C_ref).sum().item()
noise_p = (diff * diff).sum().item()
snr = 10 * math.log10(sig_p / noise_p) if noise_p > 0 else float("inf")
pass_rate = ((diff <= 3.0) | (diff / C_ref.abs().clamp(min=1.0) <= 0.10)).float().mean().item() * 100
print(f"CORRECTNESS stage={stage} layout={layout} M={M} N={N} K={K} snr_db={snr:.2f} pass_rate_pct={pass_rate:.2f}", flush=True)

# Determinism
C.zero_(); fn(); ref0 = C[:M, :N].clone()
det_ok = True
for _ in range(2):
    C.zero_(); fn()
    if not torch.equal(C[:M, :N], ref0):
        det_ok = False
        break
print(f"DETERMINISM ok={det_ok}", flush=True)

print(f"[sclk-pre-bench] {read_sclk()}", file=sys.stderr, flush=True)

tflops_list = []
for r in range(n_runs):
    avg_ms = bench_once()
    tfl = flops / (avg_ms * 1e9)
    tflops_list.append(tfl)
    print(f"RUN {r} avg_ms={avg_ms:.4f} tflops={tfl:.2f}", flush=True)

print(f"[sclk-post-bench] {read_sclk()}", file=sys.stderr, flush=True)

med = statistics.median(tflops_list)
sd = statistics.stdev(tflops_list) if len(tflops_list) > 1 else 0.0
mean = statistics.mean(tflops_list)
print(f"SUMMARY median={med:.2f} mean={mean:.2f} stdev={sd:.2f} runs={n_runs}", flush=True)
print(f"TFLOPS_LIST {','.join(f'{x:.4f}' for x in tflops_list)}", flush=True)
