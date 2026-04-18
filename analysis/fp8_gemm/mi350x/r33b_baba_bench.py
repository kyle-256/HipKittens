"""R33 Dev B: paired BABA bench for rect-V2 RCR vs square-V2 RCR @ 4096^3.

Loads BOTH preshuffles (square + rect) into the SAME process and runs the
gemm_rcr_pq_v2 dispatcher in BABA pattern. The dispatcher routes to rect when
MXFP8_RECT_BLK_N=64, so we toggle by swapping in the matching B preshuffle:
  - 'square' uses preshuffle_v2_b
  - 'rect' uses preshuffle_v2_b_rect
But the kernel build is fixed (rect mode active when MXFP8_RECT_BLK_N=64 at
build time). To compare rect vs square in the same process, we need TWO builds
or a runtime toggle. Workaround: build with rect and use the rect dispatch for
'rect' mode, BUT for 'square' mode we have to call the SQUARE preshuffle and
the RECT kernel which won't match. So instead we run TWO separate processes.

Strategy: this script measures rect-only OR square-only, controlled by mode arg.
The orchestrator (r33b_orchestrate.sh) builds rect, runs rect mode N times,
rebuilds default (no rect macro), runs square mode N times, then computes
Welch t-test from the lists. BABA pattern is enforced by interleaving runs.

Usage: HIP_VISIBLE_DEVICES=1 PHYS_GPU=1 python3 r33b_baba_bench.py rect|square M N K
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

mode = sys.argv[1].lower()
M, N, K = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
n_runs = int(os.environ.get("N_RUNS", "5"))
warmup = int(os.environ.get("MXFP8_WARMUP", "50"))
iters = int(os.environ.get("MXFP8_ITERS", "100"))
phys_gpu = os.environ.get("PHYS_GPU", "1")


def read_sclk():
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showclocks", "-d", phys_gpu], stderr=subprocess.DEVNULL
        ).decode()
        for line in out.splitlines():
            if "sclk clock level" in line:
                return line.strip()
    except Exception as e:
        return f"sclk-read-err: {e}"
    return "sclk-unknown"


def gen_fp8(rows, cols):
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda") * 0.05
    return x.to(torch.float8_e4m3fn)


def gen_scale(rows, kb):
    return torch.randint(-2, 3, (rows, kb), dtype=torch.int8, device="cuda")


def encode_raw(s):
    raw = (s.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(s == -128, torch.full_like(raw, 0xFF), raw)


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


def preshuffle_v2_b_rect(scale_exp, blk_n=128, hb_n=64, rbn_rect=16, warps_n=4):
    pack_count = 2
    rows, kb = scale_exp.shape
    padded_rows = math.ceil(rows / blk_n) * blk_n
    padded_kb = math.ceil(kb / 8) * 8
    num_ctiles = padded_rows // blk_n
    num_slabs = num_ctiles * warps_n
    raw = torch.full((padded_rows, padded_kb), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = encode_raw(scale_exp)
    raw_ct = raw.view(num_ctiles, blk_n, padded_kb)
    perm = torch.zeros(num_slabs, pack_count, 32, padded_kb, dtype=torch.uint8, device=scale_exp.device)
    perm_view = perm.view(num_ctiles, warps_n, pack_count, 32, padded_kb)
    for wn in range(warps_n):
        h0_start = wn * rbn_rect
        h1_start = hb_n + wn * rbn_rect
        perm_view[:, wn, 0, :rbn_rect, :] = raw_ct[:, h0_start:h0_start + rbn_rect, :]
        perm_view[:, wn, 1, :rbn_rect, :] = raw_ct[:, h1_start:h1_start + rbn_rect, :]
        perm_view[:, wn, 0, rbn_rect:, :] = 0x7F
        perm_view[:, wn, 1, rbn_rect:, :] = 0x7F
    kp_count = padded_kb // 8
    rv = perm.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    sh = rv.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return sh.view(num_slabs, pack_count * 32 * padded_kb)


def expand_row(s, cols):
    return torch.pow(2.0, s.float()).repeat_interleave(32, dim=1)[:, :cols]


k_blocks = (K + 31) // 32

# Allocate inputs
A = gen_fp8(M, K)
B = gen_fp8(N, K)
Ase = gen_scale(M, k_blocks)
Bse = gen_scale(N, k_blocks)

As = preshuffle_v2_a(Ase)
Bs_square = preshuffle_v2_b(Bse)
Bs_rect = preshuffle_v2_b_rect(Bse)

print(f"[r33b-baba sclk-pre-preheat] {read_sclk()}", file=sys.stderr, flush=True)
print(f"[r33b-baba preheat] sustained 30s 16k matmul...", file=sys.stderr, flush=True)
A_h = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
B_h = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
t0 = time.time()
i = 0
while time.time() - t0 < 30.0:
    C_h = A_h @ B_h
    i += 1
torch.cuda.synchronize()
print(f"[r33b-baba preheat] done {i} iters", file=sys.stderr, flush=True)
del A_h, B_h, C_h
torch.cuda.empty_cache()
print(f"[r33b-baba sclk-post-preheat] {read_sclk()}", file=sys.stderr, flush=True)

C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

if mode == "rect":
    Bs = Bs_rect
elif mode == "square":
    Bs = Bs_square
else:
    raise SystemExit(f"unknown mode {mode}")

fn = lambda: tk_mxfp8_layouts.gemm_rcr_pq_v2(A, B, As, Bs, C)


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

print(f"[sclk-pre-bench] {read_sclk()}", file=sys.stderr, flush=True)

tflops_list = []
for r in range(n_runs):
    avg_ms = bench_once()
    tfl = flops / (avg_ms * 1e9)
    tflops_list.append(tfl)
    print(f"RUN {r} mode={mode} avg_ms={avg_ms:.4f} tflops={tfl:.2f}", flush=True)

print(f"[sclk-post-bench] {read_sclk()}", file=sys.stderr, flush=True)

med = statistics.median(tflops_list)
sd = statistics.stdev(tflops_list) if len(tflops_list) > 1 else 0.0
mean = statistics.mean(tflops_list)
print(f"SUMMARY mode={mode} median={med:.2f} mean={mean:.2f} stdev={sd:.2f} runs={n_runs}", flush=True)
print(f"TFLOPS_LIST {','.join(f'{x:.4f}' for x in tflops_list)}", flush=True)
