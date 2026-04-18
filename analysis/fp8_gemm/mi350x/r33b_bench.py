"""R33 Dev B: bench harness for rect-V2 RCR Stage A2.

Adapts r29a_bench.py (square V2-RCR) with a new `preshuffle_v2_b_rect`
that targets the rect-V2 RCR slab layout (BLK_N=128, HB_N=64, RBN_RECT=16,
WARPS_N=4, pack_count=2). The kernel uses SAME b_voff/b_soff strides as
square V2-RCR (PC=2, b64 load) — the only difference is that there are 2x
more slabs (one per BLK_N=128 ctile instead of one per BLK=256 ctile).

Usage: HIP_VISIBLE_DEVICES=1 python3 r33b_bench.py rect|square M N K
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


print(f"[r33b sclk-pre-preheat] {read_sclk()}", file=sys.stderr, flush=True)
print(f"[r33b preheat] sustained 30s 16k matmul...", file=sys.stderr, flush=True)
A_h = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
B_h = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
t0 = time.time()
i = 0
while time.time() - t0 < 30.0:
    C_h = A_h @ B_h
    i += 1
torch.cuda.synchronize()
print(f"[r33b preheat] done {i} iters", file=sys.stderr, flush=True)
del A_h, B_h, C_h
torch.cuda.empty_cache()
print(f"[r33b sclk-post-preheat] {read_sclk()}", file=sys.stderr, flush=True)


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
    """SQUARE V2-RCR B preshuffle (from r29a_bench.py)."""
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
    """RECT V2-RCR B preshuffle (R33 Dev B Stage A2a).

    Differences vs square:
      - blk_n=128 instead of blk=256 -> num_ctiles doubles -> num_slabs doubles
      - hb_n=64, rbn_rect=16 — each (wn, half) covers 16 rows of a 32-row group
      - pack_count=2 (b0+b1) — SAME as square, kept so that the kernel's b_voff/b_soff
        strides remain identical to the square kernel
      - Each pack still stores a full 32-row e8m0 group; the rect kernel only consumes
        the first 16 rows (lane_nonk in [0,15] maps to a single warp's RBN_RECT=16 tile)
    """
    pack_count = 2
    rows, kb = scale_exp.shape
    padded_rows = math.ceil(rows / blk_n) * blk_n
    padded_kb = math.ceil(kb / 8) * 8
    num_ctiles = padded_rows // blk_n
    num_slabs = num_ctiles * warps_n
    rgs_per_ctile = blk_n // 32  # = 4 for blk_n=128

    raw = torch.full((padded_rows, padded_kb), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = encode_raw(scale_exp)

    # Build the per-slab raw e8m0 source: each slab = (32 rows x padded_kb).
    # For (ctile, wn): pack0 covers the 16-row sub-range starting at rgs(half=0) of (ctile, wn);
    #                  pack1 covers the 16-row sub-range starting at rgs(half=1) of (ctile, wn).
    # These 16-row chunks live INSIDE 32-row row_groups; we copy the whole 32-row group
    # but the kernel only consumes the first 16 lane_nonk slots that match its RBN_RECT.
    #
    # Mapping (matches kernel `rcr_scale_b_base_rect(half) = bc*128 + half*64 + wn*16`):
    #   rows for (wn, half=0) = [bc*128 + wn*16 : bc*128 + wn*16 + 16)
    #   rows for (wn, half=1) = [bc*128 + 64 + wn*16 : bc*128 + 64 + wn*16 + 16)
    # Each 16-row sub-range belongs to row_group floor(start_row / 32) within the ctile.
    perm = torch.zeros(num_slabs, pack_count, 32, padded_kb, dtype=torch.uint8, device=scale_exp.device)

    # Index by ctile and wn, fill pack0/pack1 with the 16-row sub-range from raw,
    # placed in the first 16 rows of the 32-row e8m0 row_group.
    raw_ct = raw.view(num_ctiles, rgs_per_ctile, 32, padded_kb)  # but rows are in row-major; reshape
    # Actually raw has shape (num_ctiles*blk_n, padded_kb). reshape:
    raw_ct = raw.view(num_ctiles, blk_n, padded_kb)  # ctile, row_in_ctile, kb
    for wn in range(warps_n):
        # half=0 (b0): rows [wn*rbn_rect : wn*rbn_rect + rbn_rect)
        h0_start = wn * rbn_rect
        # half=1 (b1): rows [hb_n + wn*rbn_rect : hb_n + wn*rbn_rect + rbn_rect)
        h1_start = hb_n + wn * rbn_rect
        # Place into first 16 rows of the 32-row pack (rest stays 0x7F-equivalent or 0)
        # We use 0x7F (which encodes scale=0 -> 2^0=1 -- but actually e8m0 0x7F decodes
        # to 1.0; we don't care as the kernel won't read those rows).
        # Use slab indexing: slab(ctile, wn) -> num_slabs index = ctile * warps_n + wn
        perm[:, 0, :rbn_rect, :] = raw_ct[:, h0_start:h0_start + rbn_rect, :].contiguous() if False else 0
    # Vectorize with slicing:
    perm_view = perm.view(num_ctiles, warps_n, pack_count, 32, padded_kb)
    for wn in range(warps_n):
        h0_start = wn * rbn_rect
        h1_start = hb_n + wn * rbn_rect
        perm_view[:, wn, 0, :rbn_rect, :] = raw_ct[:, h0_start:h0_start + rbn_rect, :]
        perm_view[:, wn, 1, :rbn_rect, :] = raw_ct[:, h1_start:h1_start + rbn_rect, :]
        # Pad the unused 16 rows with 0x7F (e8m0 = 1.0 — won't be read anyway).
        perm_view[:, wn, 0, rbn_rect:, :] = 0x7F
        perm_view[:, wn, 1, rbn_rect:, :] = 0x7F

    kp_count = padded_kb // 8
    rv = perm.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    sh = rv.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return sh.view(num_slabs, pack_count * 32 * padded_kb)


def expand_col(s, rows):
    return torch.pow(2.0, s.float()).transpose(0, 1).repeat_interleave(32, dim=0)[:rows, :]


def expand_row(s, cols):
    return torch.pow(2.0, s.float()).repeat_interleave(32, dim=1)[:, :cols]


k_blocks = (K + 31) // 32

# RCR (per r28a_bench5x.py): A is (M, K) row-layout, B is (N, K) row-layout but
# logically B^T in the GEMM expression. C = A @ B^T.
A = gen_fp8(M, K)
B = gen_fp8(N, K)
Ase = gen_scale(M, k_blocks)
Bse = gen_scale(N, k_blocks)

As = preshuffle_v2_a(Ase)
if mode == "rect":
    Bs = preshuffle_v2_b_rect(Bse)
elif mode == "square":
    Bs = preshuffle_v2_b(Bse)
else:
    raise SystemExit(f"unknown mode {mode}")

print(f"[r33b] mode={mode} As.shape={As.shape} Bs.shape={Bs.shape}", flush=True)

# Reference (matches r28a_bench5x.py RCR path).
A_ref = A.float() * expand_row(Ase, K)
B_ref = B.float() * expand_row(Bse, K)
C_ref = A_ref @ B_ref.T

C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
fn_call = lambda C: tk_mxfp8_layouts.gemm_rcr_pq_v2(A, B, As, Bs, C)
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
print(f"CORRECTNESS mode={mode} M={M} N={N} K={K} snr_db={snr:.2f} pass_rate_pct={pass_rate:.2f}", flush=True)
print(f"C[0, :8] = {C[0, :8].tolist()}", flush=True)
print(f"C_ref[0, :8] = {C_ref[0, :8].tolist()}", flush=True)

# Determinism
C.zero_(); fn(); ref0 = C[:M, :N].clone()
det_ok = True
for _ in range(2):
    C.zero_(); fn()
    if not torch.equal(C[:M, :N], ref0):
        det_ok = False
        break
print(f"DETERMINISM ok={det_ok}", flush=True)

if os.environ.get("R33B_BENCH_ONLY_CORRECTNESS", "0") == "1":
    sys.exit(0)

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
