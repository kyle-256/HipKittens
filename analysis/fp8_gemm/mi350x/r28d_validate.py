"""R28 Dev D: Stage 2/3 validation harness for rectangular BLK_N=128 scaffolding.

Stage 2 (V1 fallback): use gemm_crr_pq (V1) directly with V1 host preshuffle.
Stage 3 (V2 correctness): use gemm_crr_pq_v2 (V2 dispatch falls back to dispatch<L,true>
                          when MXFP8_RECT_BLK_N=64 disables the V2-CRR fastpath).

Usage: HIP_VISIBLE_DEVICES=1 python3 r28d_validate.py <stage> <layout> M N K
       stage in {v1, v2}
"""
import math
import os
import statistics
import sys

import torch

torch.manual_seed(0)

import tk_mxfp8_layouts

stage = sys.argv[1].lower()
layout = sys.argv[2].lower()
M, N, K = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
n_runs = int(os.environ.get("N_RUNS", "5"))
warmup = int(os.environ.get("MXFP8_WARMUP", "20"))
iters = int(os.environ.get("MXFP8_ITERS", "100"))


def gen_fp8(rows, cols):
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda") * 0.05
    return x.to(torch.float8_e4m3fn)


def gen_scale(rows, kb):
    return torch.randint(-2, 3, (rows, kb), dtype=torch.int8, device="cuda")


def encode_raw(s):
    raw = (s.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(s == -128, torch.full_like(raw, 0xFF), raw)


def preshuffle_v1(scale_exp):
    """V1 mfma16 preshuffle: matches preshuffle_scale_matrix_mfma16 in test_mxfp8_python."""
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
            perm_view[:, wm, 2 * pidx, :, :] = rg_view[:, rg_base + pidx, :, :]
            perm_view[:, wm, 2 * pidx + 1, :, :] = rg_view[:, rg_hi + pidx, :, :]
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


def expand_row(s, cols):
    return torch.pow(2.0, s.float()).repeat_interleave(32, dim=1)[:, :cols]


def expand_col(s, rows):
    return torch.pow(2.0, s.float()).transpose(0, 1).repeat_interleave(32, dim=0)[:rows, :]


k_blocks = (K + 31) // 32

A_row = gen_fp8(M, K)  # for RCR/RRR
A_col = gen_fp8(K, M)  # for CRR
B_row_n = gen_fp8(N, K)  # for RCR
B_col_k = gen_fp8(K, N)  # for RRR/CRR
Ase = gen_scale(M, k_blocks)
Bse = gen_scale(N, k_blocks)

if stage == "v1":
    if layout == "crr":
        A = A_col
        B = B_col_k
        As = preshuffle_v1(Ase)
        Bs = preshuffle_v1(Bse)
        fn_call = lambda C: tk_mxfp8_layouts.gemm_crr_pq(A, B, As, Bs, C)
        A_ref = A.float() * expand_col(Ase, K)
        B_ref = B.float() * expand_col(Bse, K)
        C_ref = A_ref.T @ B_ref
    else:
        raise SystemExit("v1 stage only supports crr in this scaffolding")
elif stage == "v2":
    if layout == "crr":
        A = A_col
        B = B_col_k
        As = preshuffle_v2_a(Ase)
        Bs = preshuffle_v2_b(Bse)
        fn_call = lambda C: tk_mxfp8_layouts.gemm_crr_pq_v2(A, B, As, Bs, C)
        A_ref = A.float() * expand_col(Ase, K)
        B_ref = B.float() * expand_col(Bse, K)
        C_ref = A_ref.T @ B_ref
    else:
        raise SystemExit("v2 stage only supports crr in this scaffolding")
else:
    raise SystemExit(f"unknown stage {stage} (use v1 or v2)")

C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
fn = lambda: fn_call(C)

# correctness
fn()
torch.cuda.synchronize()
diff = (C[:M, :N].float() - C_ref).abs()
sig_p = (C_ref * C_ref).sum().item()
noise_p = (diff * diff).sum().item()
snr = 10 * math.log10(sig_p / noise_p) if noise_p > 0 else float("inf")
pass_rate = ((diff <= 3.0) | (diff / C_ref.abs().clamp(min=1.0) <= 0.10)).float().mean().item() * 100
print(f"CORRECTNESS stage={stage} layout={layout} M={M} N={N} K={K} snr_db={snr:.2f} pass_rate_pct={pass_rate:.2f}", flush=True)

# determinism
C.zero_(); fn(); ref0 = C[:M, :N].clone()
det_ok = True
for _ in range(2):
    C.zero_(); fn()
    if not torch.equal(C[:M, :N], ref0):
        det_ok = False
        break
print(f"DETERMINISM ok={det_ok}", flush=True)

# bench
flops = 2.0 * M * N * K
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

tflops_list = []
for r in range(n_runs):
    avg_ms = bench_once()
    tfl = flops / (avg_ms * 1e9)
    tflops_list.append(tfl)
    print(f"RUN {r} avg_ms={avg_ms:.4f} tflops={tfl:.2f}", flush=True)

if len(tflops_list) > 0:
    med = statistics.median(tflops_list)
    sd = statistics.stdev(tflops_list) if len(tflops_list) > 1 else 0.0
    mean = statistics.mean(tflops_list)
    print(f"SUMMARY median={med:.2f} mean={mean:.2f} stdev={sd:.2f} runs={n_runs}", flush=True)
