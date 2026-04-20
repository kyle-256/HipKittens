#!/usr/bin/env python3
"""R59 INTEGRATION reviewer bench — 10-run @ 80% INDEPENDENT-seed gate, ITERS=500.

Fork of bench_all_42_R58_INTEGRATION.py with R59 deltas:
  - 0 binary changes vs R58. All 42 manifest entries byte-identical to R58.
  - R59 = 0 PROMOTE / 4 SMOKE_DEAD / 1 POLICY_ONLY (Opt R).
  - Reviewer integration is a *cohort-race repeatability test* under fresh
    INDEPENDENT seed sweep at ITERS=500 (per Opt R recommendation A).
    Specifically tests whether L3 (32768,14336,2048) HK R40B's R58 VC-flip
    (n_OK=9/10 fin_min=0.911) was a single-sweep tail-draw or intrinsic.
  - ITERS=500 default UNCHANGED (Opt R recommendation A: do NOT bump to 1000).
  - warmup=200, trim=0.10, INDEPENDENT seeds [101..1010] unchanged.
  - Adds a thin `--mode {smoke1,10run}` wrapper on top of the R58 CLI to match
    the reviewer task spec; --runs/--out/--log still work explicitly.
  Output defaults: R59_INTEGRATION_10RUN.{json,log}; round label R59_INTEGRATION

Bench MANDATORY: warmup=200, iters=500 (R45+ default; Opt R policy A), trim=0.10.
Verified-correct gate (10-run promotion):
  n_OK >= 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min >= 0.97
"""
import json
import math
import os
import subprocess
import sys
import threading
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WARMUP = 200
ITERS = 500  # R45+ default; Opt R recommendation (A) keeps ITERS=500, no mixed-protocol bump
TRIM = 0.10
SHAPE_TIMEOUT = 1100  # generous; aiter shape (4096,32768,28672) ~12s/run
SNR_THRESHOLD_DB = 10.0
WRONG_CELL_GATE = 0.02
FINITE_GATE = 0.97
RANDOM_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

# Aiter shim path (single shim works for all 40 cells; per-shape co/kname/tile vary)
AITER_SHIM_SO = "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so"
AITER_CO_256 = "/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co"
AITER_KNAME_256 = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"
AITER_CO_96x640 = "/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640.co"
AITER_KNAME_96x640 = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640E"
AITER_CO_64x1024 = "/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024.co"
AITER_KNAME_64x1024 = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024E"
AITER_CO_128x256 = "/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256.co"
AITER_KNAME_128x256 = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E"

# Per-shape aiter dispatch table — IDENTICAL to R58 (no R59 binary changes).
AITER_SHAPES = {
    (4096, 32768, 28672):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R50D"),
    (14336, 4096, 32768):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R51D1"),
    (16384, 4096, 28672):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R51D2"),
    (28672, 4096, 16384):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R51D3"),
    (4096, 28672, 32768):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R52D2A"),
    (4096, 32768, 128256): (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R52D2B"),
    (4096, 4096, 32768):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R52D2C"),
    (6144, 4096, 16384):   (96,  640,  AITER_CO_96x640, AITER_KNAME_96x640, "R53D3A_2"),
    (4096, 6144, 32768):   (96,  640,  AITER_CO_96x640, AITER_KNAME_96x640, "R53D3A_3"),
    (4096, 32768, 14336):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R57J1_L1"),
    (32768, 4096, 14336):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R56G1_L2"),
    (128256, 32768, 4096): (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R56G2_L3"),
    (14336, 32768, 4096):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R56G2_L5"),
    (28672, 32768, 4096):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R56G2_L4"),
    (16384, 4096, 14336):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R56G1_L6"),
    (4096, 128256, 32768): (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R56G4_C1"),
    (32768, 4096, 2048):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54E1_1"),
    (32768, 4096, 3072):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54E1_2"),
    (28672, 4096, 8192):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54E1_3"),
    (4096, 32768, 4096):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54E2_1"),
    (4096, 32768, 6144):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54E2_2"),
    (16384, 4096, 6144):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54E2_3"),
    (4096, 14336, 16384):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54D4A_1"),
    (32768, 4096, 7168):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54D4A_2"),
    (6144, 4096, 8192):    (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54D4A_3"),
    (4096, 4096, 16384):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54D4B_1"),
    (4096, 14336, 8192):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54D4B_2"),
    (16384, 4096, 7168):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R54D4B_3"),
    (16384, 14336, 2048):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55E3_1"),
    (16384, 14336, 4096):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55E3_2"),
    (16384, 28672, 2048):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55E3_3"),
    (16384, 28672, 4096):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55E3_4"),
    (16384, 6144, 4096):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55E4_1"),
    (6144, 32768, 4096):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55E4_2"),
    (16384, 4096, 4096):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55D5A_1"),
    (16384, 6144, 2048):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55D5A_2"),
    (4096, 4096, 8192):    (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55D5A_3"),
    (32768, 28672, 2048):  (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55D5B_2"),
    (32768, 6144, 2048):   (256, 256,  AITER_CO_256,    AITER_KNAME_256,    "R55D5B_3"),
    # R58 P-2 PROMOTE carried into R59 unchanged
    (16384, 4096, 3072):   (128, 256,  AITER_CO_128x256, AITER_KNAME_128x256, "R58P2"),
}

# (M, N, K, competitor_TFLOPS) — same 42 shapes
ALL_SHAPES = [
    (16384, 4096, 2048, 2995.0), (16384, 4096, 3072, 3492.3),
    (16384, 6144, 2048, 3047.6), (32768, 4096, 2048, 3131.8),
    (32768, 4096, 3072, 3630.6), (32768, 6144, 2048, 3239.9),
    (16384, 14336, 2048, 3301.3), (16384, 28672, 2048, 3482.3),
    (32768, 14336, 2048, 3351.4), (32768, 28672, 2048, 3353.4),
    (4096, 4096, 16384, 4642.1), (4096, 14336, 16384, 5013.0),
    (6144, 4096, 16384, 4428.1), (4096, 4096, 8192, 3959.9),
    (4096, 4096, 32768, 5152.8), (4096, 6144, 32768, 3784.2),
    (4096, 14336, 8192, 4345.8), (4096, 28672, 32768, 5649.9),
    (4096, 32768, 4096, 4166.5), (4096, 32768, 6144, 4548.6),
    (4096, 32768, 14336, 5296.1), (4096, 32768, 28672, 5568.2),
    (4096, 32768, 128256, 5781.1), (4096, 128256, 32768, 3195.3),
    (6144, 4096, 8192, 3822.0), (6144, 32768, 4096, 4291.0),
    (14336, 4096, 32768, 5245.4), (14336, 32768, 4096, 4462.6),
    (16384, 4096, 4096, 3951.8), (16384, 4096, 6144, 4259.9),
    (16384, 4096, 7168, 4443.2), (16384, 4096, 14336, 5142.1),
    (16384, 4096, 28672, 5525.3), (16384, 6144, 4096, 4042.5),
    (16384, 14336, 4096, 4255.8), (16384, 28672, 4096, 4411.7),
    (28672, 4096, 8192, 4810.0), (28672, 4096, 16384, 5350.6),
    (28672, 32768, 4096, 4466.6), (32768, 4096, 7168, 4666.8),
    (32768, 4096, 14336, 5223.4), (128256, 32768, 4096, 4536.4),
]


def make_hk_runner_script(module_name, so_path, m, n, k, comp, seed):
    return f"""\
import sys, math, json, importlib.util, torch
torch.manual_seed({seed})
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
SNR_DB_GATE = {SNR_THRESHOLD_DB}
WRONG_CELL_GATE = {WRONG_CELL_GATE}
FINITE_GATE = {FINITE_GATE}
M, N, K = {m}, {n}, {k}
COMP = {comp}

FP4_TBL = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0,
], dtype=torch.float32, device='cuda')

def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda') << 4) | \\
           torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')

def preshuffle(se):
    r, kb = se.shape
    pr = math.ceil(r/64)*64
    pk = math.ceil(kb/8)*8
    raw = torch.full((pr,pk),0x7F,dtype=torch.uint8,device=se.device)
    raw[:r,:kb] = (se.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)

def dequant_fp4_rows(packed_uint8, K, row_lo, row_hi):
    pk = packed_uint8[row_lo:row_hi]
    lo = (pk & 0x0F).to(torch.int64)
    hi = ((pk >> 4) & 0x0F).to(torch.int64)
    rows, cols = pk.shape
    out = torch.empty(rows, cols * 2, dtype=torch.float32, device=pk.device)
    out[:, 0::2] = FP4_TBL[lo]
    out[:, 1::2] = FP4_TBL[hi]
    return out[:, :K]

def apply_scales_rows(data_f32, scale_exp_i8, K_dim, row_lo, row_hi, block=32):
    sc = scale_exp_i8[row_lo:row_hi].to(torch.float32)
    scales = torch.pow(2.0, sc)
    rows, kb = scales.shape
    se = scales.unsqueeze(-1).expand(rows, kb, block).reshape(rows, -1)[:, :K_dim]
    return data_f32 * se

try:
    spec = importlib.util.spec_from_file_location({module_name!r}, {so_path!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    A = gen_fp4(M, K); B = gen_fp4(N, K)
    sc_a = torch.randint(-2, 3, (M, K//32), dtype=torch.int8, device='cuda')
    sc_b = torch.randint(-2, 3, (N, K//32), dtype=torch.int8, device='cuda')
    A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    run = lambda: mod.gemm_rcr(A, B, A_sc, B_sc, C)
    C.zero_(); run(); torch.cuda.synchronize()
    C0 = C.clone()
    finite_frac = float(torch.isfinite(C0.float()).sum().item()) / float(C0.numel())
    finite_frac = round(finite_frac, 6)

    MAX_M_ROWS = 1024
    MAX_N_ROWS = 1024
    m_rows = min(M, MAX_M_ROWS)
    n_rows = min(N, MAX_N_ROWS)

    A_blk = dequant_fp4_rows(A, K, 0, m_rows)
    B_blk = dequant_fp4_rows(B, K, 0, n_rows)
    A_scaled = apply_scales_rows(A_blk, sc_a, K, 0, m_rows)
    B_scaled = apply_scales_rows(B_blk, sc_b, K, 0, n_rows)
    del A_blk, B_blk

    C_ref = torch.matmul(A_scaled, B_scaled.T)
    del A_scaled, B_scaled

    C_test = C0[:m_rows, :n_rows].float()
    base_mask = torch.isfinite(C_ref) & torch.isfinite(C_test)

    REL_TOL = 0.05
    ABS_TOL = 1e-2
    n_valid = int(base_mask.sum().item())

    if n_valid < 1024:
        snr_db = float('-inf'); snr_med_db = float('-inf'); wrong_cell_frac = 1.0
    else:
        ref_d = C_ref.double(); test_d = C_test.double()
        abs_diff = (test_d - ref_d).abs()
        abs_ref = ref_d.abs(); abs_test = test_d.abs()
        catastrophic = (abs_test > 100.0 * abs_ref + 1e-3) & (abs_diff > REL_TOL * abs_ref + ABS_TOL)
        catastrophic = catastrophic & base_mask
        wrong_cell_frac = float(catastrophic.sum().item()) / float(base_mask.sum().item())

        good_mask = base_mask & ~catastrophic
        n_good = int(good_mask.sum().item())
        if n_good < 1024:
            snr_db = float('-inf'); snr_med_db = float('-inf')
        else:
            rv = ref_d[good_mask]; tv = test_d[good_mask]
            sig = float((rv * rv).mean().item())
            err = float(((tv - rv) ** 2).mean().item())
            snr_db = 10.0 * math.log10(sig / err) if (sig > 0.0 and err > 0.0) else float('-inf')

            row_n = good_mask.sum(dim=1).clamp_min(1)
            row_sig = (ref_d ** 2 * good_mask).sum(dim=1) / row_n
            row_err = ((test_d - ref_d) ** 2 * good_mask).sum(dim=1) / row_n
            valid_rows = good_mask.any(dim=1) & (row_sig > 0) & (row_err > 0)
            if int(valid_rows.sum().item()) < 8:
                snr_med_db = float('-inf')
            else:
                row_snr = 10.0 * torch.log10(row_sig[valid_rows] / row_err[valid_rows])
                snr_med_db = float(row_snr.median().item())

    correct = ((snr_med_db >= SNR_DB_GATE)
               and (wrong_cell_frac < WRONG_CELL_GATE)
               and (finite_frac >= FINITE_GATE))

    if not correct:
        out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None, "comp": COMP,
                "status": "WRONG_OUTPUT", "kernel_finite": finite_frac,
                "snr_db": snr_db if math.isfinite(snr_db) else None,
                "snr_med_db": snr_med_db if math.isfinite(snr_med_db) else None,
                "snr_n_valid": n_valid,
                "wrong_cell_frac": round(wrong_cell_frac, 6),
                "seed": {seed}}}
        print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
        sys.exit(0)

    for _ in range(WARMUP): run()
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times.sort()
    trim = int(len(times) * TRIM)
    if trim > 0: times = times[trim:-trim]
    avg = sum(times) / len(times)
    tflops = 2.0 * M * N * K / (avg * 1e-3) / 1e12

    out = {{"M": M, "N": N, "K": K, "tflops": round(tflops, 1),
            "avg_ms": round(avg, 4), "comp": COMP, "status": "OK",
            "kernel_finite": finite_frac,
            "snr_db": round(snr_db, 2) if math.isfinite(snr_db) else None,
            "snr_med_db": round(snr_med_db, 2) if math.isfinite(snr_med_db) else None,
            "snr_n_valid": n_valid,
            "wrong_cell_frac": round(wrong_cell_frac, 6),
            "seed": {seed}}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")

except torch.cuda.OutOfMemoryError:
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": "OOM", "kernel_finite": None, "seed": {seed}}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
except Exception as e:
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": f"ERR:{{e}}"[:120], "kernel_finite": None, "seed": {seed}}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
"""


def make_aiter_runner_script(m, n, k, comp, seed, tile_M, tile_N, co_path, kname):
    return f"""\
import sys, os, math, json, importlib.util
import torch
torch.manual_seed({seed})

import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import dtypes, fp4_utils

SHIM_PATH = {AITER_SHIM_SO!r}
AITER_CO  = {co_path!r}
KERNEL_NAME = {kname!r}
M, N, K = {m}, {n}, {k}
TILE_M, TILE_N = {tile_M}, {tile_N}
COMP = {comp}
WARMUP, ITERS, TRIM_FRAC = {WARMUP}, {ITERS}, {TRIM}
SNR_DB_GATE = {SNR_THRESHOLD_DB}
WCF_GATE = {WRONG_CELL_GATE}
FINITE_GATE = {FINITE_GATE}
SCALE_GROUP_SIZE = 32

def emit(d):
    print("BENCH_JSON_START")
    print(json.dumps(d))
    print("BENCH_JSON_END")

try:
    spec = importlib.util.spec_from_file_location("R50D_aiter_shim", SHIM_PATH)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)

    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)

    dtype = dtypes.bf16
    x = torch.randn((M, K), dtype=dtype, device='cuda')
    w = torch.randn((N, K), dtype=dtype, device='cuda')

    x_packed, x_scales_shuffle = quant_func(x, shuffle=True)
    w_packed, w_scales_shuffle = quant_func(w, shuffle=True)

    wshuffle = shuffle_weight(w_packed, layout=(16, 16))

    m_pad32 = ((M + 31) // 32) * 32
    C = torch.zeros((m_pad32, N), dtype=dtype, device='cuda')

    x_scales_u8 = x_scales_shuffle.view(torch.uint8)
    w_scales_u8 = w_scales_shuffle.view(torch.uint8)
    x_packed_u8 = x_packed if x_packed.dtype == torch.uint8 else x_packed.view(torch.uint8)
    w_packed_u8 = wshuffle if wshuffle.dtype == torch.uint8 else wshuffle.view(torch.uint8)

    stream_handle = int(torch.cuda.current_stream().cuda_stream)

    def run():
        shim.launch(x_packed_u8, w_packed_u8, x_scales_u8, w_scales_u8,
                    C, M, N, K,
                    TILE_M, TILE_N,
                    AITER_CO, KERNEL_NAME,
                    1.0, 0.0,
                    stream_handle)

    C.zero_()
    run()
    torch.cuda.synchronize()
    C0 = C[:M].clone()
    finite_frac = float(torch.isfinite(C0.float()).sum().item()) / float(C0.numel())
    finite_frac = round(finite_frac, 6)

    MAX_ROWS = 1024
    m_rows = min(M, MAX_ROWS)
    n_rows = min(N, MAX_ROWS)

    _, x_scales_canon = quant_func(x, shuffle=False)
    _, w_scales_canon = quant_func(w, shuffle=False)

    x_f32 = fp4_utils.mxfp4_to_f32(x_packed)[:m_rows].float()
    w_f32 = fp4_utils.mxfp4_to_f32(w_packed)[:n_rows].float()

    x_sc_canon = x_scales_canon[:m_rows].view(torch.uint8)
    x_sc_canon = x_sc_canon.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_sc_f32 = fp4_utils.e8m0_to_f32(x_sc_canon)[:, :K]
    x_f32 = x_f32 * x_sc_f32

    w_sc_canon = w_scales_canon[:n_rows].view(torch.uint8)
    w_sc_canon = w_sc_canon.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_sc_f32 = fp4_utils.e8m0_to_f32(w_sc_canon)[:, :K]
    w_f32 = w_f32 * w_sc_f32

    C_ref = torch.mm(x_f32, w_f32.T).double()
    C_test = C0[:m_rows, :n_rows].float().double()
    base_mask = torch.isfinite(C_ref) & torch.isfinite(C_test)

    REL_TOL = 0.05
    ABS_TOL = 1e-2
    n_valid = int(base_mask.sum().item())

    if n_valid < 1024:
        snr_db = float('-inf'); snr_med_db = float('-inf'); wrong_cell_frac = 1.0
    else:
        abs_diff = (C_test - C_ref).abs()
        abs_ref = C_ref.abs(); abs_test = C_test.abs()
        catastrophic = (abs_test > 100.0 * abs_ref + 1e-3) & (abs_diff > REL_TOL * abs_ref + ABS_TOL)
        catastrophic = catastrophic & base_mask
        wrong_cell_frac = float(catastrophic.sum().item()) / float(base_mask.sum().item())

        good_mask = base_mask & ~catastrophic
        n_good = int(good_mask.sum().item())
        if n_good < 1024:
            snr_db = float('-inf'); snr_med_db = float('-inf')
        else:
            rv = C_ref[good_mask]; tv = C_test[good_mask]
            sig = float((rv * rv).mean().item())
            err = float(((tv - rv) ** 2).mean().item())
            snr_db = 10.0 * math.log10(sig / err) if (sig > 0 and err > 0) else float('-inf')

            row_n = good_mask.sum(dim=1).clamp_min(1)
            row_sig = (C_ref ** 2 * good_mask).sum(dim=1) / row_n
            row_err = ((C_test - C_ref) ** 2 * good_mask).sum(dim=1) / row_n
            valid_rows = good_mask.any(dim=1) & (row_sig > 0) & (row_err > 0)
            if int(valid_rows.sum().item()) < 8:
                snr_med_db = float('-inf')
            else:
                row_snr = 10.0 * torch.log10(row_sig[valid_rows] / row_err[valid_rows])
                snr_med_db = float(row_snr.median().item())

    correct = (snr_med_db >= SNR_DB_GATE
               and wrong_cell_frac < WCF_GATE
               and finite_frac >= FINITE_GATE)

    if not correct:
        out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None, "comp": COMP,
                "status": "WRONG_OUTPUT", "kernel_finite": finite_frac,
                "snr_db": snr_db if math.isfinite(snr_db) else None,
                "snr_med_db": snr_med_db if math.isfinite(snr_med_db) else None,
                "snr_n_valid": n_valid,
                "wrong_cell_frac": round(wrong_cell_frac, 6),
                "seed": {seed}}}
        emit(out); sys.exit(0)

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times.sort()
    trim = int(len(times) * TRIM_FRAC)
    if trim > 0: times = times[trim:-trim]
    avg = sum(times) / len(times)
    tflops = 2.0 * M * N * K / (avg * 1e-3) / 1e12

    out = {{"M": M, "N": N, "K": K, "tflops": round(tflops, 1),
            "avg_ms": round(avg, 4), "comp": COMP, "status": "OK",
            "kernel_finite": finite_frac,
            "snr_db": round(snr_db, 2) if math.isfinite(snr_db) else None,
            "snr_med_db": round(snr_med_db, 2) if math.isfinite(snr_med_db) else None,
            "snr_n_valid": n_valid,
            "wrong_cell_frac": round(wrong_cell_frac, 6),
            "seed": {seed}}}
    emit(out)
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": f"ERR:{{e}}"[:160], "kernel_finite": None,
           "seed": {seed}, "tb": tb[-400:]}}
    emit(out)
"""


def bench_one_shape(m, n, k, comp, so_path, source, gpu_id, seed):
    is_aiter = ((m, n, k) in AITER_SHAPES)
    if is_aiter:
        tile_M, tile_N, co_path, kname, _src = AITER_SHAPES[(m, n, k)]
        if not os.path.exists(AITER_SHIM_SO):
            return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                    "comp": comp, "status": "MISSING_SHIM", "kernel_finite": None,
                    "source": source, "so_path": AITER_SHIM_SO, "seed": seed}
        if not os.path.exists(co_path):
            return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                    "comp": comp, "status": "MISSING_CO", "kernel_finite": None,
                    "source": source, "so_path": co_path, "seed": seed}
        script = make_aiter_runner_script(m, n, k, comp, seed, tile_M, tile_N, co_path, kname)
    else:
        if not os.path.exists(so_path):
            return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                    "comp": comp, "status": "MISSING_SO", "kernel_finite": None,
                    "source": source, "so_path": so_path, "seed": seed}
        module_name = os.path.basename(so_path).split(".")[0]
        script = make_hk_runner_script(module_name, so_path, m, n, k, comp, seed)

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=SHAPE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "TIMEOUT", "kernel_finite": None,
                "source": source, "seed": seed}

    if r.returncode != 0:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "CRASH", "kernel_finite": None,
                "source": source, "stderr_tail": r.stderr[-300:],
                "stdout_tail": r.stdout[-300:], "seed": seed}

    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        out = json.loads(r.stdout[s:e].strip())
        out["source"] = source
        return out
    except (ValueError, json.JSONDecodeError):
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "PARSE_FAIL", "kernel_finite": None,
                "source": source, "stdout_tail": r.stdout[-400:], "seed": seed}


def run_one_pass(gpus, run_label, jobs, seed):
    print(f"R59 INTEG Bench [{run_label}] seed={seed} - GPUs: {gpus}, n_jobs={len(jobs)}")
    print("=" * 110)

    gpu_tasks = {g: [] for g in gpus}
    for i, item in enumerate(jobs):
        gpu = gpus[i % len(gpus)]
        gpu_tasks[gpu].append((i, item))

    results = [None] * len(jobs)
    lock = threading.Lock()
    t0 = time.time()
    done = [0]

    def run_gpu(gpu_id):
        for (i, (m, n, k, comp, so_path, source)) in gpu_tasks[gpu_id]:
            r = bench_one_shape(m, n, k, comp, so_path, source, gpu_id, seed)
            with lock:
                results[i] = r
                done[0] += 1
                if r["status"] == "OK":
                    ratio = r["tflops"] / comp * 100
                    flag = "WIN" if r["tflops"] >= comp else "LOSE"
                    fin = r.get("kernel_finite")
                    fin_s = f"{fin:.4f}" if fin is not None else "N/A"
                    wcf = r.get("wrong_cell_frac")
                    wcf_s = f"{wcf:.4f}" if wcf is not None else "N/A"
                    print(f"  [{done[0]:>3}/{len(jobs)}] {m:>6}x{n:>6}x{k:>6} [{source}] "
                          f"{r['tflops']:>7.1f} vs {comp:>7.1f} ({ratio:>5.1f}%) "
                          f"fin={fin_s} wcf={wcf_s} {flag} GPU{gpu_id}",
                          flush=True)
                else:
                    fin = r.get("kernel_finite")
                    fin_s = f"{fin:.4f}" if fin is not None else "N/A"
                    wcf = r.get("wrong_cell_frac")
                    wcf_s = f"{wcf:.4f}" if wcf is not None else "N/A"
                    print(f"  [{done[0]:>3}/{len(jobs)}] {m:>6}x{n:>6}x{k:>6} [{source}] "
                          f"{r['status']} fin={fin_s} wcf={wcf_s} GPU{gpu_id}",
                          flush=True)

    threads = []
    for g in gpus:
        t = threading.Thread(target=run_gpu, args=(g,))
        t.start(); threads.append(t)
    for t in threads: t.join()

    elapsed = time.time() - t0
    return results, elapsed


def aggregate_consensus(per_run_results, n_runs, comp):
    """Compute n-run consensus per shape with R50 10-run @ 80% promotion gate."""
    oks = [r for r in per_run_results if r["status"] == "OK"]
    n_OK = len(oks)
    wcfs = [r.get("wrong_cell_frac") for r in per_run_results if r.get("wrong_cell_frac") is not None]
    fins = [r.get("kernel_finite") for r in per_run_results if r.get("kernel_finite") is not None]

    def safe_max(xs): return max(xs) if xs else None
    def safe_mean(xs): return sum(xs) / len(xs) if xs else None
    def safe_min(xs): return min(xs) if xs else None
    def safe_std(xs):
        if not xs or len(xs) < 2: return 0.0
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    if oks:
        tflops_list = sorted([r["tflops"] for r in oks])
        tflops_p50 = tflops_list[len(tflops_list) // 2]
    else:
        tflops_p50 = None

    wcf_max = safe_max(wcfs)
    wcf_mean = safe_mean(wcfs)
    wcf_std = safe_std(wcfs)
    fin_min = safe_min(fins)

    OK_THRESHOLD = max(1, int(math.ceil(n_runs * 0.8)))
    verified_correct = (
        n_OK >= OK_THRESHOLD
        and wcf_max is not None and wcf_max < WRONG_CELL_GATE
        and wcf_std < 0.01
        and fin_min is not None and fin_min >= FINITE_GATE
    )

    if n_OK == n_runs:
        verdict = f"PASS_{n_OK}/{n_runs}"
    elif n_OK >= OK_THRESHOLD:
        verdict = f"PASS_{n_OK}/{n_runs}"
    elif n_OK >= 1:
        verdict = f"FLAKE_{n_OK}/{n_runs}"
    else:
        wrong = sum(1 for r in per_run_results if r["status"] == "WRONG_OUTPUT")
        if wrong > 0:
            verdict = f"WRONG_{wrong}/{n_runs}"
        else:
            statuses = [r["status"] for r in per_run_results]
            verdict = f"FAIL_{statuses[0]}"

    return {
        "tflops_p50": tflops_p50,
        "wcf_max": wcf_max,
        "wcf_mean": wcf_mean,
        "wcf_std": wcf_std,
        "fin_min": fin_min,
        "n_OK_5": n_OK,
        "verdict": verdict,
        "verified_correct": verified_correct,
        "ok_threshold": OK_THRESHOLD,
        "comp": comp,
        "pct_comp": (tflops_p50 / comp * 100) if (tflops_p50 is not None and comp > 0) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--mode", choices=["smoke1", "10run"], default=None,
                    help="Convenience selector (R59 task spec). smoke1 -> --runs 1 + smoke output paths; "
                         "10run -> --runs 10 + 10run output paths. Either --mode or explicit --runs may be used.")
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--manifest", default="R59_INTEGRATION_MANIFEST.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="optional override of RANDOM_SEEDS; len must >= --runs")
    args = ap.parse_args()

    # Resolve --mode -> defaults
    if args.mode == "smoke1":
        runs = args.runs if args.runs is not None else 1
        out = args.out if args.out is not None else "R59_INTEGRATION_SMOKE1.json"
        log = args.log if args.log is not None else "R59_INTEGRATION_SMOKE1.log"
    elif args.mode == "10run":
        runs = args.runs if args.runs is not None else 10
        out = args.out if args.out is not None else "R59_INTEGRATION_10RUN.json"
        log = args.log if args.log is not None else "R59_INTEGRATION_10RUN.log"
    else:
        runs = args.runs if args.runs is not None else 10
        out = args.out if args.out is not None else "R59_INTEGRATION_10RUN.json"
        log = args.log if args.log is not None else "R59_INTEGRATION_10RUN.log"

    gpus = [int(g) for g in args.gpus.split(",")]
    seeds = args.seeds if args.seeds else RANDOM_SEEDS
    assert runs <= len(seeds), f"runs={runs} exceeds {len(seeds)} seeds"

    manifest_path = os.path.join(SCRIPT_DIR, args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)

    shapes_to_so = manifest["shapes_to_so_path"]
    shapes_to_source = manifest["shapes_to_source"]

    jobs = []
    for (m, n, k, comp) in ALL_SHAPES:
        key = f"{m}x{n}x{k}"
        if key not in shapes_to_so:
            print(f"WARNING: shape {key} missing from manifest", file=sys.stderr)
            continue
        jobs.append((m, n, k, comp, shapes_to_so[key], shapes_to_source[key]))

    log_path = os.path.join(SCRIPT_DIR, log)
    log_lines = []

    def logmsg(msg):
        log_lines.append(msg)
        print(msg, flush=True)

    logmsg(f"R59 INTEGRATION bench: {len(jobs)} shapes, runs={runs}, GPUs={gpus}")
    logmsg(f"  warmup={WARMUP}, iters={ITERS}, trim={TRIM}, seeds={seeds[:runs]}")
    logmsg(f"  gates: snr_med>={SNR_THRESHOLD_DB}dB, wcf<{WRONG_CELL_GATE}, fin>={FINITE_GATE}")
    OK_TH = max(1, int(math.ceil(runs * 0.8)))
    logmsg(f"  promote rule: n_OK>={OK_TH}/{runs} AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min>=0.97")

    all_runs = []
    for ri in range(runs):
        seed = seeds[ri]
        results, elapsed = run_one_pass(gpus, f"run{ri+1}/{runs}", jobs, seed)
        all_runs.append({"run": ri + 1, "seed": seed,
                         "elapsed_minutes": round(elapsed / 60, 1),
                         "results": results})
        logmsg(f"\n[run{ri+1}/{runs}] seed={seed} elapsed={elapsed/60:.1f} min")
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))

    consensus = {}
    for i, (m, n, k, comp, so_path, source) in enumerate(jobs):
        key = f"{m}x{n}x{k}"
        per_shape_runs = [run["results"][i] for run in all_runs]
        agg = aggregate_consensus(per_shape_runs, runs, comp)
        agg["M"], agg["N"], agg["K"] = m, n, k
        agg["source"] = source
        agg["so_path"] = so_path
        agg["per_run"] = per_shape_runs
        consensus[key] = agg

    out_path = os.path.join(SCRIPT_DIR, out)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": "R59_INTEGRATION",
            "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
            "snr_threshold_db": SNR_THRESHOLD_DB,
            "wrong_cell_gate": WRONG_CELL_GATE,
            "finite_gate": FINITE_GATE,
            "random_seeds": seeds[:runs],
            "n_shapes": len(jobs),
            "n_runs": runs,
            "ok_threshold": OK_TH,
            "gpus": gpus,
            "manifest": args.manifest,
            "consensus": consensus,
            "per_run_summary": [
                {"run": r["run"], "seed": r["seed"], "elapsed_minutes": r["elapsed_minutes"]}
                for r in all_runs
            ],
        }, f, indent=2)

    n_verified = sum(1 for v in consensus.values() if v["verified_correct"])
    n_win = sum(1 for v in consensus.values()
                if v["pct_comp"] is not None and v["pct_comp"] >= 100.0)

    logmsg(f"\n=== R59 INTEGRATION SUMMARY ===")
    logmsg(f"Verified-correct (n_OK>={OK_TH}/{runs}): {n_verified}/{len(jobs)}")
    logmsg(f"WIN (>=100% comp): {n_win}/{len(jobs)}")
    logmsg(f"Output: {out_path}")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
