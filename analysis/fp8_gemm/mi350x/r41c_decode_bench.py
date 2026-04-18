"""R41 Dev C — decode-shape coverage bench.

Loads a specified .so module (positional arg #1), runs a single layout
benchmark for the requested shape (M, N, K) using preshuffle-quant V2
when supported, otherwise PQ V1 fallback. Uses the same setup as
test_mxfp8_python.py / test_python.py — but tolerant of small M.

Usage:
    python3 r41c_decode_bench.py <module_name> <kind> <layout> <M> <N> <K> [--iters=N]
        kind: mxfp8 | fp8
        layout: rcr | rrr | crr

Environment:
    HIP_VISIBLE_DEVICES (caller sets)
    DISPATCH_TRACE=1    (forwarded to MXFP8_DISPATCH_TRACE)
    WARMUP, ITERS       (defaults: warmup=50 iters=100)
    PREHEAT_S           (preheat seconds with float16 GEMM, default 4.0)

Output (single line, easy parse):
    R41C_RESULT module=<m> kind=<k> layout=<L> shape=MxNxK avg_ms=<v> tflops=<v> snr_db=<v|NA>
"""
import math
import os
import random
import sys
import time

import torch

torch.manual_seed(0)
random.seed(0)


def parse_args():
    if len(sys.argv) < 7:
        print("usage: r41c_decode_bench.py <module> <kind> <layout> M N K", file=sys.stderr)
        sys.exit(2)
    mod_name = sys.argv[1]
    kind = sys.argv[2].lower()
    layout = sys.argv[3].lower()
    M = int(sys.argv[4]); N = int(sys.argv[5]); K = int(sys.argv[6])
    return mod_name, kind, layout, M, N, K


def preheat(seconds=4.0):
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        c = a @ b
    torch.cuda.synchronize()
    del a, b, c
    torch.cuda.empty_cache()


def gen_fp8(rows, cols, scale=0.05):
    x = torch.zeros(rows, cols, dtype=torch.float32, device="cuda")
    x[:] = torch.randn(rows, cols, dtype=torch.float32, device="cuda") * scale
    return x.to(torch.float8_e4m3fn)


def gen_scale_exp(rows, k_blocks):
    return torch.randint(low=-2, high=3, size=(rows, k_blocks), dtype=torch.int8, device="cuda")


def encode_scale_raw(scale_exp):
    raw = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(scale_exp == -128, torch.full_like(raw, 0xFF, dtype=torch.uint8), raw)


def preshuffle_v1(scale_exp):
    rows, k_blocks_local = scale_exp.shape
    padded_rows = math.ceil(rows / 32) * 32
    padded_k_blocks = math.ceil(k_blocks_local / 8) * 8
    raw = torch.full((padded_rows, padded_k_blocks), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :k_blocks_local] = encode_scale_raw(scale_exp)
    shuffled = raw.view(padded_rows // 32, 2, 16, padded_k_blocks // 8, 2, 4, 1)
    shuffled = shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return shuffled.view(padded_rows // 32, padded_k_blocks * 32)


def preshuffle_v2_a(scale_exp, blk=256, hb=128, rbm=64, warps_m=2):
    pack_count = 4
    rows, k_blocks_local = scale_exp.shape
    padded_rows = math.ceil(rows / blk) * blk
    padded_k_blocks = math.ceil(k_blocks_local / 8) * 8
    num_ctiles = padded_rows // blk
    num_slabs = num_ctiles * warps_m
    rgs_per_ctile = blk // 32
    pack_a = rbm // 32
    raw = torch.full((padded_rows, padded_k_blocks), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :k_blocks_local] = encode_scale_raw(scale_exp)
    perm_rows = torch.empty_like(raw)
    rg_view = raw.view(num_ctiles, rgs_per_ctile, 32, padded_k_blocks)
    perm_view = perm_rows.view(num_ctiles, num_slabs // num_ctiles, pack_count, 32, padded_k_blocks)
    for wm in range(warps_m):
        rg_base = wm * (rbm // 32)
        rg_hi = (hb // 32) + rg_base
        for pidx in range(pack_a):
            perm_view[:, wm, 2 * pidx, :, :] = rg_view[:, rg_base + pidx, :, :]
            perm_view[:, wm, 2 * pidx + 1, :, :] = rg_view[:, rg_hi + pidx, :, :]
    kp_count = padded_k_blocks // 8
    rows_view = perm_rows.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    shuffled = rows_view.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return shuffled.view(num_slabs, pack_count * 32 * padded_k_blocks)


def preshuffle_v2_b(scale_exp, blk=256, hb=128, rbn=32, warps_n=4):
    pack_count = 2
    rows, k_blocks_local = scale_exp.shape
    padded_rows = math.ceil(rows / blk) * blk
    padded_k_blocks = math.ceil(k_blocks_local / 8) * 8
    num_ctiles = padded_rows // blk
    num_slabs = num_ctiles * warps_n
    rgs_per_ctile = blk // 32
    raw = torch.full((padded_rows, padded_k_blocks), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :k_blocks_local] = encode_scale_raw(scale_exp)
    perm_rows = torch.empty_like(raw)
    rg_view = raw.view(num_ctiles, rgs_per_ctile, 32, padded_k_blocks)
    perm_view = perm_rows.view(num_ctiles, warps_n, pack_count, 32, padded_k_blocks)
    rbn_rg = rbn // 32
    rg_hi_offset = hb // 32
    for wn in range(warps_n):
        rg_base = wn * rbn_rg
        perm_view[:, wn, 0, :, :] = rg_view[:, rg_base, :, :]
        perm_view[:, wn, 1, :, :] = rg_view[:, rg_base + rg_hi_offset, :, :]
    kp_count = padded_k_blocks // 8
    rows_view = perm_rows.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    shuffled = rows_view.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return shuffled.view(num_slabs, pack_count * 32 * padded_k_blocks)


def bench(fn, output, warmup, iters):
    for _ in range(warmup):
        output.zero_()
        fn()
    timings = []
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        output.zero_()
        torch.cuda.synchronize()
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        timings.append(se.elapsed_time(ee))
    return timings


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    mod_name, kind, layout, M, N, K = parse_args()
    warmup = int(os.environ.get("WARMUP", "50"))
    iters = int(os.environ.get("ITERS", "100"))
    preheat_s = float(os.environ.get("PREHEAT_S", "4.0"))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
    mod = __import__(mod_name)

    flops_ref = 2 * M * N * K
    k_blocks = (K + 31) // 32
    preheat(preheat_s)

    snr_db = "NA"

    if kind == "mxfp8":
        # MXFP8 V2 fastpaths gate on g.m == M_DIM but use grid = (g.m / BLK), where BLK=256.
        # For decode shapes with M < 256 the V2 path launches a 0-block grid → HIP crash.
        # Production inference would call dispatch() / dispatch_pq() (legacy V1), which
        # falls through to the tail kernel with TAIL_BLOCK_M=16. We use V1 PQ here as the
        # representative MXFP8 decode path. (The "V1-LEGACY-FALLBACK" / "V1-PQ-DEFAULT"
        # trace will fire.)
        # When the .so was built with M_DIM matching this M, the V2 fastpath fires
        # and (for M < BLK=256) crashes with hipErrorInvalidConfiguration. Force V1
        # in that case. Otherwise call V2 entry point: with mismatched M_DIM the
        # dispatcher's V2 predicate fails and falls through to V1-LEGACY-FALLBACK.
        force_v1 = os.environ.get("R41C_FORCE_V1", "0") == "1"
        if layout == "rcr":
            A = gen_fp8(M, K); B = gen_fp8(N, K)
            A_se = gen_scale_exp(M, k_blocks); B_se = gen_scale_exp(N, k_blocks)
            if force_v1:
                A_s = preshuffle_v1(A_se); B_s = preshuffle_v1(B_se)
                C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                fn = lambda: mod.gemm_rcr_pq(A, B, A_s, B_s, C)
            else:
                A_s = preshuffle_v2_a(A_se); B_s = preshuffle_v2_b(B_se)
                C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                fn = lambda: mod.gemm_rcr_pq_v2(A, B, A_s, B_s, C)
        elif layout == "rrr":
            A = gen_fp8(M, K); B = gen_fp8(K, N)
            A_se = gen_scale_exp(M, k_blocks); B_se = gen_scale_exp(N, k_blocks)
            if force_v1:
                A_s = preshuffle_v1(A_se); B_s = preshuffle_v1(B_se)
                C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                fn = lambda: mod.gemm_rrr_pq(A, B, A_s, B_s, C)
            else:
                A_s = preshuffle_v2_a(A_se); B_s = preshuffle_v2_b(B_se)
                C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                fn = lambda: mod.gemm_rrr_pq_v2(A, B, A_s, B_s, C)
        elif layout == "crr":
            A = gen_fp8(K, M); B = gen_fp8(K, N)
            A_se = gen_scale_exp(M, k_blocks); B_se = gen_scale_exp(N, k_blocks)
            if force_v1:
                A_s = preshuffle_v1(A_se); B_s = preshuffle_v1(B_se)
                C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                fn = lambda: mod.gemm_crr_pq(A, B, A_s, B_s, C)
            else:
                A_s = preshuffle_v2_a(A_se); B_s = preshuffle_v2_b(B_se)
                C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                fn = lambda: mod.gemm_crr_pq_v2(A, B, A_s, B_s, C)
        else:
            print(f"ERROR: unknown layout {layout}", file=sys.stderr); sys.exit(2)
    elif kind == "fp8":
        scale = 1.0
        if layout == "rcr":
            A = gen_fp8(M, K); B = gen_fp8(N, K)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            fn = lambda: mod.gemm_rcr(A, B, C, scale)
        elif layout == "rrr":
            A = gen_fp8(M, K); B = gen_fp8(K, N)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            fn = lambda: mod.gemm_rrr(A, B, C, scale)
        elif layout == "crr":
            A = gen_fp8(K, M); B = gen_fp8(K, N)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            fn = lambda: mod.gemm_crr(A, B, C, scale)
        else:
            print(f"ERROR: unknown layout {layout}", file=sys.stderr); sys.exit(2)
    else:
        print(f"ERROR: unknown kind {kind}", file=sys.stderr); sys.exit(2)

    timings = bench(fn, C, warmup=warmup, iters=iters)
    avg_ms = median(timings)
    tflops = flops_ref / (avg_ms * 1e9)

    print(f"R41C_RESULT module={mod_name} kind={kind} layout={layout} "
          f"shape={M}x{N}x{K} avg_ms={avg_ms:.6f} tflops={tflops:.4f} snr_db={snr_db}")


if __name__ == "__main__":
    main()
