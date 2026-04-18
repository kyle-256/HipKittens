"""R44 Dev A — M=2..16 small-batch decode MXFP8 fastpath bench.

Mirrors r43b_decode_m1_rrr_crr_bench.py's interface.

Usage:
    python3 r44a_decode_m2_16_bench.py <module> <kind> <impl> <layout> M N K
        kind:   mxfp8 | fp8
        impl:   legacy_v1pq | decode_m2_16 | fp8_pertensor
        layout: rcr | rrr | crr
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
    if len(sys.argv) < 8:
        print("usage: r44a_decode_m2_16_bench.py <module> <kind> <impl> <layout> M N K",
              file=sys.stderr)
        sys.exit(2)
    mod_name = sys.argv[1]
    kind = sys.argv[2].lower()
    impl = sys.argv[3].lower()
    layout = sys.argv[4].lower()
    M = int(sys.argv[5]); N = int(sys.argv[6]); K = int(sys.argv[7])
    return mod_name, kind, impl, layout, M, N, K


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
    x = torch.randn(rows, cols, dtype=torch.float32, device="cuda") * scale
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
    raw = torch.full((padded_rows, padded_k_blocks), 0x7F, dtype=torch.uint8,
                     device=scale_exp.device)
    raw[:rows, :k_blocks_local] = encode_scale_raw(scale_exp)
    shuffled = raw.view(padded_rows // 32, 2, 16, padded_k_blocks // 8, 2, 4, 1)
    shuffled = shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return shuffled.view(padded_rows // 32, padded_k_blocks * 32)


def reference_mxfp8(A, B, A_se, B_se, layout):
    Af = A.to(torch.float32)
    Bf = B.to(torch.float32)
    A_sf = (2.0 ** A_se.to(torch.float32))
    B_sf = (2.0 ** B_se.to(torch.float32))
    if layout == "rrr":
        K = Af.shape[1]
        A_sf_full = A_sf.repeat_interleave(32, dim=1)[:, :K]
        B_sf_full = B_sf.repeat_interleave(32, dim=1)[:, :K]
        A_scaled = Af * A_sf_full
        B_scaled = Bf * B_sf_full.transpose(0, 1)
        C_ref = A_scaled @ B_scaled
    elif layout == "crr":
        K = Af.shape[0]
        A_sf_full = A_sf.repeat_interleave(32, dim=1)[:, :K]
        B_sf_full = B_sf.repeat_interleave(32, dim=1)[:, :K]
        A_scaled = Af * A_sf_full.transpose(0, 1)
        B_scaled = Bf * B_sf_full.transpose(0, 1)
        C_ref = A_scaled.transpose(0, 1) @ B_scaled
    else:  # rcr
        K = Af.shape[1]
        A_sf_full = A_sf.repeat_interleave(32, dim=1)[:, :K]
        B_sf_full = B_sf.repeat_interleave(32, dim=1)[:, :K]
        A_scaled = Af * A_sf_full
        B_scaled = Bf * B_sf_full
        C_ref = A_scaled @ B_scaled.transpose(0, 1)
    return C_ref.to(torch.bfloat16)


def reference_fp8(A, B, layout, scale=1.0):
    Af = A.to(torch.float32)
    Bf = B.to(torch.float32)
    if layout == "rrr":
        C_ref = (Af @ Bf) * scale
    elif layout == "crr":
        C_ref = (Af.transpose(0, 1) @ Bf) * scale
    else:
        C_ref = (Af @ Bf.transpose(0, 1)) * scale
    return C_ref.to(torch.bfloat16)


def snr_db(ref, got):
    diff = (ref.to(torch.float32) - got.to(torch.float32))
    sig = ref.to(torch.float32).pow(2).mean().item()
    noise = diff.pow(2).mean().item()
    if noise <= 0:
        return float("inf")
    return 10.0 * math.log10(sig / max(noise, 1e-30))


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
    mod_name, kind, impl, layout, M, N, K = parse_args()
    warmup = int(os.environ.get("WARMUP", "30"))
    iters = int(os.environ.get("ITERS", "100"))
    preheat_s = float(os.environ.get("PREHEAT_S", "30.0"))
    check_snr = os.environ.get("CHECK_SNR", "1") == "1"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
    mod = __import__(mod_name)

    flops_ref = 2 * M * N * K
    k_blocks = (K + 31) // 32
    preheat(preheat_s)

    snr_value = "NA"

    if kind == "mxfp8":
        if layout == "rrr":
            A = gen_fp8(M, K); B = gen_fp8(K, N)
            A_se = gen_scale_exp(M, k_blocks); B_se = gen_scale_exp(N, k_blocks)
            A_s = preshuffle_v1(A_se); B_s = preshuffle_v1(B_se)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            fn = lambda: mod.gemm_rrr_pq(A, B, A_s, B_s, C)
        elif layout == "crr":
            A = gen_fp8(K, M); B = gen_fp8(K, N)
            A_se = gen_scale_exp(M, k_blocks); B_se = gen_scale_exp(N, k_blocks)
            A_s = preshuffle_v1(A_se); B_s = preshuffle_v1(B_se)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            fn = lambda: mod.gemm_crr_pq(A, B, A_s, B_s, C)
        else:  # rcr
            A = gen_fp8(M, K); B = gen_fp8(N, K)
            A_se = gen_scale_exp(M, k_blocks); B_se = gen_scale_exp(N, k_blocks)
            A_s = preshuffle_v1(A_se); B_s = preshuffle_v1(B_se)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            fn = lambda: mod.gemm_rcr_pq(A, B, A_s, B_s, C)

        if check_snr:
            fn(); torch.cuda.synchronize()
            try:
                C_ref = reference_mxfp8(A, B, A_se, B_se, layout)
                snr_value = f"{snr_db(C_ref, C):.2f}"
            except Exception as e:
                snr_value = f"ERR({e})"

    elif kind == "fp8":
        scale = 1.0
        if layout == "rrr":
            A = gen_fp8(M, K); B = gen_fp8(K, N)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            fn = lambda: mod.gemm_rrr(A, B, C, scale)
        elif layout == "crr":
            A = gen_fp8(K, M); B = gen_fp8(K, N)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            fn = lambda: mod.gemm_crr(A, B, C, scale)
        else:
            A = gen_fp8(M, K); B = gen_fp8(N, K)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            fn = lambda: mod.gemm_rcr(A, B, C, scale)

        if check_snr:
            fn(); torch.cuda.synchronize()
            try:
                C_ref = reference_fp8(A, B, layout, scale)
                snr_value = f"{snr_db(C_ref, C):.2f}"
            except Exception as e:
                snr_value = f"ERR({e})"
    else:
        print(f"ERROR: kind {kind}", file=sys.stderr); sys.exit(2)

    timings = bench(fn, C, warmup=warmup, iters=iters)
    avg_ms = median(timings)
    tflops = flops_ref / (avg_ms * 1e9)

    print(f"R44A_RESULT module={mod_name} kind={kind} impl={impl} layout={layout} "
          f"shape={M}x{N}x{K} avg_ms={avg_ms:.6f} tflops={tflops:.4f} snr_db={snr_value}")


if __name__ == "__main__":
    main()
