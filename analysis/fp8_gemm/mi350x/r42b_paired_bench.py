"""R42 Dev B — paired bench for SMALLM-B32-TAIL vs baseline + FP8.

Runs N_PAIRS interleaved measurements: (baseline, smallm, fp8) per pair.
Computes Welch t-statistic between smallm and baseline; reports MXFP8/FP8 ratio.

Usage:
  HIP_VISIBLE_DEVICES=N python3 r42b_paired_bench.py <baseline_mod> <smallm_mod> <fp8_mod> M N K [N_PAIRS]
"""
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import r41c_decode_bench as bench

torch.manual_seed(0)


def setup(M, N, K):
    k_blocks = (K + 31) // 32
    A = bench.gen_fp8(M, K)
    B = bench.gen_fp8(N, K)
    A_se = bench.gen_scale_exp(M, k_blocks)
    B_se = bench.gen_scale_exp(N, k_blocks)
    A_s = bench.preshuffle_v1(A_se)
    B_s = bench.preshuffle_v1(B_se)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    A_fp8 = A
    B_fp8 = B
    return A, B, A_s, B_s, A_fp8, B_fp8, C


def time_one(fn, C, iters):
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        C.zero_()
        torch.cuda.synchronize()
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    return statistics.median(times)


def welch_t(a, b):
    if not a or not b or len(a) < 2 or len(b) < 2:
        return float("nan")
    ma = statistics.mean(a); mb = statistics.mean(b)
    va = statistics.variance(a); vb = statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se <= 0:
        return float("inf")
    return (ma - mb) / se


def main():
    base_mod = sys.argv[1]
    smallm_mod = sys.argv[2]
    fp8_mod = sys.argv[3]
    M = int(sys.argv[4]); N = int(sys.argv[5]); K = int(sys.argv[6])
    n_pairs = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    iters = int(os.environ.get("ITERS", "100"))
    preheat_s = float(os.environ.get("PREHEAT_S", "60"))

    bench.preheat(preheat_s)

    base = __import__(base_mod)
    sm = __import__(smallm_mod)
    fp = __import__(fp8_mod)

    A, B, A_s, B_s, A_fp, B_fp, C = setup(M, N, K)
    fn_base = lambda: base.gemm_rcr_pq(A, B, A_s, B_s, C)
    fn_sm = lambda: sm.gemm_rcr_pq(A, B, A_s, B_s, C)
    fn_fp8 = lambda: fp.gemm_rcr(A_fp, B_fp, C, 1.0)

    # Warmup each
    for _ in range(50):
        C.zero_(); fn_base(); fn_sm(); fn_fp8()
    torch.cuda.synchronize()

    base_tf, sm_tf, fp_tf = [], [], []
    flops = 2 * M * N * K
    for i in range(n_pairs):
        # interleave per pair
        t_base = time_one(fn_base, C, iters)
        t_sm = time_one(fn_sm, C, iters)
        t_fp = time_one(fn_fp8, C, iters)
        base_tf.append(flops / (t_base * 1e9))
        sm_tf.append(flops / (t_sm * 1e9))
        fp_tf.append(flops / (t_fp * 1e9))
        print(f"  pair {i+1}/{n_pairs}: base={base_tf[-1]:.4f} smallm={sm_tf[-1]:.4f} fp8={fp_tf[-1]:.4f}")

    base_med = statistics.median(base_tf)
    sm_med = statistics.median(sm_tf)
    fp_med = statistics.median(fp_tf)

    delta_pct = 100.0 * (sm_med - base_med) / base_med
    ratio_sm_fp = 100.0 * sm_med / fp_med
    ratio_base_fp = 100.0 * base_med / fp_med
    t_stat = welch_t(sm_tf, base_tf)

    print(f"\nR42B_PAIRED M={M} N={N} K={K} n_pairs={n_pairs}")
    print(f"  base_med = {base_med:.4f} TF")
    print(f"  smallm_med = {sm_med:.4f} TF")
    print(f"  fp8_med = {fp_med:.4f} TF")
    print(f"  delta_pct (smallm vs base) = {delta_pct:+.2f}%")
    print(f"  welch_t (smallm vs base) = {t_stat:+.2f}")
    print(f"  ratio smallm/fp8 = {ratio_sm_fp:.2f}%")
    print(f"  ratio base/fp8 = {ratio_base_fp:.2f}%")
    if ratio_sm_fp >= 95.0:
        print(f"R42B_VERDICT PASS_95 (smallm/fp8 = {ratio_sm_fp:.2f}% >= 95%)")
    else:
        print(f"R42B_VERDICT FAIL_95 (smallm/fp8 = {ratio_sm_fp:.2f}% < 95%)")
    if delta_pct >= 25.0:
        print(f"R42B_VERDICT PASS_DELTA (delta = {delta_pct:+.2f}% >= +25%)")
    else:
        print(f"R42B_VERDICT FAIL_DELTA (delta = {delta_pct:+.2f}% < +25%)")


if __name__ == "__main__":
    main()
