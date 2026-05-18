#!/usr/bin/env python3
"""Bench Primus-Turbo Triton blockwise FP8 GEMM across all 18 dense target
shapes × 3 sections (fwd / dgrad / wgrad → NT / NN / TN). Used as the live
SOTA reference for HK comparisons.

The competitor TFLOPS listed in `_shapes_target.py` are kernel-only
(quant excluded from the timer). This script does the same: pre-quantize
inputs once outside the timer, time only the matmul.

Stdout: markdown table (model, op, M, N, K, section, listed, triton, ratio).
JSON cache at scripts/.bench_blockwise_triton_target_shapes_cache.json.

Env:
  PRIMUS_TURBO_PATH  default /wekafs/kyle/Primus-Turbo
  BENCH_WARMUP       default 20
  BENCH_ITERS        default 50
  BENCH_ONLY         comma list of section names (default fwd,dgrad,wgrad)
  BENCH_LIMIT        if set, stop after N shapes (debug)
  BENCH_RESUME       if 1, skip shapes already in cache
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _shapes_target import SHAPES, SECTIONS, LAYOUT_BY_SECTION, TARGET_MULTIPLIER

PT_PATH = os.environ.get("PRIMUS_TURBO_PATH")
if not PT_PATH:
    sys.exit("ERROR: set PRIMUS_TURBO_PATH=/path/to/Primus-Turbo before running "
             "(needed to import primus_turbo.pytorch.kernels for the Triton baseline)")
sys.path.insert(0, PT_PATH)

CACHE = os.path.join(HERE, ".bench_blockwise_triton_target_shapes_cache.json")

WARMUP = int(os.environ.get("BENCH_WARMUP", "20"))
ITERS  = int(os.environ.get("BENCH_ITERS",  "50"))
ONLY   = set((os.environ.get("BENCH_ONLY", ",".join(SECTIONS))).split(","))
LIMIT  = int(os.environ.get("BENCH_LIMIT", "0"))
RESUME = int(os.environ.get("BENCH_RESUME", "0"))


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


import torch
from primus_turbo.triton.gemm.gemm_fp8_kernel import (
    gemm_fp8_blockwise_triton_kernel,
    gemm_fp8_blockwise_wgrad_kc_triton_kernel,
)

torch.manual_seed(0)

# ────────────────────────────────────────────────────────────────────────────
# Methodology note:
#   fwd / dgrad  → raw call to gemm_fp8_blockwise_triton_kernel (NT, NN)
#                  Same kernel as Primus-Turbo's FP8GemmBlockFunction uses
#                  internally via gemm_fp8_impl, but bypasses the
#                  @torch.library.custom_op dispatch layer (saves 20-100 µs
#                  per call). Numbers will be SLIGHTLY HIGHER than the
#                  through-autograd `listed` measurement on small shapes,
#                  but closer to rocprof kernel-only ground truth.
#   wgrad        → gemm_fp8_blockwise_wgrad_kc_triton_kernel (K-contig path)
#                  Matches FP8GemmBlockFunction.backward — A and grad come
#                  from a single-pass quant returning col-T FP8 with the
#                  reduction dim contiguous. Strictly better than the
#                  generic TN path (gemm_fp8_blockwise_triton_kernel with
#                  trans_a=True) on large-K shapes (10-19% on K=7168).
# ────────────────────────────────────────────────────────────────────────────


def gen_fp8(rows: int, cols: int) -> torch.Tensor:
    return (torch.randn(rows, cols, device="cuda") * 0.1).to(torch.float8_e4m3fnuz)


def make_inputs(M: int, N: int, K: int, section: str):
    """Build (a, a_scale, b, b_scale, trans_a, trans_b) for fwd/dgrad.
    Scale layout matches gemm_fp8_blockwise_triton_kernel signature.
    NOTE: wgrad uses a different kernel and is constructed in bench_one."""
    Kb = (K + 127) // 128
    Mb = (M + 127) // 128
    Nb = (N + 127) // 128

    if section == "fwd":              # NT (RCR)  C[M,N] = A[M,K] @ B[N,K]^T
        a = gen_fp8(M, K)
        b = gen_fp8(N, K)
        a_scale = torch.rand(M,  Kb, device="cuda") * 0.5 + 0.5
        b_scale = torch.rand(Nb, Kb, device="cuda") * 0.5 + 0.5
        return a, a_scale, b, b_scale, False, True

    if section == "dgrad":            # NN (RRR)  C[M,K] = A[M,N] @ B[N,K]
        a = gen_fp8(M, K)             # [M,K] post-quant dY
        b = gen_fp8(K, N)             # [K,N] post-quant W
        a_scale = torch.rand(M,  Kb, device="cuda") * 0.5 + 0.5
        b_scale = torch.rand(Nb, Kb, device="cuda") * 0.5 + 0.5
        return a, a_scale, b, b_scale, False, False

    raise ValueError(f"section {section!r} handled separately")


def make_kc_wgrad_inputs(M: int, N: int, K: int):
    """K-contig wgrad: C[K, N] = A_T[K, M] @ grad_T[N, M]^T (effectively
    A^T @ grad in math, with M as the contraction).

    Inputs match what quant_fp8_blockwise_with_xpose_impl produces in
    Primus-Turbo's FP8GemmBlockFunction.backward — column-transposed FP8
    with the contraction dim (M) as the inner / contiguous axis.

    Per the kernel signature:
      a_T:      fp8 [K_out, M_red]   = [K, M] in our notation
      a_scale:  fp32 [M_red/128, K_out] = [Mb, K] (Mb-major)
      grad_T:   fp8 [N_out, M_red]   = [N, M]
      grad_scale: fp32 [M_red/128, N_out] = [Mb, N]
    """
    Mb = (M + 127) // 128
    a_T        = gen_fp8(K, M)
    grad_T     = gen_fp8(N, M)
    a_scale    = torch.rand(Mb, K, device="cuda") * 0.5 + 0.5
    grad_scale = torch.rand(Mb, N, device="cuda") * 0.5 + 0.5
    return a_T, a_scale, grad_T, grad_scale


def bench_one(M: int, N: int, K: int, section: str) -> tuple[float, float]:
    if section == "wgrad":
        a_T, a_s, g_T, g_s = make_kc_wgrad_inputs(M, N, K)
        def run():
            return gemm_fp8_blockwise_wgrad_kc_triton_kernel(
                a_T, a_s, g_T, g_s,
                out_dtype=torch.bfloat16, trans_c=False,
            )
    else:
        a, a_s, b, b_s, ta, tb = make_inputs(M, N, K, section)
        def run():
            return gemm_fp8_blockwise_triton_kernel(
                a, a_s, b, b_s,
                trans_a=ta, trans_b=tb, out_dtype=torch.bfloat16, trans_c=False,
            )

    # Triton autotune warmup
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    for _ in range(WARMUP):
        run()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))   # ms

    avg_ms = sum(times) / len(times)
    flops = 2 * M * N * K
    tflops = flops / (avg_ms * 1e9)
    return avg_ms, tflops


def main() -> None:
    cache: dict = {}
    if RESUME and os.path.exists(CACHE):
        try: cache = json.load(open(CACHE))
        except Exception: cache = {}

    shapes_iter = SHAPES if not LIMIT else SHAPES[:LIMIT]

    print(f"| {'Model':<18} {'Op':<7} {'M':>5} {'N':>5} {'K':>5} | "
          f"{'sec':<5} {'listed':>7} {'triton':>7} {'ratio':>6} {'ms':>6} |")
    print("|" + "-"*47 + "|" + "-"*46 + "|")

    for shape in shapes_iter:
        for section in SECTIONS:
            if section not in ONLY:
                continue
            key = f"{shape.name}::{section}"
            if RESUME and key in cache:
                tflops = cache[key]["tflops"]
                avg_ms = cache[key].get("avg_ms", 0.0)
            else:
                try:
                    avg_ms, tflops = bench_one(shape.M, shape.N, shape.K, section)
                except Exception as e:
                    log(f"  {key} failed: {e}")
                    avg_ms, tflops = 0.0, 0.0
                cache[key] = {"tflops": tflops, "avg_ms": avg_ms, "ts": time.time()}
                json.dump(cache, open(CACHE, "w"), indent=2)
            listed = shape.listed(section)
            ratio = tflops / listed if listed else 0
            print(f"| {shape.model:<18} {shape.op:<7} {shape.M:>5} {shape.N:>5} {shape.K:>5} | "
                  f"{section:<5} {listed:>7.1f} {tflops:>7.1f} {ratio*100:>5.0f}% {avg_ms:>5.2f} |")
            sys.stdout.flush()

    # Section means
    print()
    for s in SECTIONS:
        if s not in ONLY: continue
        vals = [cache[f"{r.name}::{s}"]["tflops"] for r in shapes_iter
                if f"{r.name}::{s}" in cache]
        listed = [r.listed(s) for r in shapes_iter]
        if vals and listed:
            mean_t = sum(vals) / len(vals)
            mean_l = sum(listed) / len(listed)
            ratio = mean_t / mean_l * 100 if mean_l else 0
            print(f"  Section {s} ({LAYOUT_BY_SECTION[s]}): "
                  f"triton avg = {mean_t:.1f} TFLOPS  vs  listed avg = {mean_l:.1f}  "
                  f"({ratio:.0f}% of listed)")

    print()
    print(f"Cache: {CACHE}")


if __name__ == "__main__":
    main()
