#!/usr/bin/env python3
"""Benchmark BF16 GEMM kernels: TK vs hipBLAS (torch.mm) vs Triton.

Layout mapping for TK:
  (N, T) → RCR:  A[M,K] row, B_stored[N,K] row, C[M,N]
  (T, N) → CRR:  A_stored[K,M] row, B[K,N] row, C[M,N]
  (T, T) → RRR via C^T = B_stored * A_stored
"""
import torch
import sys
import time
import argparse
sys.path.insert(0, '.')
import bf16_gemm

# Try importing Triton kernel
try:
    sys.path.insert(0, '/workspace/triton_bench')
    from bf16_triton_gemm import triton_bf16_gemm
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton kernel not available, skipping Triton benchmarks")

torch.manual_seed(0)
dtype = torch.bfloat16
device = 'cuda'

# All unique shapes from user data: (M, N, K, rowMajorA, rowMajorB)
ALL_SHAPES = [
    (3584, 3584,  8192, 'N', 'T'),
    (3584, 3584, 16384, 'N', 'T'),
    (3584, 3584, 32768, 'N', 'T'),
    (3584, 18944,  8192, 'N', 'T'),
    (3584, 18944, 16384, 'N', 'T'),
    (3584, 18944, 32768, 'N', 'T'),
    (4096, 4096,  4096, 'N', 'T'),
    (4096, 4096,  4096, 'T', 'N'),
    (4096, 4096,  4096, 'T', 'T'),
    (4096, 4096,  6144, 'T', 'T'),
    (4096, 4096,  8192, 'N', 'T'),
    (4096, 4096, 11008, 'T', 'N'),
    (4096, 4096, 12288, 'T', 'T'),
    (4096, 4096, 14336, 'T', 'N'),
    (4096, 4096, 16384, 'N', 'T'),
    (4096, 4096, 22016, 'T', 'T'),
    (4096, 4096, 28672, 'T', 'T'),
    (4096, 4096, 32768, 'N', 'T'),
    (4096, 6144,  4096, 'T', 'N'),
    (4096, 8192,  8192, 'T', 'N'),
    (4096, 8192,  8192, 'T', 'T'),
    (4096, 8192, 10240, 'T', 'T'),
    (4096, 8192, 28672, 'T', 'N'),
    (4096, 8192, 57344, 'T', 'T'),
    (4096, 10240,  8192, 'T', 'N'),
    (4096, 11008,  4096, 'N', 'T'),
    (4096, 11008,  4096, 'T', 'T'),
    (4096, 11008,  8192, 'N', 'T'),
    (4096, 11008, 16384, 'N', 'T'),
    (4096, 12288,  4096, 'T', 'N'),
    (4096, 14336,  4096, 'N', 'T'),
    (4096, 14336,  4096, 'T', 'T'),
    (4096, 14336,  8192, 'N', 'T'),
    (4096, 14336, 16384, 'N', 'T'),
    (4096, 14336, 32768, 'N', 'T'),
    (4096, 22016,  4096, 'T', 'N'),
    (4096, 28672,  4096, 'T', 'N'),
    (4096, 28672,  8192, 'T', 'T'),
    (4096, 57344,  8192, 'T', 'N'),
    (4608, 3584,  8192, 'N', 'T'),
    (4608, 3584, 16384, 'N', 'T'),
    (4608, 3584, 32768, 'N', 'T'),
    (6144, 4096,  4096, 'N', 'T'),
    (6144, 4096,  8192, 'N', 'T'),
    (6144, 4096, 16384, 'N', 'T'),
    (6144, 4096, 32768, 'N', 'T'),
    (8192, 3584,  3584, 'T', 'N'),
    (8192, 3584,  3584, 'T', 'T'),
    (8192, 3584,  4608, 'T', 'T'),
    (8192, 3584, 18944, 'T', 'N'),
    (8192, 3584, 37888, 'T', 'T'),
    (8192, 4096,  4096, 'T', 'N'),
    (8192, 4096,  4096, 'T', 'T'),
    (8192, 4096,  6144, 'T', 'T'),
    (8192, 4096, 11008, 'T', 'N'),
    (8192, 4096, 12288, 'T', 'T'),
    (8192, 4096, 14336, 'T', 'N'),
    (8192, 4096, 22016, 'T', 'T'),
    (8192, 4096, 28672, 'T', 'T'),
    (8192, 4608,  3584, 'T', 'N'),
    (8192, 6144,  4096, 'T', 'N'),
]


def bench_fn(fn, num_warmup=50, num_iter=200):
    """Benchmark a function, return elapsed_ms per call."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(num_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iter


def bench_one(M, N, K, rA, rB):
    """Benchmark a single shape+layout combo for TK, hipBLAS, and Triton."""
    flop = 2 * M * N * K

    # Prepare tensors matching the layout
    if rA == 'N' and rB == 'T':
        layout = 'RCR'
        A_row = torch.randn(M, K, dtype=dtype, device=device)    # row-major [M,K]
        B_stored = torch.randn(N, K, dtype=dtype, device=device)  # row-major [N,K]
        C = torch.zeros(M, N, dtype=dtype, device=device)
        tk_fn = lambda: bf16_gemm.rcr(A_row, B_stored, C)
        # hipBLAS: C = A_row @ B_stored.T  (B_stored is [N,K], so B_stored.T is [K,N])
        hipblas_fn = lambda: torch.mm(A_row, B_stored.t())
        # Triton: expects A[M,K], B[K,N] contiguous
        B_col = B_stored.t().contiguous()  # [K,N]
        triton_fn = (lambda: triton_bf16_gemm(A_row, B_col)) if HAS_TRITON else None
    elif rA == 'T' and rB == 'N':
        layout = 'CRR'
        A_stored = torch.randn(K, M, dtype=dtype, device=device)  # row-major [K,M] = col-major A
        B = torch.randn(K, N, dtype=dtype, device=device)         # row-major [K,N]
        C = torch.zeros(M, N, dtype=dtype, device=device)         # row-major [M,N]
        tk_fn = lambda: bf16_gemm.crr(A_stored, B, C)
        # hipBLAS: C = A_stored.T @ B
        hipblas_fn = lambda: torch.mm(A_stored.t(), B)
        # Triton: expects A[M,K], B[K,N] contiguous
        A_row = A_stored.t().contiguous()  # [M,K]
        triton_fn = (lambda: triton_bf16_gemm(A_row, B)) if HAS_TRITON else None
    elif rA == 'T' and rB == 'T':
        layout = 'RRR'
        A_stored = torch.randn(K, M, dtype=dtype, device=device)  # stored [K,M]
        B_stored = torch.randn(N, K, dtype=dtype, device=device)  # stored [N,K]
        C_t = torch.zeros(N, M, dtype=dtype, device=device)
        # C[M,N] = A^T * B^T = (B*A)^T
        # C^T[N,M] = B_stored[N,K] * A_stored[K,M] → RRR
        tk_fn = lambda: bf16_gemm.rrr(B_stored, A_stored, C_t)
        # hipBLAS: C = A_stored.T @ B_stored.T
        hipblas_fn = lambda: torch.mm(A_stored.t(), B_stored.t())
        # Triton: expects A[M,K], B[K,N] contiguous
        A_row = A_stored.t().contiguous()  # [M,K]
        B_col = B_stored.t().contiguous()  # [K,N]
        triton_fn = (lambda: triton_bf16_gemm(A_row, B_col)) if HAS_TRITON else None
    else:
        raise ValueError(f"Unknown layout: rA={rA}, rB={rB}")

    # Benchmark TK
    tk_ms = bench_fn(tk_fn)
    tk_tflops = flop / (tk_ms * 1e9)

    # Benchmark hipBLAS
    hipblas_ms = bench_fn(hipblas_fn)
    hipblas_tflops = flop / (hipblas_ms * 1e9)

    # Benchmark Triton
    triton_tflops = 0
    if triton_fn is not None:
        triton_ms = bench_fn(triton_fn)
        triton_tflops = flop / (triton_ms * 1e9)

    return layout, tk_tflops, hipblas_tflops, triton_tflops


def filter_shapes(shapes, layout_filter=None):
    """Filter shapes by layout."""
    if layout_filter is None:
        return shapes
    mapping = {'rcr': ('N', 'T'), 'crr': ('T', 'N'), 'rrr': ('T', 'T')}
    rA, rB = mapping[layout_filter.lower()]
    return [(M, N, K, a, b) for (M, N, K, a, b) in shapes if a == rA and b == rB]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', choices=['rcr', 'crr', 'rrr'], default=None,
                        help='Only benchmark this layout')
    parser.add_argument('--no-triton', action='store_true', help='Skip Triton benchmarks')
    args = parser.parse_args()

    if args.no_triton:
        HAS_TRITON = False

    shapes = filter_shapes(ALL_SHAPES, args.layout)

    triton_hdr = 'Triton' if HAS_TRITON else ''
    triton_sep = '----------' if HAS_TRITON else ''
    triton_ratio_hdr = ' TK/Tri' if HAS_TRITON else ''

    print(f"{'M':>5s} {'N':>6s} {'K':>6s} {'A':>2s} {'B':>2s} {'Lay':>4s} "
          f"{'TK TFLOPS':>10s} {'hipBLAS':>8s} {'TK/hB':>6s}"
          f"{(' ' + triton_hdr):>8s}{triton_ratio_hdr:>8s}")
    print("-" * (62 + (18 if HAS_TRITON else 0)))

    sum_tk = sum_hb = sum_tr = 0
    count = 0
    count_rcr = count_other = 0
    sum_tk_rcr = sum_hb_rcr = 0
    sum_tk_other = sum_hb_other = 0

    for (M, N, K, rA, rB) in shapes:
        try:
            layout, tk, hb, tr = bench_one(M, N, K, rA, rB)
            ratio_hb = tk / hb if hb > 0 else 0
            sum_tk += tk; sum_hb += hb; sum_tr += tr; count += 1
            if layout == 'RCR':
                count_rcr += 1; sum_tk_rcr += tk; sum_hb_rcr += hb
            else:
                count_other += 1; sum_tk_other += tk; sum_hb_other += hb

            triton_str = ''
            if HAS_TRITON:
                ratio_tr = tk / tr if tr > 0 else 0
                triton_str = f" {tr:8.1f} {ratio_tr:7.2f}x"

            print(f"{M:5d} {N:6d} {K:6d} {rA:>2s} {rB:>2s} {layout:>4s} "
                  f"{tk:10.1f} {hb:8.1f} {ratio_hb:6.2f}x{triton_str}")
        except Exception as e:
            print(f"{M:5d} {N:6d} {K:6d} {rA:>2s} {rB:>2s}  ERROR: {e}")

    print("-" * (62 + (18 if HAS_TRITON else 0)))
    if count > 0:
        print(f"\nOverall Average ({count} shapes): TK={sum_tk/count:.1f}  hipBLAS={sum_hb/count:.1f}  "
              f"TK/hipBLAS={sum_tk/sum_hb:.2f}x")
    if count_rcr > 0:
        print(f"  RCR only  ({count_rcr:2d} shapes): TK={sum_tk_rcr/count_rcr:.1f}  hipBLAS={sum_hb_rcr/count_rcr:.1f}  "
              f"TK/hipBLAS={sum_tk_rcr/sum_hb_rcr:.2f}x")
    if count_other > 0:
        print(f"  CRR+RRR   ({count_other:2d} shapes): TK={sum_tk_other/count_other:.1f}  hipBLAS={sum_hb_other/count_other:.1f}  "
              f"TK/hipBLAS={sum_tk_other/sum_hb_other:.2f}x")
