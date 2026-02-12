#!/usr/bin/env python3
"""Quick correctness test for BF16 GEMM kernels: RCR / RRR / CRR."""

import torch
import sys
sys.path.insert(0, '.')
import bf16_gemm

torch.manual_seed(42)

def test_rcr(M, N, K, atol=1e-1, rtol=1e-2):
    """RCR: A[M,K] row-major, B stored as B^T [N,K] (col-major B), C[M,N]"""
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B_stored = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')  # [N,K]
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    bf16_gemm.rcr(A, B_stored, C)
    torch.cuda.synchronize()
    ref = A.float() @ B_stored.float().T  # C = A @ B^T
    err = (C.float() - ref).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    return max_err, mean_err

def test_rrr(M, N, K, atol=1e-1, rtol=1e-2):
    """RRR: A[M,K], B[K,N], C[M,N] all row-major."""
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    bf16_gemm.rrr(A, B, C)
    torch.cuda.synchronize()
    ref = A.float() @ B.float()
    err = (C.float() - ref).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    return max_err, mean_err

def test_crr(M, N, K, atol=1e-1, rtol=1e-2):
    """CRR: A col-major (stored as At[K,M]), B[K,N] row-major, C[M,N] row-major."""
    At = torch.randn(K, M, dtype=torch.bfloat16, device='cuda')  # At[K,M]
    B  = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    C  = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    bf16_gemm.crr(At, B, C)
    torch.cuda.synchronize()
    ref = At.float().T @ B.float()  # C[M,N] = At^T @ B
    err = (C.float() - ref).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    return max_err, mean_err

shapes = [
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (8192, 8192, 8192),
    (4096, 14336, 4096),
    (4096, 4096, 16384),
    (4096, 4096, 32768),
    (8192, 4096, 4096),
    (3584, 3584, 8192),
]

all_pass = True
for M, N, K in shapes:
    for name, fn in [("RCR", test_rcr), ("RRR", test_rrr), ("CRR", test_crr)]:
        max_err, mean_err = fn(M, N, K)
        # BF16 max error scales roughly as sqrt(K/64) * bf16_eps_contribution
        threshold = 0.1 * (K ** 0.5)
        ok = max_err < threshold
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name} M={M:5d} N={N:5d} K={K:5d}  {status}  max_err={max_err:.4f} mean_err={mean_err:.6f}")

print("\n" + ("ALL PASSED" if all_pass else "SOME FAILED"))
