#!/usr/bin/env python3
"""
Minimal dV sanity check: dV = P^T @ dO (causal softmax P) vs HipKittens backward dV only.

Does not modify the kernel — compares HK output to PyTorch float64 reference with the same
GQA expansion as test_python_d192v128.reference_fwd_bwd.

Usage (after building with matching ATTN_*):
  PYTHONPATH=. python test_dv_minimal_d192v128.py [B] [N] [H] [H_KV]

Build example:
  make asymmetric ATTN_D_QK=192 ATTN_D_V=128 ATTN_B=1 ATTN_H=1 ATTN_H_KV=1 ATTN_N=256
"""
import math
import os
import sys
import torch

torch.manual_seed(0)


def expand_kv_for_gqa(K, V, h_q, h_kv):
    g = h_q // h_kv
    return K.repeat_interleave(g, dim=1), V.repeat_interleave(g, dim=1)


def dV_reference_autograd(Q_bhnd, K_bhnd, V_bhnd, dO_bhnd, causal: bool):
    """Float64 autograd dV (matches test_python_d192v128.reference_fwd_bwd GQA semantics)."""
    B, H, N, D_QK = Q_bhnd.shape
    H_KV = K_bhnd.shape[1]
    q_ = Q_bhnd.detach().double().requires_grad_(True)
    k_ = K_bhnd.detach().double().requires_grad_(True)
    v_ = V_bhnd.detach().double().requires_grad_(True)
    k_exp, v_exp = expand_kv_for_gqa(k_, v_, H, H_KV)
    scale = 1.0 / math.sqrt(D_QK)
    S = torch.matmul(q_, k_exp.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, v_exp)
    O.backward(dO_bhnd.detach().double())
    return P, v_.grad


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    H_KV = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    causal = True
    D_QK, D_V = 192, 128
    dtype = torch.bfloat16

    import tk_kernel_fwd
    import tk_kernel_bkwd_prep
    import tk_kernel_bkwd

    Q = torch.randn(B, H, N, D_QK, device="cuda", dtype=dtype)
    K = torch.randn(B, H_KV, N, D_QK, device="cuda", dtype=dtype)
    V = torch.randn(B, H_KV, N, D_V, device="cuda", dtype=dtype)
    dO = torch.randn(B, H, N, D_V, device="cuda", dtype=dtype)

    P_ref, dV_ref = dV_reference_autograd(Q, K, V, dO, causal)
    print(
        f"P_ref (fp64 softmax): min={P_ref.min().item():.6g} max={P_ref.max().item():.6g} "
        f"any_nan={torch.isnan(P_ref).any().item()} any_inf={torch.isinf(P_ref).any().item()}"
    )

    Q_tk = Q.transpose(1, 2).contiguous()
    K_tk = K.transpose(1, 2).contiguous()
    V_tk = V.transpose(1, 2).contiguous()
    dO_tk = dO.transpose(1, 2).contiguous()

    O_tk = torch.zeros(B, N, H, D_V, dtype=dtype, device="cuda")
    L_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")
    tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tk, L_tk)

    delta_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")
    tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta_tk)

    dQ_in = torch.zeros(B, H, N, D_QK, dtype=dtype, device="cuda").transpose(1, 2).contiguous()
    dK_tk = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device="cuda")
    dV_tk = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device="cuda")

    tk_kernel_bkwd.dispatch_bwd_combined(
        Q_tk, K_tk, V_tk, dO_tk, dQ_in, dK_tk, dV_tk, L_tk, delta_tk
    )
    torch.cuda.synchronize()

    dV_ref_bnhd = dV_ref.transpose(1, 2).contiguous().to(dtype)

    def cos(a, b):
        a = a.float().flatten()
        b = b.float().flatten()
        return (a @ b) / (a.norm() * b.norm() + 1e-30)

    c = cos(dV_ref_bnhd, dV_tk)
    print(f"cos(dV_tk, fp64 autograd dV ref): {c.item():.6f}")
    sys.stdout.flush()
    if c.item() < 0.99:
        print(
            "Below 0.99: check P_ij / L indexing, mma_AtB operands, and transpose+store vs D=128 kernel.",
            file=sys.stderr,
        )
        return 1 if not os.environ.get("HK_DV_MINIMAL_ALLOW_FAIL") else 0
    print("PASS dV vs fp64 reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
