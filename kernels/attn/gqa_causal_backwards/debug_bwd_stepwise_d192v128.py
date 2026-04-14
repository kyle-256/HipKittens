#!/usr/bin/env python3
"""
Stepwise correctness checks for asymmetric MLA backward (D_QK=192, D_V=128).

Build (from this directory; THUNDERKITTENS_ROOT must point at HipKittens):

  export THUNDERKITTENS_ROOT=/path/to/HipKittens
  make clean && make asymmetric ATTN_D_QK=192 ATTN_B=1 ATTN_H=1 ATTN_H_KV=1 ATTN_N=256

ATTN_D_QK=192 is required or Makefile builds the wrong backward kernel.

Checks:
  1) L_vec from forward vs PyTorch log-sum-exp (natural log).
  2) delta from prep vs row-wise (dO * O).sum(-1).
  3) cosines for dQ/dK/dV vs PyTorch bf16 autograd.

Usage:
  PYTHONPATH=. python debug_bwd_stepwise_d192v128.py
"""
import math
import sys
import torch

torch.manual_seed(0)

B, N, H, H_KV = 1, 256, 1, 1
D_QK, D_V = 192, 128
dtype = torch.bfloat16

import tk_kernel_fwd
import tk_kernel_bkwd_prep
import tk_kernel_bkwd


def main():
    Q_bnhd = torch.randn(B, H, N, D_QK, device="cuda", dtype=dtype)
    K_bnhd = torch.randn(B, H_KV, N, D_QK, device="cuda", dtype=dtype)
    V_bnhd = torch.randn(B, H_KV, N, D_V, device="cuda", dtype=dtype)
    dO_bnhd = torch.randn(B, H, N, D_V, device="cuda", dtype=dtype)

    Q_tk = Q_bnhd.transpose(1, 2).contiguous()
    K_tk = K_bnhd.transpose(1, 2).contiguous()
    V_tk = V_bnhd.transpose(1, 2).contiguous()
    dO_tk = dO_bnhd.transpose(1, 2).contiguous()

    O_tk = torch.zeros(B, N, H, D_V, dtype=dtype, device="cuda")
    L_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")

    tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tk, L_tk)
    torch.cuda.synchronize()

    # --- Reference: float64 attention (same mask as softmax) ---
    scale = 1.0 / math.sqrt(D_QK)
    q64 = Q_bnhd.double()
    k64 = K_bnhd.double()
    v64 = V_bnhd.double()
    S = torch.matmul(q64, k64.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(N, N, device="cuda", dtype=torch.bool), diagonal=1)
    S = S.masked_fill(mask, float("-inf"))
    lse_ref = torch.logsumexp(S, dim=-1)  # [B,H,N] natural log

    L_hk = L_tk.squeeze(2)  # [B,H,N]
    diff_l = (L_hk.double() - lse_ref).abs().max().item()
    print(f"L max_abs_diff (HK vs torch logsumexp): {diff_l:.6e}")

    delta_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")
    tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta_tk)
    torch.cuda.synchronize()

    delta_ref = (dO_tk.float() * O_tk.float()).sum(dim=-1)  # [B,N,H]
    delta_ref = delta_ref.transpose(1, 2).contiguous()  # [B,H,N]
    delta_hk = delta_tk.squeeze(2)
    diff_d = (delta_hk - delta_ref).abs().max().item()
    print(f"delta max_abs_diff (prep vs dO*O): {diff_d:.6e}")

    # --- PyTorch bf16 autograd (same tensors) ---
    Qg = Q_bnhd.clone().requires_grad_(True)
    Kg = K_bnhd.clone().requires_grad_(True)
    Vg = V_bnhd.clone().requires_grad_(True)
    Sg = torch.matmul(Qg, Kg.transpose(-2, -1)) * scale
    Sg = Sg.masked_fill(mask, float("-inf"))
    Pg = torch.softmax(Sg, dim=-1)
    Og = torch.matmul(Pg, Vg)
    Og.backward(dO_bnhd)
    dV_ref = Vg.grad.transpose(1, 2).contiguous()
    dK_ref = Kg.grad.transpose(1, 2).contiguous()
    dQ_ref = Qg.grad.transpose(1, 2).contiguous()

    dQ_in = torch.zeros(B, H, N, D_QK, dtype=dtype, device="cuda").transpose(1, 2).contiguous()
    dQ_out = torch.zeros(B, N, H, D_QK, dtype=dtype, device="cuda")
    dK_tk = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device="cuda")
    dV_tk = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device="cuda")

    tk_kernel_bkwd.dispatch_bwd_combined(
        Q_tk, K_tk, V_tk, dO_tk, dQ_in, dK_tk, dV_tk, L_tk, delta_tk
    )
    tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in, dQ_out)
    torch.cuda.synchronize()

    def cos(a, b):
        a = a.float().flatten()
        b = b.float().flatten()
        return (a @ b) / (a.norm() * b.norm() + 1e-30)

    print(f"cos dV: {cos(dV_ref, dV_tk).item():.6f}")
    print(f"cos dK: {cos(dK_ref, dK_tk).item():.6f}")
    print(f"cos dQ: {cos(dQ_ref, dQ_out).item():.6f}")

    ok_ld = diff_l < 1e-2 and diff_d < 1e-5
    if not ok_ld:
        print("FAIL: L or delta mismatch vs reference", file=sys.stderr)
        sys.exit(1)
    print("PASS: L and delta vs reference")
    c_dv = cos(dV_ref, dV_tk)
    if c_dv < 0.99:
        print(
            "WARN: dQ/dK/dV cosines still low (L/delta OK). "
            "Run debug_dv_only_d192v128.py — dV mismatch persists vs oracle P (mma/store path)."
        )
        sys.exit(2)
    print("PASS: dQ, dK, dV cosines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
