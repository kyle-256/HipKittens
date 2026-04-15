#!/usr/bin/env python3
"""dK only diagnostics for D_QK=192, D_V=128 causal backward.

Compares the kernel dK against:
  1) float64 autograd reference
  2) a bf16/fp32 reference that mirrors the kernel's softmax path more closely

Also prints per-32-dim D_QK chunk cosines so it is easy to see whether the
error is localized to a specific Q/K tile range.
"""
import math
import sys
import torch

torch.manual_seed(0)

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
H = int(sys.argv[3]) if len(sys.argv) > 3 else 1
H_KV = int(sys.argv[4]) if len(sys.argv) > 4 else 1
D_QK, D_V = 192, 128
dtype = torch.bfloat16
gs = H // H_KV

sys.path.insert(0, ".")
import tk_kernel_bkwd
import tk_kernel_bkwd_prep
import tk_kernel_fwd


def cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b) / (a.norm() * b.norm() + 1e-30)


def main():
    scale = 1.0 / math.sqrt(D_QK)

    Q_bhnd = torch.randn(B, H, N, D_QK, device="cuda", dtype=dtype)
    K_bhnd = torch.randn(B, H_KV, N, D_QK, device="cuda", dtype=dtype)
    V_bhnd = torch.randn(B, H_KV, N, D_V, device="cuda", dtype=dtype)
    dO_bhnd = torch.randn(B, H, N, D_V, device="cuda", dtype=dtype)

    mask = torch.triu(torch.ones(N, N, device="cuda", dtype=torch.bool), diagonal=1)

    # Reference 1: float64 autograd.
    q64 = Q_bhnd.double().requires_grad_(True)
    k64 = K_bhnd.double().requires_grad_(True)
    v64 = V_bhnd.double().requires_grad_(True)
    ke64 = k64.repeat_interleave(gs, dim=1)
    ve64 = v64.repeat_interleave(gs, dim=1)
    S64 = torch.matmul(q64, ke64.transpose(-2, -1)) * scale
    S64 = S64.masked_fill(mask, float("-inf"))
    P64 = torch.softmax(S64, dim=-1)
    O64 = torch.matmul(P64, ve64)
    O64.backward(dO_bhnd.double())
    dK_ref64 = k64.grad.transpose(1, 2).contiguous()

    # Reference 2: bf16 matmul + fp32 softmax + fp32 dS path.
    K_exp = K_bhnd.repeat_interleave(gs, dim=1)
    V_exp = V_bhnd.repeat_interleave(gs, dim=1)
    S_bf = torch.matmul(Q_bhnd, K_exp.transpose(-2, -1)) * scale
    S_bf = S_bf.masked_fill(mask, float("-inf"))
    P_fp32 = torch.softmax(S_bf.float(), dim=-1)
    O_fp32 = torch.matmul(P_fp32, V_exp.float())
    delta_fp32 = (dO_bhnd.float() * O_fp32).sum(dim=-1, keepdim=True)
    dP_fp32 = torch.matmul(dO_bhnd.float(), V_exp.float().transpose(-2, -1))
    dS_fp32 = P_fp32 * (dP_fp32 - delta_fp32) * scale

    dK_ref_bf = torch.zeros(B, N, H_KV, D_QK, device="cuda", dtype=torch.float32)
    for kv_head in range(H_KV):
        q_heads = slice(kv_head * gs, (kv_head + 1) * gs)
        # Sum exact per-head contribution instead of averaging Q; keep semantics explicit.
        dK_acc = torch.zeros(B, N, D_QK, device="cuda", dtype=torch.float32)
        for qh in range(q_heads.start, q_heads.stop):
            dK_acc += torch.matmul(dS_fp32[:, qh].transpose(-2, -1), Q_bhnd[:, qh].float())
        dK_ref_bf[:, :, kv_head, :] = dK_acc

    # Kernel path.
    Q_tk = Q_bhnd.transpose(1, 2).contiguous()
    K_tk = K_bhnd.transpose(1, 2).contiguous()
    V_tk = V_bhnd.transpose(1, 2).contiguous()
    dO_tk = dO_bhnd.transpose(1, 2).contiguous()

    O_tk = torch.zeros(B, N, H, D_V, dtype=dtype, device="cuda")
    L_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")
    delta_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")

    tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tk, L_tk)
    tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta_tk)

    dQ_dummy = torch.zeros(B, H, N, D_QK, dtype=dtype, device="cuda").transpose(1, 2).contiguous()
    dK_tk = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device="cuda")
    dV_dummy = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device="cuda")
    tk_kernel_bkwd.dispatch_bwd_combined(
        Q_tk, K_tk, V_tk, dO_tk, dQ_dummy, dK_tk, dV_dummy, L_tk, delta_tk
    )
    torch.cuda.synchronize()

    print(f"cos(dK_kernel, dK_ref float64): {cos(dK_ref64, dK_tk).item():.6f}")
    print(f"cos(dK_kernel, dK_ref bf16-softmax): {cos(dK_ref_bf, dK_tk).item():.6f}")
    print(f"cos(dK_ref bf16-softmax, dK_ref float64): {cos(dK_ref_bf, dK_ref64).item():.6f}")

    print("\nPer-D chunk cosine (vs float64):")
    for start in range(0, D_QK, 32):
        end = start + 32
        c = cos(dK_ref64[..., start:end], dK_tk[..., start:end]).item()
        print(f"  D[{start:3d}:{end:3d}]: {c:.6f}")

    print("\nPer-KV chunk cosine (vs float64):")
    for start in range(0, N, 32):
        end = min(start + 32, N)
        c = cos(dK_ref64[:, start:end], dK_tk[:, start:end]).item()
        print(f"  KV[{start:4d}:{end:4d}]: {c:.6f}")


if __name__ == "__main__":
    main()
