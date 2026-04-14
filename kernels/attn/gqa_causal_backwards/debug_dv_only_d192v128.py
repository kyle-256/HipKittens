#!/usr/bin/env python3
"""dV only: PyTorch P from same Q,K (bf16, causal) vs HipKittens dV_out."""
import math
import sys
import torch

torch.manual_seed(0)
B, N, H, H_KV = 1, 256, 1, 1
D_QK, D_V = 192, 128
dtype = torch.bfloat16

sys.path.insert(0, ".")
import tk_kernel_bkwd
import tk_kernel_bkwd_prep
import tk_kernel_fwd

scale = 1.0 / math.sqrt(D_QK)

Q_bnhd = torch.randn(B, H, N, D_QK, device="cuda", dtype=dtype)
K_bnhd = torch.randn(B, H_KV, N, D_QK, device="cuda", dtype=dtype)
V_bnhd = torch.randn(B, H_KV, N, D_V, device="cuda", dtype=dtype)
dO_bnhd = torch.randn(B, H, N, D_V, device="cuda", dtype=dtype)

mask = torch.triu(torch.ones(N, N, device="cuda", dtype=torch.bool), diagonal=1)

# Reference 1: float64 scores
S64 = torch.matmul(Q_bnhd.double(), K_bnhd.double().transpose(-2, -1)) * scale
S64 = S64.masked_fill(mask, float("-inf"))
P64 = torch.softmax(S64, dim=-1)
dV_ref64 = torch.matmul(P64.transpose(-2, -1), dO_bnhd.double()).transpose(1, 2).contiguous()

# Reference 2: bf16 matmul like the kernel, then softmax in fp32
S_bf = torch.matmul(Q_bnhd, K_bnhd.transpose(-2, -1)) * scale
S_bf = S_bf.masked_fill(mask, float("-inf"))
P_bf = torch.softmax(S_bf.float(), dim=-1)
dV_ref_bf = torch.matmul(P_bf.transpose(-2, -1), dO_bnhd.float()).transpose(1, 2).contiguous()

Q_tk = Q_bnhd.transpose(1, 2).contiguous()
K_tk = K_bnhd.transpose(1, 2).contiguous()
V_tk = V_bnhd.transpose(1, 2).contiguous()
dO_tk = dO_bnhd.transpose(1, 2).contiguous()

O_tk = torch.zeros(B, N, H, D_V, dtype=dtype, device="cuda")
L_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")
tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tk, L_tk)

delta_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")
tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta_tk)

dQ_in = torch.zeros(B, H, N, D_QK, dtype=dtype, device="cuda").transpose(1, 2).contiguous()
dK_tk = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device="cuda")
dV_tk = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device="cuda")

tk_kernel_bkwd.dispatch_bwd_combined(Q_tk, K_tk, V_tk, dO_tk, dQ_in, dK_tk, dV_tk, L_tk, delta_tk)
torch.cuda.synchronize()

b = dV_tk.float().flatten()

def cos(a, b):
    a = a.float().flatten()
    return (a @ b) / (a.norm() * b.norm() + 1e-30)

print("cos(dV_kernel, dV_ref float64 P):", cos(dV_ref64, b).item())
print("cos(dV_kernel, dV_ref bf16 S softmax):", cos(dV_ref_bf, b).item())
