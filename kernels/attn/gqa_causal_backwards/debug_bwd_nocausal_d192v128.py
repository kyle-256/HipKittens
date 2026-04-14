#!/usr/bin/env python3
"""Non-causal dV check: kernel causal=false vs PyTorch full attention."""
import math
import torch
import tk_kernel_bkwd

torch.manual_seed(0)
B, N, H, H_KV = 1, 256, 1, 1
D_QK, D_V = 192, 128
dtype = torch.bfloat16

Q = torch.randn(B, H, N, D_QK, device="cuda", dtype=dtype)
K = torch.randn(B, H_KV, N, D_QK, device="cuda", dtype=dtype)
V = torch.randn(B, H_KV, N, D_V, device="cuda", dtype=dtype)
dO = torch.randn(B, H, N, D_V, device="cuda", dtype=dtype)

Q_tk = Q.transpose(1, 2).contiguous()
K_tk = K.transpose(1, 2).contiguous()
V_tk = V.transpose(1, 2).contiguous()
dO_tk = dO.transpose(1, 2).contiguous()

scale = 1.0 / math.sqrt(D_QK)
S = torch.matmul(Q.double(), K.double().transpose(-2, -1)) * scale
P = torch.softmax(S, dim=-1)
dV_ref = torch.matmul(P.transpose(-2, -1), dO.double())
dV_ref = dV_ref.transpose(1, 2).contiguous()

import tk_kernel_fwd
import tk_kernel_bkwd_prep

O_tk = torch.zeros(B, N, H, D_V, dtype=dtype, device="cuda")
L_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")
tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tk, L_tk)

delta_tk = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")
tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta_tk)

dQ_in = torch.zeros(B, H, N, D_QK, dtype=dtype, device="cuda").transpose(1, 2).contiguous()
dK_tk = torch.zeros(B, N, H_KV, D_QK, dtype=dtype, device="cuda")
dV_tk = torch.zeros(B, N, H_KV, D_V, dtype=dtype, device="cuda")

tk_kernel_bkwd.dispatch_bwd_combined(Q_tk, K_tk, V_tk, dO_tk, dQ_in, dK_tk, dV_tk, L_tk, delta_tk)

a = dV_ref.float().flatten()
b = dV_tk.float().flatten()
cos = (a @ b) / (a.norm() * b.norm() + 1e-30)
print("cos dV (non-causal ref vs kernel causal=false):", cos.item())
